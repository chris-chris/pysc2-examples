import joblib
import tensorflow as tf
from baselines.a2c.utils import cat_entropy, find_trainable_variables, Scheduler

import nsml


class Model(object):
    def __init__(self,
                 policy,
                 ob_space,
                 ac_space,
                 nenvs,
                 total_timesteps,
                 nprocs=32,
                 nscripts=16,
                 nsteps=20,
                 nstack=4,
                 ent_coef=0.1,
                 vf_coef=0.5,
                 vf_fisher_coef=1.0,
                 lr=0.25,
                 max_grad_norm=0.001,
                 kfac_clip=0.001,
                 lrschedule='linear',
                 alpha=0.99,
                 epsilon=1e-5):
        config = tf.ConfigProto(allow_soft_placement=True, intra_op_parallelism_threads=nprocs, inter_op_parallelism_threads=nprocs)
        config.gpu_options.allow_growth = True
        self.sess = sess = tf.Session(config=config)
        nsml.bind(sess=sess)
        nbatch = nenvs * nsteps
        A = tf.placeholder(tf.int32, [nbatch])

        XY0 = tf.placeholder(tf.int32, [nbatch])
        XY1 = tf.placeholder(tf.int32, [nbatch])

        ADV = tf.placeholder(tf.float32, [nbatch])
        TD_TARGET = tf.placeholder(tf.float32, [nbatch])
        PG_LR = tf.placeholder(tf.float32, [])

        self.model = step_model = policy(sess, ob_space, ac_space, nenvs, 1, nstack, reuse=False)
        self.model2 = train_model = policy(sess, ob_space, ac_space, nenvs, nsteps, nstack, reuse=True)

        script_mask = tf.concat([tf.zeros([nscripts * nsteps, 1]), tf.ones([(nprocs - nscripts) * nsteps, 1])], axis=0)

        pi = train_model.pi
        pac_weight = script_mask * (tf.nn.softmax(pi) - 1.0) + 1.0
        pac_weight = tf.reduce_sum(pac_weight * tf.one_hot(A, depth=3), axis=1)
        neglogpac = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pi, labels=A)
        neglogpac *= tf.stop_gradient(pac_weight)

        xy0_mask = tf.cast(A, tf.float32)
        xy1_mask = tf.cast(A, tf.float32)

        condition0 = tf.equal(xy0_mask, 2)
        xy0_mask = tf.where(condition0, tf.ones(tf.shape(xy0_mask)), xy0_mask)
        xy0_mask = 1.0 - xy0_mask

        condition1 = tf.equal(xy1_mask, 2)
        xy1_mask = tf.where(condition1, tf.zeros(tf.shape(xy1_mask)), xy1_mask)

        pi_xy0 = train_model.pi_xy0
        pac_weight = script_mask * (tf.nn.softmax(pi_xy0) - 1.0) + 1.0
        pac_weight = tf.reduce_sum(pac_weight * tf.one_hot(XY0, depth=1024), axis=1)

        logpac_xy0 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pi_xy0, labels=XY0)
        logpac_xy0 *= tf.stop_gradient(pac_weight)
        logpac_xy0 *= tf.cast(xy0_mask, tf.float32)

        pi_xy1 = train_model.pi_xy1
        pac_weight = script_mask * (tf.nn.softmax(pi_xy1) - 1.0) + 1.0
        pac_weight = tf.reduce_sum(pac_weight * tf.one_hot(XY0, depth=1024), axis=1)

        logpac_xy1 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pi_xy1, labels=XY1)
        logpac_xy1 *= tf.stop_gradient(pac_weight)
        logpac_xy1 *= tf.cast(xy1_mask, tf.float32)

        pg_loss = tf.reduce_mean(ADV * neglogpac)
        pg_loss_xy0 = tf.reduce_mean(ADV * logpac_xy0)
        pg_loss_xy1 = tf.reduce_mean(ADV * logpac_xy1)

        vf_ = tf.squeeze(train_model.vf)

        vf_r = tf.concat([tf.ones([nscripts * nsteps, 1]), tf.zeros([(nprocs - nscripts) * nsteps, 1])], axis=0) * TD_TARGET
        vf_masked = vf_ * script_mask + vf_r

        vf_loss = tf.reduce_mean(mse(vf_masked, TD_TARGET))
        entropy_a = tf.reduce_mean(cat_entropy(train_model.pi))
        entropy_xy0 = tf.reduce_mean(cat_entropy(train_model.pi_xy0))
        entropy_xy1 = tf.reduce_mean(cat_entropy(train_model.pi_xy1))
        entropy = entropy_a + entropy_xy0 + entropy_xy1

        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef

        params = find_trainable_variables("model")
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, _ = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        trainer = tf.train.RMSPropOptimizer(learning_rate=lr, decay=alpha, epsilon=epsilon)
        _train = trainer.apply_gradients(grads)

        self.logits = train_model.pi

        self.params_common = params_common = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='model/common')
        self.params_xy0 = params_xy0 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='model/xy0') + params_common

        train_loss_xy0 = pg_loss_xy0 - entropy * ent_coef + vf_coef * vf_loss

        self.grads_check_xy0 = grads_xy0 = tf.gradients(train_loss_xy0, params_xy0)
        if max_grad_norm is not None:
            grads_xy0, _ = tf.clip_by_global_norm(grads_xy0, max_grad_norm)

        grads_xy0 = list(zip(grads_xy0, params_xy0))
        trainer_xy0 = tf.train.RMSPropOptimizer(learning_rate=lr, decay=alpha, epsilon=epsilon)
        _train_xy0 = trainer_xy0.apply_gradients(grads_xy0)

        self.params_xy1 = params_xy1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='model/xy1') + params_common
        train_loss_xy1 = pg_loss_xy1 - entropy * ent_coef + vf_coef * vf_loss

        self.grads_check_xy1 = grads_xy1 = tf.gradients(train_loss_xy1, params_xy1)
        if max_grad_norm is not None:
            grads_xy1, _ = tf.clip_by_global_norm(grads_xy1, max_grad_norm)

        grads_xy1 = list(zip(grads_xy1, params_xy1))
        trainer_xy1 = tf.train.RMSPropOptimizer(learning_rate=lr, decay=alpha, epsilon=epsilon)
        _train_xy1 = trainer_xy1.apply_gradients(grads_xy1)

        self.lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)

        def train(obs, states, td_targets, masks, actions, xy0, xy1, values):
            advs = td_targets - values
            for step in range(len(obs)):
                cur_lr = self.lr.value()

            td_map = {
                train_model.X: obs,
                A: actions,
                XY0: xy0,
                XY1: xy1,
                ADV: advs,
                TD_TARGET: td_targets,
                PG_LR: cur_lr
            }
            if states != []:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks

            policy_loss, value_loss, policy_entropy, _, policy_loss_xy0, policy_entropy_xy0, _, policy_loss_xy1, policy_entropy_xy1, _ = sess.run([pg_loss, vf_loss, entropy, _train, pg_loss_xy0, entropy_xy0, _train_xy0, pg_loss_xy1, entropy_xy1, _train_xy1], td_map)
            return policy_loss, value_loss, policy_entropy, policy_loss_xy0, policy_entropy_xy0, policy_loss_xy1, policy_entropy_xy1

        def save(save_path):
            ps = sess.run(params)
            joblib.dump(ps, save_path)

        def load(load_path):
            loaded_params = joblib.load(load_path)
            restores = []
            for p, loaded_p in zip(params, loaded_params):
                restores.append(p.assign(loaded_p))
            sess.run(restores)

        self.train = train
        self.save = save
        self.load = load
        self.train_model = train_model
        self.step_model = step_model
        self.step = step_model.step
        self.value = step_model.value
        self.initial_state = step_model.initial_state
        print("global_variables_initializer start")
        tf.global_variables_initializer().run(session=sess)
        print("global_variables_initializer complete")


def mse(pred, target):
    return tf.square(pred - target) / 2.