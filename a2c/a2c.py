import os.path as osp
import time
import joblib
import numpy as np
import tensorflow as tf
from baselines import logger

from baselines.common import set_global_seeds, explained_variance

from baselines.acktr.utils import discount_with_dones
from baselines.acktr.utils import Scheduler, find_trainable_variables
from baselines.acktr.utils import cat_entropy, mse
# from a2c import kfac

from pysc2.env import environment
from pysc2.lib import actions as sc2_actions

from defeat_zerglings import common

_CONTROL_GROUP_RECALL = 0
_NOT_QUEUED = 0

# np.set_printoptions(threshold=np.inf)

class Model(object):

  def __init__(self, policy, ob_space, ac_space,
               nenvs,total_timesteps, nprocs=32, nscripts=16, nsteps=20,
               nstack=4, ent_coef=0.1, vf_coef=0.5, vf_fisher_coef=1.0,
               lr=0.25, max_grad_norm=0.001,
               kfac_clip=0.001, lrschedule='linear', alpha=0.99, epsilon=1e-5):
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=nprocs,
                            inter_op_parallelism_threads=nprocs)
    config.gpu_options.allow_growth = True
    self.sess = sess = tf.Session(config=config)
    #nact = ac_space.n
    nbatch = nenvs * nsteps
    A = tf.placeholder(tf.int32, [nbatch])

    XY0 = tf.placeholder(tf.int32, [nbatch])
    XY1 = tf.placeholder(tf.int32, [nbatch])

    ADV = tf.placeholder(tf.float32, [nbatch])
    R = tf.placeholder(tf.float32, [nbatch])
    PG_LR = tf.placeholder(tf.float32, [])
    VF_LR = tf.placeholder(tf.float32, [])

    self.model = step_model = policy(sess, ob_space, ac_space, nenvs, 1, nstack, reuse=False)
    self.model2 = train_model = policy(sess, ob_space, ac_space, nenvs, nsteps, nstack, reuse=True)

    # Policy 1 : Base Action : train_model.pi label = A

    script_mask = tf.concat([tf.zeros([nscripts * nsteps, 1]),tf.ones([(nprocs - nscripts) * nsteps, 1])],axis=0)

    pi = train_model.pi
    pac_weight = script_mask * (tf.nn.softmax(pi) - 1.0) + 1.0
    pi = pi * pac_weight
    neglogpac = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pi, labels=A)

    pi_xy0 = train_model.pi_xy0
    pac_weight = script_mask * (tf.nn.softmax(pi_xy0) - 1.0) + 1.0
    pi_xy0 = pi_xy0 * pac_weight
    logpac_xy0 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pi_xy0, labels=XY0)

    pi_xy1 = train_model.pi_xy1
    pac_weight = script_mask * (tf.nn.softmax(pi_xy1) - 1.0) + 1.0
    pi_xy1 = pi_xy1 * pac_weight
    logpac_xy1 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pi_xy1, labels=XY1)

    pg_loss = tf.reduce_mean(ADV * neglogpac)
    # logpac_xy0 = logpac_xy0 *  tf.cast(tf.equal(A, 2), tf.float32)
    pg_loss_xy0 = tf.reduce_mean(ADV * logpac_xy0)
    pg_loss_xy1 = tf.reduce_mean(ADV * logpac_xy1)
    # pg_loss_xy0 = pg_loss_xy0 * tf.cast(tf.equal(A, 2), tf.float32)
    # pg_loss_xy0 = pg_loss_xy0 - ent_coef * entropy_xy0

    vf_ = tf.squeeze(train_model.vf)

    vf_r = tf.concat([tf.ones([nscripts * nsteps, 1]),tf.zeros([(nprocs - nscripts) * nsteps, 1])],axis=0) * R
    vf_masked = vf_ * script_mask + vf_r



    #vf_mask[0:nscripts * nsteps] = R[0:nscripts * nsteps]

    vf_loss = tf.reduce_mean(mse(vf_masked, R))
    entropy = tf.reduce_mean(cat_entropy(train_model.pi))
    entropy_xy0 = tf.reduce_mean(cat_entropy(train_model.pi_xy0))
    entropy_xy1 = tf.reduce_mean(cat_entropy(train_model.pi_xy1))

    loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef

    params = find_trainable_variables("model")
    grads = tf.gradients(loss, params)
    if max_grad_norm is not None:
      grads, _ = tf.clip_by_global_norm(grads, max_grad_norm)
    grads = list(zip(grads, params))
    trainer = tf.train.RMSPropOptimizer(learning_rate=lr, decay=alpha, epsilon=epsilon)
    _train = trainer.apply_gradients(grads)

    self.logits = logits = train_model.pi


    # xy0

    self.params_common = params_common = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='model/common')
    self.params_xy0 = params_xy0 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='model/xy0') + params_common

    train_loss_xy0 = pg_loss_xy0 + vf_coef * vf_loss

    self.grads_check_xy0 = grads_xy0 = tf.gradients(train_loss_xy0, params_xy0)
    if max_grad_norm is not None:
      grads_xy0, _ = tf.clip_by_global_norm(grads_xy0, max_grad_norm)

    grads_xy0 = list(zip(grads_xy0, params_xy0))
    trainer_xy0 = tf.train.RMSPropOptimizer(learning_rate=lr, decay=alpha, epsilon=epsilon)
    _train_xy0 = trainer_xy0.apply_gradients(grads_xy0)


    # xy1

    self.params_xy1 = params_xy1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='model/xy1') + params_common

    train_loss_xy1 = pg_loss_xy1 + vf_coef * vf_loss

    self.grads_check_xy1 = grads_xy1 = tf.gradients(train_loss_xy1, params_xy1)
    if max_grad_norm is not None:
      grads_xy1, _ = tf.clip_by_global_norm(grads_xy1, max_grad_norm)

    grads_xy1 = list(zip(grads_xy1, params_xy1))
    trainer_xy1 = tf.train.RMSPropOptimizer(learning_rate=lr, decay=alpha, epsilon=epsilon)
    _train_xy1 = trainer_xy1.apply_gradients(grads_xy1)


    self.lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)

    def train(obs, states, rewards, masks, actions,
              xy0, xy1, values):
      advs = rewards - values
      for step in range(len(obs)):
        cur_lr = self.lr.value()

      td_map = {train_model.X:obs, A:actions,
                XY0:xy0, XY1:xy1, ADV:advs, R:rewards, PG_LR:cur_lr}
      if states != []:
        td_map[train_model.S] = states
        td_map[train_model.M] = masks

      policy_loss, value_loss, policy_entropy, _, \
      policy_loss_xy0, policy_entropy_xy0, _, \
      policy_loss_xy1, policy_entropy_xy1, _ = sess.run(
        [pg_loss, vf_loss, entropy, _train,
         pg_loss_xy0, entropy_xy0, _train_xy0,
         pg_loss_xy1, entropy_xy1, _train_xy1],
        td_map
      )
      print("policy_loss : ", policy_loss, " value_loss : ", value_loss, " policy_entropy : ", policy_entropy)

      print("policy_loss_xy0 : ", policy_loss_xy0, " policy_entropy_xy0 : ", policy_entropy_xy0)
      print("policy_loss_xy1 : ", policy_loss_xy1, " policy_entropy_xy1 : ", policy_entropy_xy1)

      return policy_loss, value_loss, policy_entropy, \
             policy_loss_xy0, policy_entropy_xy0, \
             policy_loss_xy1, policy_entropy_xy1

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

class Runner(object):

  def __init__(self, env, model, nsteps, nscripts, nstack, gamma, callback=None):
    self.env = env
    self.model = model
    nh, nw, nc = (32, 32, 1)
    self.nsteps = nsteps
    self.nscripts = nscripts
    self.nenv = nenv = env.num_envs
    self.batch_ob_shape = (nenv*nsteps, nh, nw, nc*nstack)
    self.batch_coord_shape = (nenv*nsteps, 32)
    self.obs = np.zeros((nenv, nh, nw, nc*nstack), dtype=np.uint8)
    self.available_actions = None
    self.base_act_mask = np.full((self.nenv, 2), 0, dtype=np.uint8)
    obs, rewards, dones, available_actions, army_counts, control_groups, selected, xy_per_marine = env.reset()
    self.xy_per_marine = [{} for _ in range(nenv)]
    for env_num, data in enumerate(xy_per_marine):
      self.xy_per_marine[env_num] = data
    self.army_counts = army_counts
    self.control_groups = control_groups
    self.selected = selected
    self.update_obs(obs) # (2,13,32,32)
    self.update_available(available_actions)
    self.gamma = gamma
    self.states = model.initial_state
    self.dones = [False for _ in range(nenv)]
    self.total_reward = [0.0 for _ in range(nenv)]
    self.episode_rewards = [0.0]
    self.episode_rewards_script = [0.0]
    self.episode_rewards_a2c = [0.0]
    self.episodes = 0
    self.steps = 0
    self.callback = callback

    self.action_queue = [[] for _ in range(nenv)]
    self.group_list = [[] for _ in range(nenv)]
    self.agent_state = ["IDLE" for _ in range(nenv)]
    self.dest_per_marine = [{} for _ in range(nenv)]
    
    self.group_id = [0 for _ in range(nenv)]

  def update_obs(self, obs): # (self.nenv, 32, 32, 2)
    obs = np.asarray(obs, dtype=np.int32).swapaxes(1, 2).swapaxes(2, 3)
    self.obs = np.roll(self.obs, shift=-1, axis=3)
    self.obs[:, :, :, -1:] = obs[:, :, :, :]
    # could not broadcast input array from shape (4,1,32,32) into shape (4,4,32)

  def update_available(self, _available_actions):
    #print("update_available : ", _available_actions)
    self.available_actions = _available_actions
    # avail = np.array([[0,1,2,3,4,7], [0,1,2,3,4,7]])
    self.base_act_mask = np.full((self.nenv, 2), 0, dtype=np.uint8)
    for env_num, list in enumerate(_available_actions):
      # print("env_num :", env_num, " list :", list)
      for action_num in list:
        # print("action_num :", action_num)
        if(action_num == 4):
          self.base_act_mask[env_num][0] = 1
          self.base_act_mask[env_num][1] = 1
        # elif(action_num == 331):
        #   self.base_act_mask[env_num][2] = 1


  def valid_base_action(self, base_actions):
    for env_num, list in enumerate(self.available_actions):
      avail = []
      for action_num in list:
        if(action_num == 4):
          avail.append(0)
          avail.append(1)
        # elif(action_num == 331):
        #   avail.append(2)

      if base_actions[env_num] not in avail:
        print("env_num", env_num, " argmax is not valid. random pick ", avail)
        base_actions[env_num] = np.random.choice(avail)

    return base_actions

  def trans_base_actions(self, base_actions):
    new_base_actions = np.copy(base_actions)
    for env_num, ba in enumerate(new_base_actions):
      if(ba==0):
        new_base_actions[env_num] = 4 # move marine control group 0
      elif(ba==1):
        new_base_actions[env_num] = 4 # move marine control group 1
      # elif(ba==2):
      #   new_base_actions[env_num] = 331 # move marine xy0

    return new_base_actions

  # def get_sub_act_mask(self, base_action_spec):
  #   sub1_act_mask = np.zeros((self.nenv, 2), np.int)
  #   sub2_act_mask = np.zeros((self.nenv, 10))
  #   sub3_act_mask = np.zeros((self.nenv, 500))
  #   for env_num, spec in enumerate(base_action_spec):
  #     for arg_idx, arg in enumerate(spec.args):
  #       if(len(arg.sizes) == 1 and arg.sizes[0] == 2):
  #         sub_act_len = spec.args[arg_idx].sizes[0]
  #         sub1_act_mask[env_num][0:sub_act_len-1] = 1
  #       elif(len(arg.sizes) == 1 and arg.sizes[0] == 500):
  #         sub_act_len = spec.args[arg_idx].sizes[0]
  #         sub3_act_mask[env_num][0:sub_act_len-1] = 1
  #       elif(len(arg.sizes) == 1):
  #         sub_act_len = spec.args[arg_idx].sizes[0]
  #         sub2_act_mask[env_num][0:sub_act_len-1] = 1
  #
  #   return sub1_act_mask, sub2_act_mask, sub3_act_mask

  def construct_action(self, base_actions, base_action_spec,
                       x0, y0, x1, y1):
    actions = []
    for env_num, spec in enumerate(base_action_spec):
      #print("spec", spec.args)
      args = []
      # for arg_idx, arg in enumerate(spec.args):
      #   #print("arg", arg)
      #   #print("arg.id", arg.id)
      #   if(arg.id==0): # screen (32,32) x0, y0
      #     args.append([int(x0[env_num]), int(y0[env_num])])
      #   # elif(arg.id==1): # minimap (32,32) x1, y1
      #   #   args.append([int(x1[env_num]), int(y1[env_num])])
      #   # elif(arg.id==2): # screen2 (32,32) x2, y2
      #   #   args.append([int(x2[env_num]), y2[env_num]])
      #   elif(arg.id==3): # pi3 queued (2)
      #     args.append([int(0)])
      #   elif(arg.id==4): # pi4 control_group_act (5)
      #     args.append([_CONTROL_GROUP_RECALL])
      #   elif(arg.id==5): # pi5 control_group_id 10
      #     args.append([int(base_actions[env_num])]) # 0 => cg 0 / 1 => cg 1
      #   # elif(arg.id==6): # pi6 select_point_act 4
      #   #   args.append([int(sub6[env_num])])
      #   # elif(arg.id==7): # pi7 select_add 2
      #   #   args.append([int(sub7[env_num])])
      #   # elif(arg.id==8): # pi8 select_unit_act 4
      #   #   args.append([int(sub8[env_num])])
      #   # elif(arg.id==9): # pi9 select_unit_id 500
      #   #   args.append([int(sub9[env_num])])
      #   # elif(arg.id==10): # pi10 select_worker 4
      #   #   args.append([int(sub10[env_num])])
      #   # elif(arg.id==11): # pi11 build_queue_id 10
      #   #   args.append([int(sub11[env_num])])
      #   # elif(arg.id==12): # pi12 unload_id 500
      #   #   args.append([int(sub12[env_num])])
      #   else:
      #     raise NotImplementedError("cannot construct this arg", spec.args)
      two_action = []
      if(base_actions[env_num]==0):
        two_action.append(sc2_actions.FunctionCall(4, [[_CONTROL_GROUP_RECALL], [0]]))
        two_action.append(sc2_actions.FunctionCall(331, [[_NOT_QUEUED], [int(x0[env_num]), y0[env_num]]]))

      elif(base_actions[env_num]==1):
        two_action.append(sc2_actions.FunctionCall(4, [[_CONTROL_GROUP_RECALL], [1]]))
        two_action.append(sc2_actions.FunctionCall(331, [[_NOT_QUEUED], [int(x1[env_num]), y1[env_num]]]))

      #action = sc2_actions.FunctionCall(a, args)
      actions.append(two_action)

    return actions

  def run(self):
    mb_obs, mb_rewards, mb_base_actions, \
    mb_xy0, mb_xy1, \
    mb_values, mb_dones \
      = [],[],[],[],[],[], []
      # ,[],[],[],[],[],[],[],[],[],[],[]

    mb_states = self.states
    for n in range(self.nsteps):
      # pi, pi2, x1, y1, x2, y2, v0
      pi1, pi_xy0, pi_xy1, values, states = self.model.step(self.obs, self.states, self.dones)

      pi1_noise = np.random.random_sample((self.nenv,2)) * 0.3
      # avail = self.env.available_actions()
      # print("pi1 : ", pi1)
      # print("pi1 * self.base_act_mask : ", pi1 * self.base_act_mask)
      # print("pi1 * self.base_act_mask + pi1_noise : ", pi1 * self.base_act_mask + pi1_noise)

      base_actions = np.argmax(pi1 * self.base_act_mask + pi1_noise, axis=1)
      xy0 = np.argmax(pi_xy0, axis=1)

      x0 = (xy0 % 32).astype(int)
      y0 = (xy0 / 32).astype(int)

      xy1 = np.argmax(pi_xy1, axis=1)
      x1 = (xy1 % 32).astype(int)
      y1 = (xy1 / 32).astype(int)

      # pi (2?, 524) * (2?, 524) masking
      # print("base_actions : ", base_actions)
      # print("base_action_spec : ", base_action_spec)
      # sub1_act_mask, sub2_act_mask, sub3_act_mask = self.get_sub_act_mask(base_action_spec)
      # print("base_actions : ", base_actions, "base_action_spec", base_action_spec,
      #       "sub1_act_mask :", sub1_act_mask, "sub2_act_mask :", sub2_act_mask, "sub3_act_mask :", sub3_act_mask)
      # sub3_actions = np.argmax(pi_sub3, axis=1) # pi (2?, 2) [1 0]
      # sub4_actions = np.argmax(pi_sub4, axis=1) # pi (2?, 5) [4 4]
      # sub5_actions = np.argmax(pi_sub5, axis=1) # pi (2?, 10) [1 4]
      # sub6_actions = np.argmax(pi_sub6, axis=1) # pi (2?, 4) [3 1]
      # sub7_actions = np.argmax(pi_sub7, axis=1) # pi (2?, 2)
      # sub8_actions = np.argmax(pi_sub8, axis=1) # pi (2?, 4)
      # sub9_actions = np.argmax(pi_sub9, axis=1) # pi (2?, 500)
      # sub10_actions = np.argmax(pi_sub10, axis=1) # pi (2?, 4)
      # sub11_actions = np.argmax(pi_sub11, axis=1) # pi (2?, 10)
      # sub12_actions = np.argmax(pi_sub12, axis=1) # pi (2?, 500)

      # Scripted Agent Hacking

      for env_num in range(self.nenv):
        if(env_num >= self.nscripts): # only for scripted agents
          continue

        ob = self.obs[env_num, :, :, :]
        # extra = ob[:,:,-1]
        # selected = ob[:, :, -2]
        player_relative = ob[:, :, -1]

        #if(common.check_group_list())
        self.group_list[env_num] = common.update_group_list2(self.control_groups[env_num])
        # if(len(self.action_queue[env_num]) == 0 and len(self.group_list[env_num]) == 0):
        #
        #   # Scripted Agent is only for even number agents
        #   self.action_queue[env_num] = common.group_init_queue(player_relative)

        if(len(self.action_queue[env_num]) == 0):

          self.action_queue[env_num], self.group_id[env_num], self.dest_per_marine[env_num], self.xy_per_marine[env_num] =\
            common.solve_tsp(player_relative,
                             self.selected[env_num][0],
                             self.group_list[env_num],
                             self.group_id[env_num],
                             self.dest_per_marine[env_num],
                             self.xy_per_marine[env_num])

        base_actions[env_num] = 0
        x0[env_num] = 0
        y0[env_num] = 0
        x1[env_num] = 0
        y1[env_num] = 0

        if(len(self.action_queue[env_num]) > 0):
          action = self.action_queue[env_num].pop(0)
          # print("action :", action)
          base_actions[env_num] = action.get("base_action",0)

          x0[env_num] = action.get("x0", 0)
          y0[env_num] = action.get("y0", 0)
          xy0[env_num] = y0[env_num] * 32 + x0[env_num]

          x1[env_num] = action.get("x1", 0)
          y1[env_num] = action.get("y1", 0)
          xy1[env_num] = y1[env_num] * 32 + x1[env_num]

      base_actions = self.valid_base_action(base_actions)
      # print("valid_base_actions : ", base_actions)
      new_base_actions = self.trans_base_actions(base_actions)
      # print("new_base_actions : ", new_base_actions)

      base_action_spec = self.env.action_spec(new_base_actions)

      actions = self.construct_action(base_actions, base_action_spec,
                                      # sub3_actions, sub4_actions, sub5_actions,
                                      # sub6_actions,
                                      # sub7_actions, sub8_actions,
                                      # sub9_actions, sub10_actions,
                                      # sub11_actions, sub12_actions,
                                      x0, y0, x1, y1
                                      # , x2, y2
                                      )

      mb_obs.append(np.copy(self.obs))
      mb_base_actions.append(base_actions)
      # mb_sub3_actions.append(sub3_actions)
      # mb_sub4_actions.append(sub4_actions)
      # mb_sub5_actions.append(sub5_actions)
      # mb_sub6_actions.append(sub6_actions)
      # mb_sub7_actions.append(sub7_actions)
      # mb_sub8_actions.append(sub8_actions)
      # mb_sub9_actions.append(sub9_actions)
      # mb_sub10_actions.append(sub10_actions)
      # mb_sub11_actions.append(sub11_actions)
      # mb_sub12_actions.append(sub12_actions)

      mb_xy0.append(xy0)
      # mb_y0.append(y0)
      mb_xy1.append(xy1)
      # mb_y1.append(y1)
      # mb_x2.append(x2)
      # mb_y2.append(y2)
      mb_values.append(values)
      mb_dones.append(self.dones)

      #print("final acitons : ", actions)
      obs, rewards, dones, available_actions, army_counts, control_groups, selected, xy_per_marine = self.env.step(actions=actions)
      self.army_counts = army_counts
      self.control_groups = control_groups
      self.selected = selected
      for env_num, data in enumerate(xy_per_marine):
        self.xy_per_marine[env_num] = data
      self.update_available(available_actions)

      self.states = states
      self.dones = dones
      for n, done in enumerate(dones):
        self.total_reward[n] += float(rewards[n])
        if done:
          self.obs[n] = self.obs[n]*0
          self.episodes += 1
          num_episodes = self.episodes
          self.episode_rewards.append(self.total_reward[n])

          if(n < self.nscripts): # scripted agents
            self.episode_rewards_script.append(self.total_reward[n])
            mean_100ep_reward_script = round(np.mean(self.episode_rewards_script[-101:-1]), 1)
            logger.record_tabular("reward script", self.total_reward[n])
            logger.record_tabular("mean reward script", mean_100ep_reward_script)
          else:
            self.episode_rewards_a2c.append(self.total_reward[n])
            mean_100ep_reward_a2c = round(np.mean(self.episode_rewards_a2c[-101:-1]), 1)
            logger.record_tabular("reward a2c", self.total_reward[n])
            logger.record_tabular("mean reward a2c", mean_100ep_reward_a2c)

          mean_100ep_reward = round(np.mean(self.episode_rewards[-101:-1]), 1)

          print("env %s done! reward : %s mean_100ep_reward : %s " % (n, self.total_reward[n], mean_100ep_reward))
          logger.record_tabular("reward", self.total_reward[n])
          logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
          logger.record_tabular("steps", self.steps)
          logger.record_tabular("episodes", self.episodes)


          logger.dump_tabular()

          self.total_reward[n] = 0
          self.group_list[n] = []

          model = self.model
          if self.callback is not None:
            self.callback(locals(), globals())

      print("rewards : ", rewards)
      print("self.total_reward :", self.total_reward)
      self.update_obs(obs)
      mb_rewards.append(rewards)
    mb_dones.append(self.dones)
    #batch of steps to batch of rollouts
    mb_obs = np.asarray(mb_obs, dtype=np.uint8).swapaxes(1, 0).reshape(self.batch_ob_shape)
    mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
    mb_base_actions = np.asarray(mb_base_actions, dtype=np.int32).swapaxes(1, 0)
    # mb_sub3_actions = np.asarray(mb_sub3_actions, dtype=np.int32).swapaxes(1, 0)
    # mb_sub4_actions = np.asarray(mb_sub4_actions, dtype=np.int32).swapaxes(1, 0)
    # mb_sub5_actions = np.asarray(mb_sub5_actions, dtype=np.int32).swapaxes(1, 0)
    # mb_sub6_actions = np.asarray(mb_sub6_actions, dtype=np.int32).swapaxes(1, 0)
    # mb_sub7_actions = np.asarray(mb_sub7_actions, dtype=np.int32).swapaxes(1, 0)
    # mb_sub8_actions = np.asarray(mb_sub8_actions, dtype=np.int32).swapaxes(1, 0)
    # mb_sub9_actions = np.asarray(mb_sub9_actions, dtype=np.int32).swapaxes(1, 0)
    # mb_sub10_actions = np.asarray(mb_sub10_actions, dtype=np.int32).swapaxes(1, 0)
    # mb_sub11_actions = np.asarray(mb_sub11_actions, dtype=np.int32).swapaxes(1, 0)
    # mb_sub12_actions = np.asarray(mb_sub12_actions, dtype=np.int32).swapaxes(1, 0)

    mb_xy0 = np.asarray(mb_xy0, dtype=np.int32).swapaxes(1, 0)
    # mb_y0 = np.asarray(mb_y0, dtype=np.int32).swapaxes(1, 0)
    mb_xy1 = np.asarray(mb_xy1, dtype=np.int32).swapaxes(1, 0)
    # mb_y1 = np.asarray(mb_y1, dtype=np.int32).swapaxes(1, 0)
    # mb_x2 = np.asarray(mb_x2, dtype=np.int32).swapaxes(1, 0)
    # mb_y2 = np.asarray(mb_y2, dtype=np.int32).swapaxes(1, 0)

    mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
    mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
    mb_masks = mb_dones[:, :-1]
    mb_dones = mb_dones[:, 1:]
    last_values = self.model.value(self.obs, self.states, self.dones).tolist()
    #discount/bootstrap off value fn
    for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
      rewards = rewards.tolist()
      dones = dones.tolist()
      if dones[-1] == 0:
        rewards = discount_with_dones(rewards+[value], dones+[0], self.gamma)[:-1]
      else:
        rewards = discount_with_dones(rewards, dones, self.gamma)
      mb_rewards[n] = rewards
    mb_rewards = mb_rewards.flatten()
    mb_base_actions = mb_base_actions.flatten()
    # mb_sub3_actions = mb_sub3_actions.flatten()
    # mb_sub4_actions = mb_sub4_actions.flatten()
    # mb_sub5_actions = mb_sub5_actions.flatten()
    # mb_sub6_actions = mb_sub6_actions.flatten()
    # mb_sub7_actions = mb_sub7_actions.flatten()
    # mb_sub8_actions = mb_sub8_actions.flatten()
    # mb_sub9_actions = mb_sub9_actions.flatten()
    # mb_sub10_actions = mb_sub10_actions.flatten()
    # mb_sub11_actions = mb_sub11_actions.flatten()
    # mb_sub12_actions = mb_sub12_actions.flatten()
    mb_xy0 = mb_xy0.flatten()
    # mb_y0 = mb_y0.flatten()
    mb_xy1 = mb_xy1.flatten()
    # mb_y1 = mb_y1.flatten()
    # mb_x2 = mb_x2.flatten()
    # mb_y2 = mb_y2.flatten()

    mb_values = mb_values.flatten()
    mb_masks = mb_masks.flatten()
    return mb_obs, mb_states, mb_rewards, mb_masks, \
           mb_base_actions, mb_xy0, mb_xy1, mb_values

def learn(policy, env, seed, total_timesteps=int(40e6),
          gamma=0.99, log_interval=1, nprocs=24, nscripts=12, nsteps=20,
          nstack=4, ent_coef=0.01, vf_coef=0.5, vf_fisher_coef=1.0,
          lr=0.25, max_grad_norm=0.01,
          kfac_clip=0.001, save_interval=None, lrschedule='linear',
          callback=None):
  tf.reset_default_graph()
  set_global_seeds(seed)

  nenvs = nprocs
  ob_space = (32, 32, 1) # env.observation_space
  ac_space = (32, 32)
  make_model = lambda : Model(policy, ob_space, ac_space, nenvs,
                              total_timesteps,
                              nprocs=nprocs,
                              nscripts=nscripts,
                              nsteps=nsteps,
                              nstack=nstack,
                              ent_coef=ent_coef,
                              vf_coef=vf_coef,
                              vf_fisher_coef=vf_fisher_coef,
                              lr=lr,
                              max_grad_norm=max_grad_norm,
                              kfac_clip=kfac_clip,
                              lrschedule=lrschedule)

  if save_interval and logger.get_dir():
    import cloudpickle
    with open(osp.join(logger.get_dir(), 'make_model.pkl'), 'wb') as fh:
      fh.write(cloudpickle.dumps(make_model))
  model = make_model()
  print("make_model complete!")
  runner = Runner(env, model, nsteps=nsteps, nscripts=nscripts, nstack=nstack, gamma=gamma, callback=callback)
  nbatch = nenvs*nsteps
  tstart = time.time()
  #enqueue_threads = model.q_runner.create_threads(model.sess, coord=tf.train.Coordinator(), start=True)
  for update in range(1, total_timesteps//nbatch+1):
    obs, states, rewards, masks, actions, xy0, xy1, values = runner.run()
    policy_loss, value_loss, policy_entropy, \
    policy_loss_xy0, policy_entropy_xy0, \
    policy_loss_xy1, policy_entropy_xy1, \
      = model.train(obs, states, rewards,
                    masks, actions,
                    xy0, xy1, values)

    model.old_obs = obs
    nseconds = time.time()-tstart
    fps = int((update*nbatch)/nseconds)
    if update % log_interval == 0 or update == 1:
      ev = explained_variance(values, rewards)
      logger.record_tabular("nupdates", update)
      logger.record_tabular("total_timesteps", update*nbatch)
      logger.record_tabular("fps", fps)
      logger.record_tabular("policy_entropy", float(policy_entropy))
      logger.record_tabular("policy_loss", float(policy_loss))

      logger.record_tabular("policy_loss_xy0", float(policy_loss_xy0))
      logger.record_tabular("policy_entropy_xy0", float(policy_entropy_xy0))
      logger.record_tabular("policy_loss_xy1", float(policy_loss_xy1))
      logger.record_tabular("policy_entropy_xy1", float(policy_entropy_xy1))
      # logger.record_tabular("policy_loss_y0", float(policy_loss_y0))
      # logger.record_tabular("policy_entropy_y0", float(policy_entropy_y0))

      logger.record_tabular("value_loss", float(value_loss))
      logger.record_tabular("explained_variance", float(ev))
      logger.dump_tabular()

    if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir():
      savepath = osp.join(logger.get_dir(), 'checkpoint%.5i'%update)
      print('Saving to', savepath)
      model.save(savepath)

  env.close()
