import os.path as osp
import time
import numpy as np
import tensorflow as tf
from baselines import logger

from baselines.common import set_global_seeds

from a2c.model import Model
from a2c.runner import Runner

np.set_printoptions(threshold=np.inf)

def learn(policy,
          env,
          seed,
          total_timesteps=int(40e6),
          gamma=0.99,
          log_interval=1,
          nprocs=24,
          nscripts=12,
          nsteps=20,
          nstack=4,
          ent_coef=0.01,
          vf_coef=0.5,
          vf_fisher_coef=1.0,
          lr=0.25,
          max_grad_norm=0.01,
          kfac_clip=0.001,
          save_interval=None,
          lrschedule='linear',
          callback=None):
    tf.reset_default_graph()
    set_global_seeds(seed)

    nenvs = nprocs
    ob_space = (32, 32, 3)  # env.observation_space
    ac_space = (32, 32)
    make_model = lambda: Model(policy, ob_space, ac_space, nenvs,
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
    runner = Runner(
        env,
        model,
        nsteps=nsteps,
        nscripts=nscripts,
        nstack=nstack,
        gamma=gamma,
        callback=callback)
    nbatch = nenvs * nsteps
    tstart = time.time()
    # enqueue_threads = model.q_runner.create_threads(model.sess, coord=tf.train.Coordinator(), start=True)
    for update in range(1, total_timesteps // nbatch + 1):
        obs, states, td_targets, masks, actions, xy0, xy1, values = runner.run()

        model.policy_loss, model.value_loss, model.policy_entropy, model.policy_loss_xy0, model.policy_entropy_xy0, model.policy_loss_xy1, model.policy_entropy_xy1, = model.train(obs, states, td_targets, masks, actions, xy0, xy1, values)

        model.old_obs = obs
        nseconds = time.time() - tstart
        fps = int((update * nbatch) / nseconds)

        if save_interval and (update % save_interval == 0
                              or update == 1) and logger.get_dir():
            savepath = osp.join(logger.get_dir(), 'checkpoint%.5i' % update)
            print('Saving to', savepath)
            model.save(savepath)

    env.close()
