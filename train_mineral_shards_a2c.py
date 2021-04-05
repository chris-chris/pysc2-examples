import sys
import os
import datetime
import random

import absl
import baselines

import pysc2.lib
from baselines.logger import Logger, TensorBoardOutputFormat

import common.vec_env.subproc_vec_env
from a2c.a2c import learn
from a2c.policies import CnnPolicy

_MOVE_SCREEN = pysc2.lib.actions.FUNCTIONS.Move_screen.id
_SELECT_ARMY = pysc2.lib.actions.FUNCTIONS.select_army.id
_SELECT_ALL = [0]
_NOT_QUEUED = [0]

step_mul = 8

FLAGS = absl.flags.FLAGS

import sys

FLAGS(sys.argv)

absl.flags.DEFINE_string("map", "CollectMineralShards",
                         "Name of a map to use to play.")
start_time = datetime.datetime.now().strftime("%Y%m%d%H%M")
absl.flags.DEFINE_string("log", "tensorboard", "logging type(stdout, tensorboard)")
absl.flags.DEFINE_string("algorithm", "a2c", "RL algorithm to use.")
absl.flags.DEFINE_integer("timesteps", 2000000, "Steps to train")
absl.flags.DEFINE_float("exploration_fraction", 0.5, "Exploration Fraction")
absl.flags.DEFINE_boolean("prioritized", True, "prioritized_replay")
absl.flags.DEFINE_boolean("dueling", True, "dueling")
absl.flags.DEFINE_float("lr", 0.0005, "Learning rate")
absl.flags.DEFINE_integer("num_agents", 4, "number of RL agents for A2C")
absl.flags.DEFINE_integer("num_scripts", 0, "number of script agents for A2C")
absl.flags.DEFINE_integer("nsteps", 20, "number of batch steps for A2C")

PROJ_DIR = os.path.dirname(os.path.abspath(__file__))

max_mean_reward = 0
last_filename = ""
logdir = ""


def main():
    FLAGS(sys.argv)

    print("algorithm : %s" % FLAGS.algorithm)
    print("timesteps : %s" % FLAGS.timesteps)
    print("exploration_fraction : %s" % FLAGS.exploration_fraction)
    print("prioritized : %s" % FLAGS.prioritized)
    print("dueling : %s" % FLAGS.dueling)
    print("num_agents : %s" % FLAGS.num_agents)
    print("lr : %s" % FLAGS.lr)

    if FLAGS.lr == 0:
        FLAGS.lr = random.uniform(0.00001, 0.001)
    print("random lr : %s" % FLAGS.lr)
    lr_round = round(FLAGS.lr, 8)

    logdir = "tensorboard/mineral/%s/%s_n%s_s%s_nsteps%s/lr%s/%s" % (
        FLAGS.algorithm, FLAGS.timesteps, FLAGS.num_agents + FLAGS.num_scripts, FLAGS.num_scripts, FLAGS.nsteps,
        lr_round,
        start_time)

    Logger.DEFAULT = Logger.CURRENT = Logger(dir=None, output_formats=[TensorBoardOutputFormat(logdir)])

    num_timesteps = int(40e6)
    num_timesteps //= 4

    seed = 0

    env = common.vec_env.subproc_vec_env.SubprocVecEnv(FLAGS.num_agents + FLAGS.num_scripts, FLAGS.num_scripts,
                                                       FLAGS.map)

    policy_fn = CnnPolicy
    learn(
        policy_fn,
        env,
        seed,
        total_timesteps=num_timesteps,
        gamma=0.99,
        nprocs=FLAGS.num_agents + FLAGS.num_scripts,
        nscripts=FLAGS.num_scripts,
        nsteps=FLAGS.nsteps,
        ent_coef=0.5,
        vf_coef=0.5,
        lr=0.25,
        max_grad_norm=0.01,
        save_interval=1000,
        lrschedule='linear',
        callback=a2c_callback)


def a2c_callback(locals, globals):
    global max_mean_reward, last_filename

    baselines.logger.record_tabular("mean 100 episode reward a2c", locals['mean_100ep_reward_a2c'])
    baselines.logger.record_tabular("num_episodes", locals['num_episodes'])
    baselines.logger.record_tabular("environment_number", locals['env_num'])
    baselines.logger.record_tabular("policy_loss", locals['model'].policy_loss)
    baselines.logger.record_tabular("policy_loss_xy0", locals['model'].policy_loss_xy0)
    baselines.logger.record_tabular("policy_loss_xy1", locals['model'].policy_loss_xy1)
    baselines.logger.record_tabular("policy_entropy", locals['model'].policy_entropy)
    baselines.logger.record_tabular("policy_entropy_xy0", locals['model'].policy_entropy_xy0)
    baselines.logger.record_tabular("policy_entropy_xy1", locals['model'].policy_entropy_xy1)
    baselines.logger.record_tabular("learning_rate_N", locals['model'].lr.n)
    baselines.logger.record_tabular("learning_rate_V", locals['model'].lr.v)
    baselines.logger.record_tabular("value_loss", locals['model'].value_loss)
    baselines.logger.record_tabular("done", locals['done'])

    if 'mean_100ep_reward_a2c' in locals and locals['num_episodes'] >= 10 and locals[
        'mean_100ep_reward_a2c'] > max_mean_reward:
        print("mean_100ep_reward_a2c : %s max_mean_reward : %s" %
              (locals['mean_100ep_reward_a2c'], max_mean_reward))
        max_mean_reward = locals['mean_100ep_reward_a2c']
        baselines.logger.record_tabular("max_mean_reward", max_mean_reward)

        if not os.path.exists(os.path.join(PROJ_DIR, 'models/a2c/')):
            try:
                os.mkdir(os.path.join(PROJ_DIR, 'models/'))
            except Exception as e:
                print(str(e))
            try:
                os.mkdir(os.path.join(PROJ_DIR, 'models/a2c/'))
            except Exception as e:
                print(str(e))

        if last_filename != "":
            os.remove(last_filename)
            print("delete last model file : %s" % last_filename)

        model = locals['model']

        filename = os.path.join(PROJ_DIR, 'models/a2c/mineral_%s.pkl' % locals['mean_100ep_reward_a2c'])
        model.save(filename)
        print("save best mean_100ep_reward model to %s" % filename)
        last_filename = filename

    baselines.logger.dump_tabular()


if __name__ == '__main__':
    main()
