import sys
import os

from absl import flags
from baselines import deepq
from pysc2.env import sc2_env
from pysc2.lib import actions
import os
from baselines import logger
from baselines.common import set_global_seeds

import deepq_mineral_shards
import datetime

from common.vec_env.subproc_vec_env import SubprocVecEnv
from a2c.policies import CnnPolicy
from a2c import a2c
from baselines.logger import Logger, TensorBoardOutputFormat, HumanOutputFormat

import random

import deepq_mineral_4way

import threading
import time

_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_ALL = [0]
_NOT_QUEUED = [0]

step_mul = 1

FLAGS = flags.FLAGS
flags.DEFINE_string("map", "CollectMineralShards",
                    "Name of a map to use to play.")
start_time = datetime.datetime.now().strftime("%Y%m%d%H%M")
flags.DEFINE_string("log", "tensorboard", "logging type(stdout, tensorboard)")
flags.DEFINE_string("algorithm", "a2c", "RL algorithm to use.")
flags.DEFINE_integer("timesteps", 2000000, "Steps to train")
flags.DEFINE_float("exploration_fraction", 0.5, "Exploration Fraction")
flags.DEFINE_boolean("prioritized", True, "prioritized_replay")
flags.DEFINE_boolean("dueling", True, "dueling")
flags.DEFINE_float("lr", 0.0005, "Learning rate")
flags.DEFINE_integer("num_agents", 4, "number of RL agents for A2C")
flags.DEFINE_integer("num_scripts", 4, "number of script agents for A2C")
flags.DEFINE_integer("nsteps", 20, "number of batch steps for A2C")

PROJ_DIR = os.path.dirname(os.path.abspath(__file__))

max_mean_reward = 0
last_filename = ""

start_time = datetime.datetime.now().strftime("%m%d%H%M")


def main():
  FLAGS(sys.argv)

  print("algorithm : %s" % FLAGS.algorithm)
  print("timesteps : %s" % FLAGS.timesteps)
  print("exploration_fraction : %s" % FLAGS.exploration_fraction)
  print("prioritized : %s" % FLAGS.prioritized)
  print("dueling : %s" % FLAGS.dueling)
  print("num_agents : %s" % FLAGS.num_agents)
  print("lr : %s" % FLAGS.lr)

  if(FLAGS.lr == 0):
    FLAGS.lr = random.uniform(0.00001, 0.001)
    
  print("random lr : %s" % FLAGS.lr)

  lr_round = round(FLAGS.lr, 8)

  logdir = "tensorboard"

  if (FLAGS.algorithm == "deepq-4way"):
    logdir = "tensorboard/mineral/%s/%s_%s_prio%s_duel%s_lr%s/%s" % (
      FLAGS.algorithm, FLAGS.timesteps, FLAGS.exploration_fraction,
      FLAGS.prioritized, FLAGS.dueling, lr_round, start_time)
  elif (FLAGS.algorithm == "deepq"):
    logdir = "tensorboard/mineral/%s/%s_%s_prio%s_duel%s_lr%s/%s" % (
      FLAGS.algorithm, FLAGS.timesteps, FLAGS.exploration_fraction,
      FLAGS.prioritized, FLAGS.dueling, lr_round, start_time)
  elif (FLAGS.algorithm == "a2c"):
    logdir = "tensorboard/mineral/%s/%s_n%s_s%s_nsteps%s/lr%s/%s" % (
      FLAGS.algorithm,
      FLAGS.timesteps,
      FLAGS.num_agents + FLAGS.num_scripts,
      FLAGS.num_scripts,
      FLAGS.nsteps,
      lr_round,
      start_time)

  if (FLAGS.log == "tensorboard"):
    Logger.DEFAULT \
      = Logger.CURRENT \
      = Logger(dir=None,
               output_formats=[TensorBoardOutputFormat(logdir)])

  elif (FLAGS.log == "stdout"):
    Logger.DEFAULT \
      = Logger.CURRENT \
      = Logger(dir=None,
               output_formats=[HumanOutputFormat(sys.stdout)])

  if (FLAGS.algorithm == "deepq"):

    with sc2_env.SC2Env(
        map_name="CollectMineralShards",
        step_mul=step_mul,
        visualize=True) as env:

      model = deepq.models.cnn_to_mlp(
        convs=[(16, 8, 4), (32, 4, 2)], hiddens=[256], dueling=True)

      act = deepq_mineral_shards.learn(
        env,
        q_func=model,
        num_actions=4,
        lr=1e-3,
        max_timesteps=20000000,
        buffer_size=10000,
        exploration_fraction=0.5,
        exploration_final_eps=0.01,
        train_freq=4,
        learning_starts=10000,
        target_network_update_freq=1000,
        gamma=0.99,
        prioritized_replay=True,
        callback=deepq_callback)
      act.save("mineral_shards.pkl")


  elif (FLAGS.algorithm == "deepq-4way"):

    with sc2_env.SC2Env(
        map_name="CollectMineralShards",
        step_mul=step_mul,
        screen_size_px=(32,32),
        minimap_size_px=(32,32),
        visualize=True) as env:

      model = deepq.models.cnn_to_mlp(
        convs=[(16, 8, 4), (32, 4, 2)], hiddens=[256], dueling=True)

      act = deepq_mineral_4way.learn(
        env,
        q_func=model,
        num_actions=4,
        lr=FLAGS.lr,
        max_timesteps=FLAGS.timesteps,
        buffer_size=10000,
        exploration_fraction=FLAGS.exploration_fraction,
        exploration_final_eps=0.01,
        train_freq=4,
        learning_starts=10000,
        target_network_update_freq=1000,
        gamma=0.99,
        prioritized_replay=True,
        callback=deepq_4way_callback)

      act.save("mineral_shards.pkl")

  elif (FLAGS.algorithm == "a2c"):

    num_timesteps = int(40e6)

    num_timesteps //= 4

    seed = 0

    env = SubprocVecEnv(FLAGS.num_agents + FLAGS.num_scripts, FLAGS.map)

    policy_fn = CnnPolicy
    a2c.learn(
      policy_fn,
      env,
      seed,
      total_timesteps=num_timesteps,
      nprocs=FLAGS.num_agents + FLAGS.num_scripts,
      nscripts=FLAGS.num_scripts,
      ent_coef=0.5,
      nsteps=FLAGS.nsteps,
      max_grad_norm=0.01,
      callback=a2c_callback)


from pysc2.env import environment
import numpy as np


def deepq_callback(locals, globals):
  #pprint.pprint(locals)
  global max_mean_reward, last_filename
  if ('done' in locals and locals['done'] == True):
    if ('mean_100ep_reward' in locals and locals['num_episodes'] >= 10 and
            locals['mean_100ep_reward'] > max_mean_reward):
      print("mean_100ep_reward : %s max_mean_reward : %s" %
            (locals['mean_100ep_reward'], max_mean_reward))

      if (not os.path.exists(os.path.join(PROJ_DIR, 'models/deepq/'))):
        try:
          os.mkdir(os.path.join(PROJ_DIR, 'models/'))
        except Exception as e:
          print(str(e))
        try:
          os.mkdir(os.path.join(PROJ_DIR, 'models/deepq/'))
        except Exception as e:
          print(str(e))

      if (last_filename != ""):
        os.remove(last_filename)
        print("delete last model file : %s" % last_filename)

      max_mean_reward = locals['mean_100ep_reward']
      act_x = deepq_mineral_shards.ActWrapper(locals['act_x'])
      act_y = deepq_mineral_shards.ActWrapper(locals['act_y'])

      filename = os.path.join(
        PROJ_DIR,
        'models/deepq/mineral_x_%s.pkl' % locals['mean_100ep_reward'])
      act_x.save(filename)
      filename = os.path.join(
        PROJ_DIR,
        'models/deepq/mineral_y_%s.pkl' % locals['mean_100ep_reward'])
      act_y.save(filename)
      print("save best mean_100ep_reward model to %s" % filename)
      last_filename = filename


def deepq_4way_callback(locals, globals):
  #pprint.pprint(locals)
  global max_mean_reward, last_filename
  if ('done' in locals and locals['done'] == True):
    if ('mean_100ep_reward' in locals and locals['num_episodes'] >= 10 and
            locals['mean_100ep_reward'] > max_mean_reward):
      print("mean_100ep_reward : %s max_mean_reward : %s" %
            (locals['mean_100ep_reward'], max_mean_reward))

      if (not os.path.exists(os.path.join(PROJ_DIR, 'models/deepq-4way/'))):
        try:
          os.mkdir(os.path.join(PROJ_DIR, 'models/'))
        except Exception as e:
          print(str(e))
        try:
          os.mkdir(os.path.join(PROJ_DIR, 'models/deepq-4way/'))
        except Exception as e:
          print(str(e))

      if (last_filename != ""):
        os.remove(last_filename)
        print("delete last model file : %s" % last_filename)

      max_mean_reward = locals['mean_100ep_reward']
      act = deepq_mineral_4way.ActWrapper(locals['act'])
      #act_y = deepq_mineral_shards.ActWrapper(locals['act_y'])

      filename = os.path.join(
        PROJ_DIR,
        'models/deepq-4way/mineral_%s.pkl' % locals['mean_100ep_reward'])
      act.save(filename)
      # filename = os.path.join(
      #   PROJ_DIR,
      #   'models/deepq/mineral_y_%s.pkl' % locals['mean_100ep_reward'])
      # act_y.save(filename)
      print("save best mean_100ep_reward model to %s" % filename)
      last_filename = filename


def a2c_callback(locals, globals):
  global max_mean_reward, last_filename
  #pprint.pprint(locals)

  if ('mean_100ep_reward' in locals and locals['num_episodes'] >= 10 and
          locals['mean_100ep_reward'] > max_mean_reward):
    print("mean_100ep_reward : %s max_mean_reward : %s" %
          (locals['mean_100ep_reward'], max_mean_reward))

    if (not os.path.exists(os.path.join(PROJ_DIR, 'models/a2c/'))):
      try:
        os.mkdir(os.path.join(PROJ_DIR, 'models/'))
      except Exception as e:
        print(str(e))
      try:
        os.mkdir(os.path.join(PROJ_DIR, 'models/a2c/'))
      except Exception as e:
        print(str(e))

    if (last_filename != ""):
      os.remove(last_filename)
      print("delete last model file : %s" % last_filename)

    max_mean_reward = locals['mean_100ep_reward']
    model = locals['model']

    filename = os.path.join(
      PROJ_DIR, 'models/a2c/mineral_%s.pkl' % locals['mean_100ep_reward'])
    model.save(filename)
    print("save best mean_100ep_reward model to %s" % filename)
    last_filename = filename


if __name__ == '__main__':
  main()
