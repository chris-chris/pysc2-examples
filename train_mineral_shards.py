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

import threading
import time

_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_ALL = [0]
_NOT_QUEUED = [0]

step_mul = 8

FLAGS = flags.FLAGS
flags.DEFINE_string("map", "CollectMineralShards",
                    "Name of a map to use to play.")
start_time = datetime.datetime.now().strftime("%Y%m%d%H%M")
flags.DEFINE_string("log", "tensorboard", "logging type(stdout, tensorboard)")
flags.DEFINE_string("algorithm", "acktr", "RL algorithm to use.")
flags.DEFINE_integer("timesteps", 2000000, "Steps to train")
flags.DEFINE_float("exploration_fraction", 0.5, "Exploration Fraction")
flags.DEFINE_boolean("prioritized", True, "prioritized_replay")
flags.DEFINE_boolean("dueling", True, "dueling")
flags.DEFINE_float("lr", 0.0005, "Learning rate")
flags.DEFINE_integer("num_cpu", 4, "number of cpus")

PROJ_DIR = os.path.dirname(os.path.abspath(__file__))

max_mean_reward = 0
last_filename = ""

start_time = datetime.datetime.now().strftime("%Y%m%d%H%M")


def main():
  FLAGS(sys.argv)

  print("algorithm : %s" % FLAGS.algorithm)
  print("timesteps : %s" % FLAGS.timesteps)
  print("exploration_fraction : %s" % FLAGS.exploration_fraction)
  print("prioritized : %s" % FLAGS.prioritized)
  print("dueling : %s" % FLAGS.dueling)
  print("num_cpu : %s" % FLAGS.num_cpu)
  print("lr : %s" % FLAGS.lr)

  logdir = "tensorboard"
  if (FLAGS.algorithm == "deepq"):
    logdir = "tensorboard/mineral/%s/%s_%s_prio%s_duel%s_lr%s/%s" % (
      FLAGS.algorithm, FLAGS.timesteps, FLAGS.exploration_fraction,
      FLAGS.prioritized, FLAGS.dueling, FLAGS.lr, start_time)
  elif (FLAGS.algorithm == "acktr"):
    logdir = "tensorboard/mineral/%s/%s_num%s_lr%s/%s" % (FLAGS.algorithm,
                                                          FLAGS.timesteps,
                                                          FLAGS.num_cpu,
                                                          FLAGS.lr, start_time)

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
        "CollectMineralShards", step_mul=step_mul, visualize=True) as env:

      model = deepq.models.cnn_to_mlp(
        convs=[(16, 8, 4), (32, 4, 2)], hiddens=[256], dueling=True)

      act = deepq_mineral_shards.learn(
        env,
        q_func=model,
        num_actions=64,
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

  elif (FLAGS.algorithm == "acktr"):

    num_timesteps = int(40e6)

    num_timesteps //= 4

    seed = 0

    # def make_env(rank):
    #   # env = sc2_env.SC2Env(
    #   #   "CollectMineralShards",
    #   #   step_mul=step_mul)
    #   # return env
    #   #env.seed(seed + rank)
    #   def _thunk():
    #     env = sc2_env.SC2Env(
    #         map_name=FLAGS.map,
    #         step_mul=step_mul,
    #         visualize=True)
    #     #env.seed(seed + rank)
    #     if logger.get_dir():
    #      env = bench.Monitor(env, os.path.join(logger.get_dir(), "{}.monitor.json".format(rank)))
    #     return env
    #   return _thunk

    # agents = [Agent()
    #           for _ in range(num_cpu)]
    #
    # for agent in agents:
    #   time.sleep(1)
    #   agent.daemon = True
    #   agent.start()

    # agent_controller = AgentController(agents)

    #set_global_seeds(seed)
    env = SubprocVecEnv(FLAGS.num_cpu, FLAGS.map)

    policy_fn = CnnPolicy
    a2c.learn(
      policy_fn,
      env,
      seed,
      total_timesteps=num_timesteps,
      nprocs=FLAGS.num_cpu,
      ent_coef=0.5,
      callback=acktr_callback)


from pysc2.env import environment
import numpy as np


class Agent(threading.Thread):

  def __init__(self):
    threading.Thread.__init__(self)
    self.env = sc2_env.SC2Env(map_name=FLAGS.map, step_mul=step_mul)

    def run(self):
      print(threading.currentThread().getName(), self.receive_messages)

    def do_thing_with_message(self, message):
      if self.receive_messages:
        print(threading.currentThread().getName(),
              "Received %s".format(message))


class AgentController(object):

  def __init__(self, agents):
    self.agents = agents
    self.observation_space = (64, 64, 1)

  def step(self, actions):
    obs, rewards, dones, infos = [], [], [], []
    for idx, agent in enumerate(self.agents):
      result = agent.env.step(actions=actions[idx])
      ob = result[0].observation["screen"]
      reward = result[0].reward
      done = result[0].step_type == environment.StepType.LAST
      info = result[0].observation["available_actions"]
      obs.append(ob)
      rewards.append(reward)
      dones.append(done)
      infos.append(info)
    return np.stack(obs), np.stack(rewards), np.stack(dones), np.stack(infos)

  def close(self, actions):
    for idx, agent in enumerate(self.agents):
      agent.env.close()

  def reset(self):
    obs, rewards, dones, infos = [], [], [], []
    for idx, agent in enumerate(self.agents):
      result = agent.env.reset()
      ob = result[0].observation["screen"]
      reward = result[0].reward
      done = result[0].step_type == environment.StepType.LAST
      info = result[0].observation["available_actions"]
      obs.append(ob)
      rewards.append(reward)
      dones.append(done)
      infos.append(info)
    return np.stack(obs), np.stack(rewards), np.stack(dones), np.stack(infos)


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


def acktr_callback(locals, globals):
  global max_mean_reward, last_filename
  #pprint.pprint(locals)

  if ('mean_100ep_reward' in locals and locals['num_episodes'] >= 10 and
          locals['mean_100ep_reward'] > max_mean_reward):
    print("mean_100ep_reward : %s max_mean_reward : %s" %
          (locals['mean_100ep_reward'], max_mean_reward))

    if (not os.path.exists(os.path.join(PROJ_DIR, 'models/acktr/'))):
      try:
        os.mkdir(os.path.join(PROJ_DIR, 'models/'))
      except Exception as e:
        print(str(e))
      try:
        os.mkdir(os.path.join(PROJ_DIR, 'models/acktr/'))
      except Exception as e:
        print(str(e))

    if (last_filename != ""):
      os.remove(last_filename)
      print("delete last model file : %s" % last_filename)

    max_mean_reward = locals['mean_100ep_reward']
    model = locals['model']

    filename = os.path.join(
      PROJ_DIR, 'models/acktr/mineral_%s.pkl' % locals['mean_100ep_reward'])
    model.save(filename)
    print("save best mean_100ep_reward model to %s" % filename)
    last_filename = filename


if __name__ == '__main__':
  main()
