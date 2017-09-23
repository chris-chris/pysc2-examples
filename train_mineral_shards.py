import sys

import gflags as flags
from baselines import deepq
from pysc2.env import sc2_env
from pysc2.lib import actions
import os
from baselines import logger
from baselines.common import set_global_seeds

import deepq_mineral_shards


from baselines import bench
from common.vec_env.subproc_vec_env import SubprocVecEnv
from acktr.policies import CnnPolicy
from acktr import acktr_disc


_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_ALL = [0]
_NOT_QUEUED = [0]

step_mul = 8

FLAGS = flags.FLAGS
flags.DEFINE_string("map", "CollectMineralShards", "Name of a map to use to play.")
flags.DEFINE_string("algorithm", "acktr", "RL algorithm to use.")

def main():
  FLAGS(sys.argv)

  if(FLAGS.algorithm == "deepq"):

    with sc2_env.SC2Env(
        "DefeatZerglingsAndBanelings",
        step_mul=step_mul,
        visualize=True) as env:

      model = deepq.models.cnn_to_mlp(
        convs=[(16, 8, 4), (32, 4, 2)],
        hiddens=[256],
        dueling=True
      )

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
        prioritized_replay=True
      )
      act.save("mineral_shards.pkl")

  elif(FLAGS.algorithm == "acktr"):

    num_timesteps=int(40e6)

    num_timesteps //= 4

    seed=0
    num_cpu=2

    def make_env(rank):
      env = sc2_env.SC2Env(
        "DefeatZerglingsAndBanelings",
        step_mul=step_mul)
      #env.seed(seed + rank)
      #def _thunk():
        # env = sc2_env.SC2Env(
        #     FLAGS.map,
        #     step_mul=step_mul,
        #     visualize=True)
        # env.seed(seed + rank)
        #if logger.get_dir():
        #  env = bench.Monitor(env, os.path.join(logger.get_dir(), "{}.monitor.json".format(rank)))
      return env
      #return _thunk

    set_global_seeds(seed)
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])

    policy_fn = CnnPolicy
    acktr_disc.learn(policy_fn, env, seed, total_timesteps=num_timesteps, nprocs=num_cpu)

if __name__ == '__main__':
  main()
