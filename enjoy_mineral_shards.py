import sc2_deepq

from baselines import deepq

import gflags as flags
import sys

from pysc2.env import sc2_env

import baselines.common.tf_util as U

step_mul = 16
steps = 200

FLAGS = flags.FLAGS

def main():
  FLAGS(sys.argv)
  with sc2_env.SC2Env(
      "CollectMineralShards",
      step_mul=step_mul,
      visualize=True,
      game_steps_per_episode=steps * step_mul) as env:

    model = deepq.models.cnn_to_mlp(
      convs=[(4, 8, 4), (2, 4, 2), (2, 3, 2)],
      hiddens=[128],
      dueling=True
    )

    def make_obs_ph(name):
      return U.BatchInput(env.observation_spec()["screen"], name=name)

    act_params = {
      'make_obs_ph': make_obs_ph,
      'q_func': model,
      'num_actions': 64 * 64,
    }

    act = sc2_deepq.load("sc2_mineral_shards_2mil.pkl", act_params=act_params)

    while True:
      obs, done = env.reset(), False
      episode_rew = 0
      while not done:
        env.render()
        obs, rew, done, _ = env.step(act(obs[None])[0])
        episode_rew += rew
      print("Episode reward", episode_rew)


if __name__ == '__main__':
  main()
