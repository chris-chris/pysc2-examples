import sc2_deepq
from baselines import deepq

from pysc2.env import sc2_env
from pysc2.lib import actions

import gflags as flags
import sys
from s2clientprotocol import sc2api_pb2


_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_ALL = [0]
_NOT_QUEUED = [0]

step_mul = 8
steps = 400

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

    act = sc2_deepq.learn(
      env,
      q_func=model,
      num_actions=4,
      lr=1e-4,
      max_timesteps=2000000,
      buffer_size=10000,
      exploration_fraction=0.1,
      exploration_final_eps=0.01,
      train_freq=4,
      learning_starts=10000,
      target_network_update_freq=1000,
      gamma=0.99,
      prioritized_replay=True
    )
    act.save("sc2_mineral_shards_2mil.pkl")


if __name__ == '__main__':
  main()
