import sys

import numpy as np
from absl import flags
from pysc2.env import environment
from pysc2.env import sc2_env
from pysc2.lib import actions
from pysc2.lib import actions as sc2_actions
from pysc2.lib import features

import deepq_mineral_shards
from baselines_legacy import BatchInput, cnn_to_mlp

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_FRIENDLY = 1
_PLAYER_NEUTRAL = 3  # beacon/minerals
_PLAYER_HOSTILE = 4
_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_NOT_QUEUED = [0]
_SELECT_ALL = [0]

step_mul = 16
steps = 400

FLAGS = flags.FLAGS


def main():
  FLAGS(sys.argv)
  AGENT_INTERFACE_FORMAT = sc2_env.AgentInterfaceFormat(
    feature_dimensions=sc2_env.Dimensions(screen=16, minimap=16))
  with sc2_env.SC2Env(
      map_name="CollectMineralShards",
      players=[sc2_env.Agent(sc2_env.Race.terran)],
      step_mul=step_mul,
      visualize=True,
      agent_interface_format=AGENT_INTERFACE_FORMAT) as env:

    # model = cnn_to_mlp(
    #   convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
    #   hiddens=[256],
    #   dueling=True)

    # def make_obs_ph(name):
    #   return BatchInput((1, 64, 64), name=name)

    model = cnn_to_mlp(
        convs=[(16, 8, 4), (32, 4, 2)], hiddens=[256], dueling=True)

    def make_obs_ph(name):
      return BatchInput((1, 16, 16), name=name)

    # Using deepq_x here instead of deepq for agent x
    act_params = {
      'make_obs_ph': make_obs_ph,
      'q_func': model,
      'num_actions': 16,
      'scope': "deepq_x"
    }

    # This needs to be the saved model for deepq_x
    # You can change the scope to deepq_y for agent y
    act = deepq_mineral_shards.load(
        "mineral_shards.pkl", act_params=act_params)

    while True:

      obs = env.reset()
      episode_rew = 0

      done = False

      step_result = env.step(actions=[
        sc2_actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])
      ])

      while not done:

        player_relative = step_result[0].observation["feature_screen"][
          _PLAYER_RELATIVE]

        obs = player_relative

        player_y, player_x = (
            player_relative == _PLAYER_FRIENDLY).nonzero()
        player = [int(player_x.mean()), int(player_y.mean())]

        if (player[0] > 32):
          obs = shift(LEFT, player[0] - 32, obs)
        elif (player[0] < 32):
          obs = shift(RIGHT, 32 - player[0], obs)

        if (player[1] > 32):
          obs = shift(UP, player[1] - 32, obs)
        elif (player[1] < 32):
          obs = shift(DOWN, 32 - player[1], obs)

        action = act(np.expand_dims(obs[None], axis=0))[0]
        coord = [player[0], player[1]]

        if (action == 0):  # UP

          if (player[1] >= 16):
            coord = [player[0], player[1] - 16]
          elif (player[1] > 0):
            coord = [player[0], 0]

        elif (action == 1):  # DOWN

          if (player[1] <= 47):
            coord = [player[0], player[1] + 16]
          elif (player[1] > 47):
            coord = [player[0], 63]

        elif (action == 2):  # LEFT

          if (player[0] >= 16):
            coord = [player[0] - 16, player[1]]
          elif (player[0] < 16):
            coord = [0, player[1]]

        elif (action == 3):  # RIGHT

          if (player[0] <= 47):
            coord = [player[0] + 16, player[1]]
          elif (player[0] > 47):
            coord = [63, player[1]]

        new_action = [
          sc2_actions.FunctionCall(_MOVE_SCREEN,
                                   [_NOT_QUEUED, coord])
        ]

        step_result = env.step(actions=new_action)

        rew = step_result[0].reward
        done = step_result[0].step_type == environment.StepType.LAST

        episode_rew += rew
      print("Episode reward", episode_rew)


UP, DOWN, LEFT, RIGHT = 'up', 'down', 'left', 'right'


def shift(direction, number, matrix):
  ''' shift given 2D matrix in-place the given number of rows or columns
    in the specified (UP, DOWN, LEFT, RIGHT) direction and return it
'''
  if direction in (UP):
    matrix = np.roll(matrix, -number, axis=0)
    matrix[number:, :] = -2
    return matrix
  elif direction in (DOWN):
    matrix = np.roll(matrix, number, axis=0)
    matrix[:number, :] = -2
    return matrix
  elif direction in (LEFT):
    matrix = np.roll(matrix, -number, axis=1)
    matrix[:, number:] = -2
    return matrix
  elif direction in (RIGHT):
    matrix = np.roll(matrix, number, axis=1)
    matrix[:, :number] = -2
    return matrix
  else:
    return matrix


if __name__ == '__main__':
  main()
