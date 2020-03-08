import random
import sys

from absl import flags
from pysc2.lib import actions as sc2_actions

from common.vec_env.subproc_vec_env import SubprocVecEnv

FLAGS = flags.FLAGS

def construct_action(marine_num, x, y):
  move_action = []
  # Base action is choosing a control group.
  # 4 == select_control_group
  move_action.append(
    sc2_actions.FunctionCall(4, [[0], [marine_num]]))
  # Right click.
  # 331 == Move
  move_action.append(
    sc2_actions.FunctionCall(331, [[0], [int(x), int(y)]]))
  return move_action


def get_position(env, marine_num):
  """Get position by selecting a unit.

  This function has a side effect, so we return rewards and dones.
  """
  select_action = construct_action(marine_num, -1, -1)
  _, rs, dones, _, _, _, selected, _ = env.step([select_action])

  xys = []
  for s in selected:
    pos = s.nonzero()
    x = pos[1][0]
    y = pos[2][0]
    xys.append((x, y))

  return xys, rs, dones


def main():
  FLAGS(sys.argv)
  env = SubprocVecEnv(1, 'CollectMineralShards')
  env.reset()
  total_reward = 0
  for _ in range(1000):
    marine = random.randrange(2)
    x = random.randrange(32)
    y = random.randrange(32)
    print('Move %d to (%d, %d)' % (marine, x, y))
    move_action = construct_action(marine, x, y)
    # This controls the APM.
    for _ in range(7):
      obs, rs, dones, _, _, _, selected, screens = env.step([move_action])
      total_reward += rs
    # Querying the position
    m_pos = {}
    m_pos['0'], rs, dones = get_position(env, 0)
    total_reward += rs
    m_pos['1'], rs, dones = get_position(env, 1)
    total_reward += rs

    print(rs)
    print(dones)
    print('Total reward: ', total_reward)
    print(m_pos)

  env.close()


if __name__ == '__main__':
  main()