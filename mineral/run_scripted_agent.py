from baselines import deepq
from pysc2.env import sc2_env
from pysc2.lib import actions
from pysc2.env import run_loop

from mineral.scripted_agent import CollectMineralShards

_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_ALL = [0]
_NOT_QUEUED = [0]

step_mul = 1
steps = 2000000

from absl import flags
import sys
FLAGS = flags.FLAGS
FLAGS(sys.argv)

def main():
  with sc2_env.SC2Env(
      map_name="CollectMineralShards",
      step_mul=step_mul,
      visualize=True,
      save_replay_episodes=10,
      replay_dir='replay',
      game_steps_per_episode=steps * step_mul
  ) as env:

    demo_replay = []

    agent = CollectMineralShards(env=env)
    run_loop.run_loop([agent], env, steps)


if __name__ == '__main__':
  main()
