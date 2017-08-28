#!/usr/bin/python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pysc2.agents import random_agent
from pysc2.env import run_loop
from pysc2.env import sc2_env
from pysc2.tests import utils
from pysc2.lib import actions as sc2_actions
from pysc2.lib import features

from pysc2.lib import basetest
import gflags as flags
import sys

_NO_OP = sc2_actions.FUNCTIONS.no_op.id
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index

FLAGS = flags.FLAGS

class TestScripted(utils.TestCase):
  steps = 2000
  step_mul = 1

  def test_defeat_zerglings(self):
    FLAGS(sys.argv)

    with sc2_env.SC2Env(
        "DefeatZerglingsAndBanelings",
        step_mul=self.step_mul,
        visualize=True,
        game_steps_per_episode=self.steps * self.step_mul) as env:
      obs = env.step(actions=[sc2_actions.FunctionCall(_NO_OP, [])])
      player_relative = obs[0].observation["screen"][_PLAYER_RELATIVE]

      # Break Point!!
      print(player_relative)

      agent = random_agent.RandomAgent()
      run_loop.run_loop([agent], env, self.steps)

    self.assertEqual(agent.steps, self.steps)

if __name__ == "__main__":
  basetest.main()
