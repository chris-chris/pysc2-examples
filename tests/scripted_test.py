#!/usr/bin/python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

from absl import flags
from absl.testing import absltest
from pysc2.agents import random_agent
from pysc2.env import run_loop
from pysc2.env import sc2_env
from pysc2.lib import actions as sc2_actions
from pysc2.lib import features
from pysc2.tests import utils

_NO_OP = sc2_actions.FUNCTIONS.no_op.id
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index

FLAGS = flags.FLAGS
FLAGS(sys.argv)

class TestScripted(utils.TestCase):
  steps = 2000
  step_mul = 1


  def test_defeat_zerglings(self):
    agent_format = sc2_env.AgentInterfaceFormat(
      feature_dimensions=sc2_env.Dimensions(
        screen=(32,32),
        minimap=(32,32),
      )
    )
    with sc2_env.SC2Env(
        map_name="DefeatZerglingsAndBanelings",
        step_mul=self.step_mul,
        visualize=True,
        agent_interface_format=[agent_format],
        game_steps_per_episode=self.steps * self.step_mul) as env:
      obs = env.step(actions=[sc2_actions.FunctionCall(_NO_OP, [])])
      player_relative = obs[0].observation["feature_screen"][_PLAYER_RELATIVE]

      # Break Point!!
      print(player_relative)

      agent = random_agent.RandomAgent()
      run_loop.run_loop([agent], env, self.steps)

    self.assertEqual(agent.steps, self.steps)

if __name__ == "__main__":
  absltest.main()
