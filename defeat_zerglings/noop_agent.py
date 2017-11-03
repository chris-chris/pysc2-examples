"""A random agent for starcraft."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy

from pysc2.agents import base_agent
from pysc2.lib import actions

from pysc2.lib import actions as sc2_actions
from pysc2.lib import features
from pysc2.lib import actions

from defeat_zerglings import common

import numpy as np

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index

_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_SELECTED = features.SCREEN_FEATURES.selected.index
_PLAYER_FRIENDLY = 1
_PLAYER_NEUTRAL = 3  # beacon/minerals
_PLAYER_HOSTILE = 4
_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_UNIT_ID = 1

_CONTROL_GROUP_SET = 1
_CONTROL_GROUP_RECALL = 0

_SELECT_CONTROL_GROUP = actions.FUNCTIONS.select_control_group.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_UNIT = actions.FUNCTIONS.select_unit.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id

_NOT_QUEUED = [0]
_SELECT_ALL = [0]


class NOOPAgent(base_agent.BaseAgent):
  """A random agent for starcraft."""
  demo_replay = []

  def __init__(self, env):
    self.env = env

  def step(self, obs):
    super(NOOPAgent, self).step(obs)

    return sc2_actions.FunctionCall(_NO_OP, [])
