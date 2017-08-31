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

class MarineAgent(base_agent.BaseAgent):
  """A random agent for starcraft."""
  demo_replay = []

  def __init__(self, env):
    self.env = env

  def step(self, obs):
    super(MarineAgent, self).step(obs)

    #1. Select marine!
    obs, screen, player = common.select_marine(self.env, [obs])

    player_relative = obs[0].observation["screen"][_PLAYER_RELATIVE]

    enemy_y, enemy_x = (player_relative == _PLAYER_HOSTILE).nonzero()


    #2. Run away from nearby enemy
    closest, min_dist = None, None

    if(len(player) == 2):
      for p in zip(enemy_x, enemy_y):
        dist = np.linalg.norm(np.array(player) - np.array(p))
        if not min_dist or dist < min_dist:
          closest, min_dist = p, dist


    #3. Sparse!
    friendly_y, friendly_x = (player_relative == _PLAYER_FRIENDLY).nonzero()

    closest_friend, min_dist_friend = None, None
    if(len(player) == 2):
      for p in zip(friendly_x, friendly_y):
        dist = np.linalg.norm(np.array(player) - np.array(p))
        if not min_dist_friend or dist < min_dist_friend:
          closest_friend, min_dist_friend = p, dist

    if(min_dist != None and min_dist <= 7):

      obs, new_action = common.marine_action(self.env, obs, player, 2)

    elif(min_dist_friend != None and min_dist_friend <= 3):

      sparse_or_attack = np.random.randint(0,2)

      obs, new_action = common.marine_action(self.env, obs, player, sparse_or_attack)

    else:

      obs, new_action = common.marine_action(self.env, obs, player, 1)

    return new_action[0]
