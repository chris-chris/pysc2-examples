"""Scripted agents."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

from defeat_zerglings import common

import random

#from mineral.tsp import travelling_salesman
from mineral.tsp2 import multistart_localsearch, mk_matrix, distL2

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_FRIENDLY = 1
_PLAYER_NEUTRAL = 3  # beacon/minerals
_PLAYER_HOSTILE = 4
_SELECTED = features.SCREEN_FEATURES.selected.index

_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id

_CONTROL_GROUP_SET = 1
_CONTROL_GROUP_RECALL = 0

_SELECT_CONTROL_GROUP = actions.FUNCTIONS.select_control_group.id

_NOT_QUEUED = [0]
_SELECT_ALL = [0]


class CollectMineralShards(base_agent.BaseAgent):
  """An agent specifically for solving the CollectMineralShards map."""

  def __init__(self, env):
    super(CollectMineralShards, self).__init__()
    player = []
    self.env = env
    self.group_id = 0
    self.group_list = []
    self.dest_per_marine = {}

  def step(self, obs):
    super(CollectMineralShards, self).step(obs)

    if (len(self.group_list) == 0
        or common.check_group_list(self.env, [obs])):
      print("init group list")
      obs, xy_per_marine = common.init(self.env, [obs])
      obs = obs[0]
      self.group_list = common.update_group_list([obs])

    #print("group_list ", self.group_list)
    player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
    neutral_y, neutral_x = (player_relative == _PLAYER_NEUTRAL).nonzero()
    player_y, player_x = (player_relative == _PLAYER_FRIENDLY).nonzero()
    if not neutral_y.any() or not player_y.any():
      return actions.FunctionCall(_NO_OP, [])

    r = random.randint(0, 1)

    if _MOVE_SCREEN in obs.observation["available_actions"] and r == 0:

      selected = obs.observation["screen"][_SELECTED]
      player_y, player_x = (selected == _PLAYER_FRIENDLY).nonzero()

      player = [int(player_x.mean()), int(player_y.mean())]
      points = [player]
      closest, min_dist = None, None
      other_dest = None
      my_dest = None
      if ("0" in self.dest_per_marine and "1" in self.dest_per_marine):
        if (self.group_id == 0):
          my_dest = self.dest_per_marine["0"]
          other_dest = self.dest_per_marine["1"]
        elif (self.group_id == 1):
          other_dest = self.dest_per_marine["0"]
          my_dest = self.dest_per_marine["1"]

      for p in zip(neutral_x, neutral_y):

        if (other_dest):
          dist = numpy.linalg.norm(
            numpy.array(other_dest) - numpy.array(p))
          if (dist < 5):
            #print("continue since partner will take care of it ", p)
            continue

        pp = [p[0], p[1]]
        if (pp not in points):
          points.append(pp)

        dist = numpy.linalg.norm(numpy.array(player) - numpy.array(p))
        if not min_dist or dist < min_dist:
          closest, min_dist = p, dist

      solve_tsp = False
      if (my_dest):
        dist = numpy.linalg.norm(
          numpy.array(player) - numpy.array(my_dest))
        if (dist < 2):
          solve_tsp = True

      if (my_dest is None):
        solve_tsp = True

      if (len(points) < 2):
        solve_tsp = False

      if (solve_tsp):
        # function for printing best found solution when it is found
        from time import clock
        init = clock()

        def report_sol(obj, s=""):
          print("cpu:%g\tobj:%g\ttour:%s" % \
                (clock(), obj, s))

        #print("points: %s" % points)
        n, D = mk_matrix(points, distL2)
        # multi-start local search
        #print("random start local search:", n)
        niter = 50
        tour, z = multistart_localsearch(niter, n, D)

        #print("best found solution (%d iterations): z = %g" % (niter, z))
        #print(tour)

        left, right = None, None
        for idx in tour:
          if (tour[idx] == 0):
            if (idx == len(tour) - 1):
              #print("optimal next : ", tour[0])
              right = points[tour[0]]
              left = points[tour[idx - 1]]
            elif (idx == 0):
              #print("optimal next : ", tour[idx+1])
              right = points[tour[idx + 1]]
              left = points[tour[len(tour) - 1]]
            else:
              #print("optimal next : ", tour[idx+1])
              right = points[tour[idx + 1]]
              left = points[tour[idx - 1]]

        left_d = numpy.linalg.norm(
          numpy.array(player) - numpy.array(left))
        right_d = numpy.linalg.norm(
          numpy.array(player) - numpy.array(right))
        if (right_d > left_d):
          closest = left
        else:
          closest = right

      #print("optimal next :" , closest)
      self.dest_per_marine[str(self.group_id)] = closest
      #print("dest_per_marine", self.dest_per_marine)
      #dest_per_marine {'0': [56, 26], '1': [52, 6]}

      if (closest is None):
        return actions.FunctionCall(_NO_OP, [])

      return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, closest])
    elif (len(self.group_list) > 0):
      player_p = []
      for p in zip(player_x, player_y):
        if (p not in player_p):
          player_p.append(p)

      self.group_id = random.randint(0, len(self.group_list) - 1)
      return actions.FunctionCall(
        _SELECT_CONTROL_GROUP,
        [[_CONTROL_GROUP_RECALL], [int(self.group_id)]])
    else:
      return actions.FunctionCall(_NO_OP, [])


class CollectMineralShards2(base_agent.BaseAgent):
  """An agent specifically for solving the CollectMineralShards map."""

  def step(self, obs):
    super(CollectMineralShards2, self).step(obs)
    if _MOVE_SCREEN in obs.observation["available_actions"]:
      player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
      neutral_y, neutral_x = (
        player_relative == _PLAYER_NEUTRAL).nonzero()
      player_y, player_x = (
        player_relative == _PLAYER_FRIENDLY).nonzero()
      if not neutral_y.any() or not player_y.any():
        return actions.FunctionCall(_NO_OP, [])
      player = [int(player_x.mean()), int(player_y.mean())]
      closest, min_dist = None, None
      for p in zip(neutral_x, neutral_y):
        dist = numpy.linalg.norm(numpy.array(player) - numpy.array(p))
        if not min_dist or dist < min_dist:
          closest, min_dist = p, dist
      return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, closest])
    else:
      return actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])
