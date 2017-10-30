import numpy as np

from pysc2.lib import actions as sc2_actions
from pysc2.lib import features
from pysc2.lib import actions

import random
from mineral.tsp2 import multistart_localsearch, mk_matrix, distL2

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

_NOT_QUEUED = 0
_SELECT_ALL = 0

def init(env, obs):
  player_relative = obs[0].observation["screen"][_PLAYER_RELATIVE]
  #print("init")
  army_count = env._obs.observation.player_common.army_count

  if(army_count==0):
    return obs
  try:
    obs = env.step(actions=[sc2_actions.FunctionCall(_NO_OP, [])])
    obs = env.step(actions=[sc2_actions.FunctionCall(_NO_OP, [])])
    obs = env.step(actions=[sc2_actions.FunctionCall(_NO_OP, [])])
    obs = env.step(actions=[sc2_actions.FunctionCall(_NO_OP, [])])
    obs = env.step(actions=[sc2_actions.FunctionCall(_NO_OP, [])])

    player_y, player_x = (player_relative == _PLAYER_FRIENDLY).nonzero()
    obs = env.step(actions=[sc2_actions.FunctionCall(_SELECT_ARMY, [[_SELECT_ALL]])])
  except Exception as e:
    print(e)
  for i in range(len(player_x)):
    if i % 4 != 0:
      continue

    xy = [player_x[i], player_y[i]]
    obs = env.step(actions=[sc2_actions.FunctionCall(_SELECT_POINT, [[0], xy])])

  group_id = 0
  group_list = []
  unit_xy_list = []
  for i in range(len(player_x)):
    if i % 4 != 0:
      continue

    if group_id > 9:
      break

    xy = [player_x[i], player_y[i]]
    unit_xy_list.append(xy)

    if(len(unit_xy_list) >= 1):
      for idx, xy in enumerate(unit_xy_list):
        if(idx==0):
          obs = env.step(actions=[sc2_actions.FunctionCall(_SELECT_POINT, [[0], xy])])
        else:
          obs = env.step(actions=[sc2_actions.FunctionCall(_SELECT_POINT, [[1], xy])])

      obs = env.step(actions=[sc2_actions.FunctionCall(_SELECT_CONTROL_GROUP, [[_CONTROL_GROUP_SET], [group_id]])])
      unit_xy_list = []

      group_list.append(group_id)
      group_id += 1

  if(len(unit_xy_list) >= 1):
    for idx, xy in enumerate(unit_xy_list):
      if(idx==0):
        obs = env.step(actions=[sc2_actions.FunctionCall(_SELECT_POINT, [[0], xy])])
      else:
        obs = env.step(actions=[sc2_actions.FunctionCall(_SELECT_POINT, [[1], xy])])

    obs = env.step(actions=[sc2_actions.FunctionCall(_SELECT_CONTROL_GROUP, [[_CONTROL_GROUP_SET], [group_id]])])

    group_list.append(group_id)
    group_id += 1

  return obs

def solve_tsp(player_relative, selected, group_list, group_id, dest_per_marine):
  my_dest = None
  other_dest = None
  closest, min_dist = None, None
  actions = []
  neutral_y, neutral_x = (player_relative == _PLAYER_NEUTRAL).nonzero()
  player_y, player_x = (selected == 1).nonzero()

  #for group_id in group_list:
  if("0" in dest_per_marine and "1" in dest_per_marine):
    if(group_id == 0):
      my_dest = dest_per_marine["0"]
      other_dest = dest_per_marine["1"]
    else:
      my_dest = dest_per_marine["1"]
      other_dest = dest_per_marine["0"]

  r = random.randint(0,1)

  if(len(player_x)>0) and r == 0 :

    player = [int(player_x.mean()), int(player_y.mean())]
    points = [player]

    for p in zip(neutral_x, neutral_y):

      if(other_dest):
        dist = np.linalg.norm(np.array(other_dest) - np.array(p))
        if(dist<10):
          print("continue since partner will take care of it ", p)
          continue

      pp = [p[0]//2*2, p[1]//2*2]
      if(pp not in points):
        points.append(pp)

      dist = np.linalg.norm(np.array(player) - np.array(p))
      if not min_dist or dist < min_dist:
        closest, min_dist = p, dist


    solve_tsp = False
    if(my_dest):
      dist = np.linalg.norm(np.array(player) - np.array(my_dest))
      if(dist < 2):
        solve_tsp = True

    if(my_dest is None):
      solve_tsp = True

    if(len(points)< 2):
      solve_tsp = False

    if(solve_tsp):
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
      tour,z = multistart_localsearch(niter, n, D)

      #print("best found solution (%d iterations): z = %g" % (niter, z))
      #print(tour)

      left, right = None, None
      for idx in tour:
        if(tour[idx] == 0):
          if(idx == len(tour) - 1):
            #print("optimal next : ", tour[0])
            right = points[tour[0]]
            left = points[tour[idx-1]]
          elif(idx==0):
            #print("optimal next : ", tour[idx+1])
            right = points[tour[idx+1]]
            left = points[tour[len(tour)-1]]
          else:
            #print("optimal next : ", tour[idx+1])
            right = points[tour[idx+1]]
            left = points[tour[idx-1]]

      left_d = np.linalg.norm(np.array(player) - np.array(left))
      right_d = np.linalg.norm(np.array(player) - np.array(right))
      if(right_d > left_d):
        closest = left
      else:
        closest = right

    #print("optimal next :" , closest)
    dest_per_marine[str(group_id)] = closest
    #print("dest_per_marine", self.dest_per_marine)
    #dest_per_marine {'0': [56, 26], '1': [52, 6]}

    if(closest):
      actions.append({"base_action":_MOVE_SCREEN, "sub3":_NOT_QUEUED,
                      "x0": closest[0], "y0": closest[1]})
  elif(len(group_list)>0):

    group_id = random.randint(0,len(group_list)-1)
    actions.append({"base_action":_SELECT_CONTROL_GROUP,
                    "sub4":_CONTROL_GROUP_RECALL,
                    "sub5": group_id})
  return actions, group_id, dest_per_marine

def group_init_queue(player_relative):

  actions = []

  player_y, player_x = (player_relative == _PLAYER_FRIENDLY).nonzero()
  # try:
  #
  #   player_y, player_x = (player_relative == _PLAYER_FRIENDLY).nonzero()
  #   actions.append({"base_action":_SELECT_ARMY, "sub7":_SELECT_ALL})
  #
  # except Exception as e:
  #   print(e)
  # for i in range(len(player_x)):
  #   if i % 4 != 0:
  #     continue
  #
  #   xy = [player_x[i], player_y[i]]
  #   actions.append({"base_action":_SELECT_POINT, "sub6":0, "x0":xy[0], "y0":xy[1]})

  group_id = 0
  group_list = []
  unit_xy_list = []
  for i in range(len(player_x)):
    if i % 4 != 0:
      continue

    if group_id > 9:
      break

    xy = [player_x[i], player_y[i]]
    unit_xy_list.append(xy)
    # 2/select_point (6/select_point_act [4]; 0/screen [84, 84])
    # 4/select_control_group (4/control_group_act [5]; 5/control_group_id [10])
    if(len(unit_xy_list) >= 1):
      for idx, xy in enumerate(unit_xy_list):
        if(idx==0):
          actions.append({"base_action":_SELECT_POINT, "sub6":0, "x0":xy[1], "y0":xy[0]})
        else:
          actions.append({"base_action":_SELECT_POINT, "sub6":1, "x0":xy[1], "y0":xy[0]})

      actions.append({"base_action":_SELECT_CONTROL_GROUP, "sub4":_CONTROL_GROUP_SET, "sub5": group_id})
      unit_xy_list = []

      group_list.append(group_id)
      group_id += 1

  if(len(unit_xy_list) >= 1):
    for idx, xy in enumerate(unit_xy_list):
      if(idx==0):
        actions.append({"base_action":_SELECT_POINT, "sub6":0, "x0":xy[1], "y0":xy[0]})
      else:
        actions.append({"base_action":_SELECT_POINT, "sub6":1, "x0":xy[1], "y0":xy[0]})

    actions.append({"base_action":_SELECT_CONTROL_GROUP, "sub4":_CONTROL_GROUP_SET, "sub5":group_id})

    group_list.append(group_id)
    group_id += 1

  return actions

def update_group_list2(extra):

  group_count = 0
  group_list = []

  for control_group_id in range(10):
    unit_id = extra[control_group_id, 1]
    count = extra[control_group_id, 2]

    if(unit_id != 0):
      group_count += 1
      group_list.append(control_group_id)

  return group_list

def check_group_list2(extra):
  army_count = 0
   # (64, 64, 3)
  for control_group_id in range(10):
    unit_id = extra[control_group_id, 1]
    count = extra[control_group_id, 2]
    if(unit_id != 0):
      army_count += count

  if(army_count != extra[0,0]):
    return True

  return False

def update_group_list(obs):
  control_groups = obs[0].observation["control_groups"]
  group_count = 0
  group_list = []
  for id, group in enumerate(control_groups):
    if(group[0]!=0):
      group_count += 1
      group_list.append(id)
  return group_list

def check_group_list(env, obs):
  error = False
  control_groups = obs[0].observation["control_groups"]
  army_count = 0
  for id, group in enumerate(control_groups):
    if(group[0]==48):
      army_count += group[1]
      if(group[1] != 1):
        #print("group error group_id : %s count : %s" % (id, group[1]))
        error = True
        return error
  if(army_count != env._obs.observation.player_common.army_count):
    error = True
    # print("army_count %s !=  %s env._obs.observation.player_common.army_count "
    #      % (army_count, env._obs.observation.player_common.army_count))


  return error


UP, DOWN, LEFT, RIGHT = 'up', 'down', 'left', 'right'

def shift(direction, number, matrix):
  ''' shift given 2D matrix in-place the given number of rows or columns
      in the specified (UP, DOWN, LEFT, RIGHT) direction and return it
  '''
  if direction in (UP):
    matrix = np.roll(matrix, -number, axis=0)
    matrix[number:,:] = -2
    return matrix
  elif direction in (DOWN):
    matrix = np.roll(matrix, number, axis=0)
    matrix[:number,:] = -2
    return matrix
  elif direction in (LEFT):
    matrix = np.roll(matrix, -number, axis=1)
    matrix[:,number:] = -2
    return matrix
  elif direction in (RIGHT):
    matrix = np.roll(matrix, number, axis=1)
    matrix[:,:number] = -2
    return matrix
  else:
    return matrix

def select_marine(env, obs):

  player_relative = obs[0].observation["screen"][_PLAYER_RELATIVE]
  screen = player_relative

  group_list = update_group_list(obs)

  if(check_group_list(env, obs)):
    obs = init(env, obs)
    group_list = update_group_list(obs)

  # if(len(group_list) == 0):
  #   obs = init(env, player_relative, obs)
  #   group_list = update_group_list(obs)

  player_relative = obs[0].observation["screen"][_PLAYER_RELATIVE]

  friendly_y, friendly_x = (player_relative == _PLAYER_FRIENDLY).nonzero()

  enemy_y, enemy_x = (player_relative == _PLAYER_HOSTILE).nonzero()

  player = []

  danger_closest, danger_min_dist = None, None
  for e in zip(enemy_x, enemy_y):
    for p in zip(friendly_x, friendly_y):
      dist = np.linalg.norm(np.array(p) - np.array(e))
      if not danger_min_dist or dist < danger_min_dist:
        danger_closest, danger_min_dist = p, dist


  marine_closest, marine_min_dist = None, None
  for e in zip(friendly_x, friendly_y):
    for p in zip(friendly_x, friendly_y):
      dist = np.linalg.norm(np.array(p) - np.array(e))
      if not marine_min_dist or dist < marine_min_dist:
        if dist >= 2:
          marine_closest, marine_min_dist = p, dist

  if(danger_min_dist != None and danger_min_dist <= 5):
    obs = env.step(actions=[sc2_actions.FunctionCall(_SELECT_POINT, [[0], danger_closest])])

    selected = obs[0].observation["screen"][_SELECTED]
    player_y, player_x = (selected == _PLAYER_FRIENDLY).nonzero()
    if(len(player_y)>0):
      player = [int(player_x.mean()), int(player_y.mean())]

  elif(marine_closest != None and marine_min_dist <= 3):
    obs = env.step(actions=[sc2_actions.FunctionCall(_SELECT_POINT, [[0], marine_closest])])

    selected = obs[0].observation["screen"][_SELECTED]
    player_y, player_x = (selected == _PLAYER_FRIENDLY).nonzero()
    if(len(player_y)>0):
      player = [int(player_x.mean()), int(player_y.mean())]

  else:

    # If there is no marine in danger, select random
    while(len(group_list)>0):
      # units = env._obs.observation.raw_data.units
      # marine_list = []          # for unit in units:
      #   if(unit.alliance == 1):
      #     marine_list.append(unit)

      group_id = np.random.choice(group_list)
      #xy = [int(unit.pos.y - 10), int(unit.pos.x+8)]
      #print("check xy : %s - %s" % (xy, player_relative[xy[0],xy[1]]))
      obs = env.step(actions=[sc2_actions.FunctionCall(_SELECT_CONTROL_GROUP, [[_CONTROL_GROUP_RECALL], [int(group_id)]])])

      selected = obs[0].observation["screen"][_SELECTED]
      player_y, player_x = (selected == _PLAYER_FRIENDLY).nonzero()
      if(len(player_y)>0):
        player = [int(player_x.mean()), int(player_y.mean())]
        break
      else:
        group_list.remove(group_id)

  if(len(player) == 2):

    if(player[0]>32):
      screen = shift(LEFT, player[0]-32, screen)
    elif(player[0]<32):
      screen = shift(RIGHT, 32 - player[0], screen)

    if(player[1]>32):
      screen = shift(UP, player[1]-32, screen)
    elif(player[1]<32):
      screen = shift(DOWN, 32 - player[1], screen)

  return obs, screen, player

def marine_action(env, obs, player, action):

  player_relative = obs[0].observation["screen"][_PLAYER_RELATIVE]

  enemy_y, enemy_x = (player_relative == _PLAYER_HOSTILE).nonzero()

  closest, min_dist = None, None

  if(len(player) == 2):
    for p in zip(enemy_x, enemy_y):
      dist = np.linalg.norm(np.array(player) - np.array(p))
      if not min_dist or dist < min_dist:
        closest, min_dist = p, dist


  player_relative = obs[0].observation["screen"][_PLAYER_RELATIVE]
  friendly_y, friendly_x = (player_relative == _PLAYER_FRIENDLY).nonzero()

  closest_friend, min_dist_friend = None, None
  if(len(player) == 2):
    for p in zip(friendly_x, friendly_y):
      dist = np.linalg.norm(np.array(player) - np.array(p))
      if not min_dist_friend or dist < min_dist_friend:
        closest_friend, min_dist_friend = p, dist

  if(closest == None):

    new_action = [sc2_actions.FunctionCall(_NO_OP, [])]

  elif(action == 0 and closest_friend != None and min_dist_friend < 3):
    # Friendly marine is too close => Sparse!

    mean_friend = [int(friendly_x.mean()), int(friendly_x.mean())]

    diff = np.array(player) - np.array(closest_friend)

    norm = np.linalg.norm(diff)

    if(norm != 0):
      diff = diff / norm

    coord = np.array(player) + diff * 4

    if(coord[0]<0):
      coord[0] = 0
    elif(coord[0]>63):
      coord[0] = 63

    if(coord[1]<0):
      coord[1] = 0
    elif(coord[1]>63):
      coord[1] = 63

    new_action = [sc2_actions.FunctionCall(_MOVE_SCREEN, [[_NOT_QUEUED], coord])]

  elif(action <= 1): #Attack

    # nearest enemy

    coord = closest

    new_action = [sc2_actions.FunctionCall(_ATTACK_SCREEN, [[_NOT_QUEUED], coord])]

    #print("action : %s Attack Coord : %s" % (action, coord))

  elif(action == 2): # Oppsite direcion from enemy

    # nearest enemy opposite

    diff = np.array(player) - np.array(closest)

    norm = np.linalg.norm(diff)

    if(norm != 0):
      diff = diff / norm

    coord = np.array(player) + diff * 7

    if(coord[0]<0):
      coord[0] = 0
    elif(coord[0]>63):
      coord[0] = 63

    if(coord[1]<0):
      coord[1] = 0
    elif(coord[1]>63):
      coord[1] = 63

    new_action = [sc2_actions.FunctionCall(_MOVE_SCREEN, [[_NOT_QUEUED], coord])]

  elif(action == 4): #UP
    coord = [player[0], player[1] - 3]
    new_action = [sc2_actions.FunctionCall(_MOVE_SCREEN, [[_NOT_QUEUED], coord])]

  elif(action == 5): #DOWN
    coord = [player[0], player[1] + 3]
    new_action = [sc2_actions.FunctionCall(_MOVE_SCREEN, [[_NOT_QUEUED], coord])]

  elif(action == 6): #LEFT
    coord = [player[0] - 3, player[1]]
    new_action = [sc2_actions.FunctionCall(_MOVE_SCREEN, [[_NOT_QUEUED], coord])]

  elif(action == 7): #RIGHT
    coord = [player[0] + 3, player[1]]
    new_action = [sc2_actions.FunctionCall(_MOVE_SCREEN, [[_NOT_QUEUED], coord])]

    #print("action : %s Back Coord : %s" % (action, coord))


  return obs, new_action