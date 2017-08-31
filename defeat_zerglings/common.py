import numpy as np

from pysc2.lib import actions as sc2_actions
from pysc2.lib import features
from pysc2.lib import actions

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

def init(env, player_relative, obs):

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
    obs = env.step(actions=[sc2_actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])])
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
    obs = init(env, player_relative, obs)
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
      obs = env.step(actions=[sc2_actions.FunctionCall(_SELECT_CONTROL_GROUP, [[_CONTROL_GROUP_RECALL], [group_id]])])

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

    new_action = [sc2_actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, coord])]

  elif(action <= 1): #Attack

    # nearest enemy

    coord = closest

    new_action = [sc2_actions.FunctionCall(_ATTACK_SCREEN, [_NOT_QUEUED, coord])]

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

    new_action = [sc2_actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, coord])]

  elif(action == 4): #UP
    coord = [player[0], player[1] - 3]
    new_action = [sc2_actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, coord])]

  elif(action == 5): #DOWN
    coord = [player[0], player[1] + 3]
    new_action = [sc2_actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, coord])]

  elif(action == 6): #LEFT
    coord = [player[0] - 3, player[1]]
    new_action = [sc2_actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, coord])]

  elif(action == 7): #RIGHT
    coord = [player[0] + 3, player[1]]
    new_action = [sc2_actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, coord])]

    #print("action : %s Back Coord : %s" % (action, coord))


  return obs, new_action