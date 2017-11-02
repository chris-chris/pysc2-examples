import numpy as np
from multiprocessing import Process, Pipe
from baselines.common.vec_env import VecEnv
from pysc2.env import environment
from pysc2.env import sc2_env
from pysc2.lib import features, actions

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_SELECTED = features.SCREEN_FEATURES.selected.index

from defeat_zerglings import common

def worker(remote, map_name, i):

  with sc2_env.SC2Env(
      map_name=map_name,
      step_mul=1,
      screen_size_px=(32,32),
      minimap_size_px=(32,32)
  ) as env:
    available_actions = None
    result = None
    group_list = []
    while True:
      cmd, data = remote.recv()
      if cmd == 'step':
        # if(common.check_group_list(env, result)):
        #   result, xy_per_marine = common.init(env,result)

        reward = 0

        if(len(group_list) == 0 or common.check_group_list(env, result)):
          print("init group list")
          result, xy_per_marine = common.init(env, result)
          group_list = common.update_group_list(result)

        action1 = data[0][0]
        action2 = data[0][1]
        func = actions.FUNCTIONS[action1[0]]
        #print("agent(",i," ) action : ", action1, " func : ", func)
        func = actions.FUNCTIONS[action2[0]]
        #print("agent(",i," ) action : ", action2, " func : ", func, "xy :", action2[1][1])
        x, y = action2[1][1]
        move = True
        if(x==0 and y==0):
          move = False
        result = env.step(actions=[action1])
        reward += result[0].reward
        done = result[0].step_type == environment.StepType.LAST
        if(331 in available_actions and move and not done):
          try:
            result = env.step(actions=[action2])
            reward += result[0].reward
            done = result[0].step_type == environment.StepType.LAST
          except Exception as e:
            print("e :", e)

        ob = (result[0].observation["screen"][_PLAYER_RELATIVE:_PLAYER_RELATIVE+1] == 3).astype(int) #  (1, 32, 32)
        selected = result[0].observation["screen"][_SELECTED:_SELECTED+1] #  (1, 32, 32)
        # extra = np.zeros((1, 32, 32))
        control_groups = result[0].observation["control_groups"]
        army_count = env._obs[0].observation.player_common.army_count
        # extra[0,0,0] = army_count
        # for id, group in enumerate(control_groups):
        #   control_group_id = id
        #   unit_id = group[0]
        #   count = group[1]
        #   #print("control_group_id :", control_group_id, " unit_id :", unit_id, " count :", count)
        #   extra[0,1, control_group_id] = unit_id
        #   extra[0,2, control_group_id] = count
        #ob = np.append(ob, selected, axis=0) #  (2, 32, 32)
        #ob = np.append(ob, extra, axis=0) # (3, 32, 32)

        available_actions = result[0].observation["available_actions"]
        info = result[0].observation["available_actions"]
        if done:
          result = env.reset()

          if(len(group_list) == 0 or common.check_group_list(env, result)):
            print("init group list")
            result, xy_per_marine = common.init(env, result)
            group_list = common.update_group_list(result)

          # ob = result[0].observation["screen"]
          # reward = result[0].reward
          # done = result[0].step_type == environment.StepType.LAST
          info = result[0].observation["available_actions"]
        remote.send((ob, reward, done, info, army_count, control_groups, selected, xy_per_marine))
      elif cmd == 'reset':
        result = env.reset()
        reward = 0

        if(len(group_list) == 0 or common.check_group_list(env, result)):
          print("init group list")
          result, xy_per_marine = common.init(env, result)
          group_list = common.update_group_list(result)

        reward += result[0].reward
        ob = (result[0].observation["screen"][_PLAYER_RELATIVE:_PLAYER_RELATIVE+1] == 3).astype(int)
        selected = result[0].observation["screen"][_SELECTED:_SELECTED+1] #  (1, 32, 32)
        # extra = np.zeros((1, 32, 32))
        control_groups = result[0].observation["control_groups"]
        army_count = env._obs[0].observation.player_common.army_count
        # extra[0,0,0] = army_count
        # for id, group in enumerate(control_groups):
        #   control_group_id = id
        #   unit_id = group[0]
        #   count = group[1]
        #   #print("control_group_id :", control_group_id, " unit_id :", unit_id, " count :", count)
        #   extra[0,1, control_group_id] = unit_id
        #   extra[0,2, control_group_id] = count
        # ob = np.append(ob, selected, axis=0) #  (2, 32, 32)
        # ob = np.append(ob, extra, axis=0) # (3, 32, 32)

        done = result[0].step_type == environment.StepType.LAST
        info = result[0].observation["available_actions"]
        available_actions = result[0].observation["available_actions"]
        remote.send((ob, reward, done, info, army_count, control_groups, selected, xy_per_marine))
      elif cmd == 'close':
        remote.close()
        break
      elif cmd == 'get_spaces':
        remote.send((env.action_spec().functions[data], ""))
      elif cmd == "action_spec":
        remote.send((env.action_spec().functions[data]))
      else:
        raise NotImplementedError

class SubprocVecEnv(VecEnv):
  def __init__(self, nenvs, map_name):
    """
    envs: list of gym environments to run in subprocesses
    """

    self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])

    self.ps = []
    i = 0
    for (work_remote,) in zip(self.work_remotes,):
      self.ps.append(Process(target=worker, args=(work_remote, map_name, i)))
      i += 1

    #
    # self.ps = [Process(target=worker, args=(work_remote, (map_name)))
    #            for (work_remote,) in zip(self.work_remotes,)]
    for p in self.ps:
      p.start()

    self.remotes[0].send(('get_spaces', 1))
    self.action_space, self.observation_space = self.remotes[0].recv()
    #print("action_space: ", self.action_space, " observation_space: ", self.observation_space)

  def step(self, actions):
    for remote, action in zip(self.remotes, actions):
      remote.send(('step', [action]))
    results = [remote.recv() for remote in self.remotes]
    obs, rews, dones, infos, army_counts, control_groups, selected, xy_per_marine = zip(*results)
    return np.stack(obs), np.stack(rews), np.stack(dones), infos, army_counts, control_groups, np.stack(selected), xy_per_marine

  def reset(self):
    for remote in self.remotes:
      remote.send(('reset', None))
    results = [remote.recv() for remote in self.remotes]
    obs, rews, dones, infos, army_counts, control_groups, selected, xy_per_marine = zip(*results)
    return np.stack(obs), np.stack(rews), np.stack(dones), infos, army_counts, control_groups, np.stack(selected), xy_per_marine

  def action_spec(self, base_actions):
    for remote, base_action in zip(self.remotes, base_actions):
      remote.send(('action_spec', base_action))
    results = [remote.recv() for remote in self.remotes]

    return results

  def close(self):
    for remote in self.remotes:
      remote.send(('close', None))
    for p in self.ps:
      p.join()

  @property
  def num_envs(self):
    return len(self.remotes)
