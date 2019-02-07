import numpy as np
from multiprocessing import Process, Pipe
from baselines.common.vec_env import VecEnv
from pysc2.env import environment
from pysc2.env import sc2_env
from pysc2.lib import features, actions

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_SELECTED = features.SCREEN_FEATURES.selected.index

from common import common


def worker(remote, map_name, nscripts, i):

  agent_format = sc2_env.AgentInterfaceFormat(
      feature_dimensions=sc2_env.Dimensions(
          screen=(32,32),
          minimap=(32,32)
      )
  )

  with sc2_env.SC2Env(
      agent_interface_format=[agent_format],
      map_name=map_name,
      step_mul=2) as env:
    available_actions = []
    result = None
    group_list = []
    xy_per_marine = {}
    while True:
      cmd, data = remote.recv()
      if cmd == 'step':
        reward = 0

        if len(group_list) == 0 or common.check_group_list(env, result):
          print("init group list")
          result, xy_per_marine = common.init(env, result)
          group_list = common.update_group_list(result)

        action1 = data[0][0]
        action2 = data[0][1]
        # func = actions.FUNCTIONS[action1[0]]
        # print("agent(",i," ) action : ", action1, " func : ", func)
        func = actions.FUNCTIONS[action2[0]]
        # print("agent(",i," ) action : ", action2, " func : ", func)


        result = env.step(actions=[action1])
        reward += result[0].reward
        done = result[0].step_type == environment.StepType.LAST

        move = True

        if len(action2[1]) == 2:
          x, y = action2[1][1]
          # print("x, y:", x, y)

          # if x == 0 and y == 0:
          #   move = False

        if (331 in available_actions and move and not done):
          try:
            result = env.step(actions=[action2])
            reward += result[0].reward
            done = result[0].step_type == environment.StepType.LAST
          except Exception as e:
            print("e :", e)

        ob = (result[0].observation["feature_screen"][
            _PLAYER_RELATIVE:_PLAYER_RELATIVE + 1] == 3).astype(int)
        #  (1, 32, 32)
        selected = result[0].observation["feature_screen"][
            _SELECTED:_SELECTED + 1]  #  (1, 32, 32)
        # extra = np.zeros((1, 32, 32))
        control_groups = result[0].observation["control_groups"]
        army_count = env._obs[0].observation.player_common.army_count

        available_actions = result[0].observation["available_actions"]
        info = result[0].observation["available_actions"]
        if done:
          result = env.reset()

          if len(group_list) == 0 or common.check_group_list(env, result):
            # print("init group list")
            result, xy_per_marine = common.init(env, result)
            group_list = common.update_group_list(result)

          info = result[0].observation["available_actions"]

        if len(action1[1]) == 2:

          group_id = action1[1][1][0]

          player_y, player_x = (result[0].observation["feature_screen"][
              _SELECTED] == 1).nonzero()

          if len(player_x) > 0:
            if (group_id == 1):
              xy_per_marine["1"] = [int(player_x.mean()), int(player_y.mean())]
            else:
              xy_per_marine["0"] = [int(player_x.mean()), int(player_y.mean())]
          
        remote.send((ob, reward, done, info, army_count,
                     control_groups, selected, xy_per_marine))

      elif cmd == 'reset':
        result = env.reset()
        reward = 0

        if len(group_list) == 0 or common.check_group_list(env, result):
          # print("init group list")
          result, xy_per_marine = common.init(env, result)
          group_list = common.update_group_list(result)

        time_step = result[0]
        reward += time_step.reward
        ob = (time_step.observation["feature_screen"][
              _PLAYER_RELATIVE:_PLAYER_RELATIVE + 1] == 3).astype(int)

        selected = time_step.observation["feature_screen"][
                   _SELECTED:_SELECTED + 1]  #  (1, 32, 32)
        # extra = np.zeros((1, 32, 32))
        control_groups = time_step.observation["control_groups"]
        army_count = env._obs[0].observation.player_common.army_count

        done = time_step.step_type == environment.StepType.LAST
        info = time_step.observation["available_actions"]
        available_actions = time_step.observation["available_actions"]
        remote.send((ob, reward, done, info, army_count,
                     control_groups, selected, xy_per_marine))
      elif cmd == 'close':
        remote.close()
        break
      elif cmd == 'get_spaces':
        spec = env.action_spec()[0]
        remote.send((spec.functions[data], ""))
      elif cmd == "action_spec":
        spec = env.action_spec()[0]
        remote.send((spec.functions[data]))
      else:
        raise NotImplementedError


class SubprocVecEnv(VecEnv):
  def __init__(self, nenvs, nscripts, map_name):
    """
envs: list of gym environments to run in subprocesses
"""

    self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])

    self.ps = []
    i = 0
    for (work_remote, ) in zip(self.work_remotes, ):
      self.ps.append(
        Process(target=worker, args=(work_remote, map_name, nscripts, i)))
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
    obs, rews, dones, infos, army_counts, control_groups, selected, xy_per_marine = zip(
      *results)
    obs = [np.array(o) for o in obs]
    selected = [np.array(o) for o in selected]
    return (np.stack(obs), np.stack(rews), np.stack(dones),
      infos, army_counts, control_groups, np.stack(selected),
      xy_per_marine)

  def reset(self):
    for remote in self.remotes:
      remote.send(('reset', None))
    results = [remote.recv() for remote in self.remotes]
    obs, rews, dones, infos, army_counts, control_groups, selected, xy_per_marine = zip(
      *results)
    obs = [np.array(o) for o in obs]
    selected = [np.array(o) for o in selected]
    return (np.stack(obs), np.stack(rews), np.stack(dones),
      infos, army_counts, control_groups, np.stack(selected),
      xy_per_marine)

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

  def step_async(self, actions):
    self.actions = actions

  def step_wait(self):
    for i in range(self.num_envs):
      obs_tuple, self.buf_rews[i], self.buf_dones[i], self.buf_infos[i] = self.envs[i].step(self.actions[i])
      if self.buf_dones[i]:
        obs_tuple = self.envs[i].reset()
      if isinstance(obs_tuple, (tuple, list)):
        for t, x in enumerate(obs_tuple):
          self.buf_obs[t][i] = x
      else:
        self.buf_obs[0][i] = obs_tuple
    return (self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones),
            self.buf_infos.copy())

  @property
  def num_envs(self):
    return len(self.remotes)
