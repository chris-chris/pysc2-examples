import numpy as np
from multiprocessing import Process, Pipe
from baselines.common.vec_env import VecEnv
from pysc2.env import environment
from common.spaces.box import Box
from common.spaces.multi_discrete import MultiDiscrete
from common.spaces.tuple_space import Tuple
from common.spaces.discrete import Discrete
from pysc2.env import sc2_env
from pysc2.lib import features

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index



def worker(remote, map_name):

    with sc2_env.SC2Env(
        map_name,
        step_mul=1,
    ) as env:

        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                print("remote : ", remote, " action : ", data)
                result = env.step(actions=data)
                ob = result[0].observation["screen"][_PLAYER_RELATIVE:_PLAYER_RELATIVE+1]
                reward = result[0].reward
                done = result[0].step_type == environment.StepType.LAST
                info = result[0].observation["available_actions"]

                if done:
                    result = env.reset()
                    # ob = result[0].observation["screen"]
                    # reward = result[0].reward
                    # done = result[0].step_type == environment.StepType.LAST
                    info = result[0].observation["available_actions"]
                remote.send((ob, reward, done, info))
            elif cmd == 'reset':

                result = env.reset()
                ob = result[0].observation["screen"][_PLAYER_RELATIVE:_PLAYER_RELATIVE+1]
                reward = result[0].reward
                done = result[0].step_type == environment.StepType.LAST
                info = result[0].observation["available_actions"]

                remote.send((ob, reward, done, info))

            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_spaces':
                remote.send((env.action_spec().functions[data], ""))
            elif cmd == "action_spec":
                remote.send((env.action_spec().functions[data]))
            else:
                raise NotImplementedError

class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

class SubprocVecEnv(VecEnv):
    def __init__(self, nenvs, map_name):
        """
        envs: list of gym environments to run in subprocesses
        """

        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, (map_name)))
                   for (work_remote,) in zip(self.work_remotes,)]
        for p in self.ps:
            p.start()

        self.remotes[0].send(('get_spaces', 1))
        self.action_space, self.observation_space = self.remotes[0].recv()
        #print("action_space: ", self.action_space, " observation_space: ", self.observation_space)

    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', [action]))
        results = [remote.recv() for remote in self.remotes]
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        results = [remote.recv() for remote in self.remotes]
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

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
