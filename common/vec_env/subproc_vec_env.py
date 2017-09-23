import numpy as np
from multiprocessing import Process, Pipe
from baselines.common.vec_env import VecEnv
from pysc2.env import environment
from common.spaces.box import Box
from common.spaces.multi_discrete import MultiDiscrete
from common.spaces.tuple_space import Tuple
from common.spaces.discrete import Discrete


def worker(remote, env):
    #env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            obs = env.step(actions=data)
            ob = obs[0].observation["screen"]
            reward = obs[0].reward
            done = obs[0].step_type == environment.StepType.LAST
            info = {} # empty
            if done:
                obs = env.reset()
                ob = obs[0].observation["screen"]
                reward = obs[0].reward
                done = obs[0].step_type == environment.StepType.LAST
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':

            obs = env.reset()
            ob = obs[0].observation["screen"]

            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((Box(0, 4,(64, 64)),
                         Tuple(( Discrete(2), MultiDiscrete( [[0,64],[0,64]])))
                         ))
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
    def __init__(self, env_fns):
        """
        envs: list of gym environments to run in subprocesses
        """
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])        
        self.ps = [Process(target=worker, args=(work_remote, env_fn))
            for (work_remote, env_fn) in zip(self.work_remotes, env_fns)]
        for p in self.ps:
            p.start()

        self.remotes[0].send(('get_spaces', None))
        self.action_space, self.observation_space = self.remotes[0].recv()


    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        results = [remote.recv() for remote in self.remotes]
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

    @property
    def num_envs(self):
        return len(self.remotes)
