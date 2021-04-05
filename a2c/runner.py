import numpy as np
from baselines.a2c.utils import discount_with_dones
from pysc2.lib import actions as sc2_actions

import nsml


class Runner(object):
    def __init__(self, env, model, nsteps, nscripts, nstack, gamma, callback=None):
        self.env = env
        self.model = model
        nh, nw, nc = (32, 32, 3)
        self.nsteps = nsteps
        self.nscripts = nscripts
        self.nenv = nenv = env.num_envs
        self.batch_ob_shape = (nenv * nsteps, nh, nw, nc * nstack)
        self.batch_coord_shape = (nenv * nsteps, 32)
        self.obs = np.zeros((nenv, nh, nw, nc * nstack), dtype=np.uint8)
        self.available_actions = None
        self.base_act_mask = np.full((self.nenv, 2), 0, dtype=np.uint8)
        obs, rewards, dones, available_actions, army_counts, control_groups, selected, xy_per_marine = env.reset()
        self.xy_per_marine = [{"0": [0, 0], "1": [0, 0]} for _ in range(nenv)]
        for env_num, data in enumerate(xy_per_marine):
            self.xy_per_marine[env_num] = data
        self.army_counts = army_counts
        self.control_groups = control_groups
        self.selected = selected
        self.update_obs(obs)
        self.update_available(available_actions)
        self.gamma = gamma
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]
        self.total_reward = [0.0 for _ in range(nenv)]
        self.episode_rewards = []
        self.episode_rewards_script = []
        self.episode_rewards_a2c = []
        self.episodes = 0
        self.steps = 0
        self.callback = callback

        self.action_queue = [[] for _ in range(nenv)]
        self.group_list = [[] for _ in range(nenv)]
        self.agent_state = ["IDLE" for _ in range(nenv)]
        self.dest_per_marine = [{} for _ in range(nenv)]

        self.group_id = [0 for _ in range(nenv)]

    def update_obs(self, obs):
        obs = np.asarray(obs, dtype=np.int32).swapaxes(1, 2).swapaxes(2, 3)
        self.obs = np.roll(self.obs, shift=-3, axis=3)
        new_map = np.zeros((self.nenv, 32, 32, 3))
        new_map[:, :, :, -1] = obs[:, 0, :, :]
        for env_num in range(self.nenv):
            if "0" not in self.xy_per_marine[env_num]:
                self.xy_per_marine[env_num]["0"] = [0, 0]
            if "1" not in self.xy_per_marine[env_num]:
                self.xy_per_marine[env_num]["1"] = [0, 0]

            marine0 = self.xy_per_marine[env_num]["0"]
            marine1 = self.xy_per_marine[env_num]["1"]
            new_map[env_num, marine0[0], marine0[1], -3] = 1
            new_map[env_num, marine1[0], marine1[1], -2] = 1
        self.obs[:, :, :, -3:] = new_map

    def update_available(self, _available_actions):
        self.available_actions = _available_actions
        self.base_act_mask = np.full((self.nenv, 3), 0, dtype=np.uint8)
        for env_num, list in enumerate(_available_actions):
            for action_num in list:
                if action_num == 4:
                    self.base_act_mask[env_num][0] = 1
                    self.base_act_mask[env_num][1] = 1
                elif action_num == 0:
                    self.base_act_mask[env_num][2] = 1

    def valid_base_action(self, base_actions):
        for env_num, list in enumerate(self.available_actions):
            avail = []
            for action_num in list:
                if action_num == 4:
                    avail.append(0)
                    avail.append(1)
                elif action_num == 0:
                    avail.append(2)

            if base_actions[env_num] not in avail:
                base_actions[env_num] = np.random.choice(avail)

        return base_actions

    def trans_base_actions(self, base_actions):
        new_base_actions = np.copy(base_actions)
        for env_num, ba in enumerate(new_base_actions):
            if ba == 0:
                new_base_actions[env_num] = 4  # move marine control group 0
            elif ba == 1:
                new_base_actions[env_num] = 4  # move marine control group 1
            elif ba == 2:
                new_base_actions[env_num] = 0  # move marine control group 1

        return new_base_actions

    def construct_action(self, base_actions, base_action_spec, x0, y0, x1, y1):
        actions = []
        for env_num, spec in enumerate(base_action_spec):
            two_action = []
            if base_actions[env_num] == 0:
                two_action.append(sc2_actions.FunctionCall(4, [[_CONTROL_GROUP_RECALL], [0]]))
                two_action.append(sc2_actions.FunctionCall(331, [[_NOT_QUEUED], [int(x0[env_num]), y0[env_num]]]))

            elif base_actions[env_num] == 1:
                two_action.append(sc2_actions.FunctionCall(4, [[_CONTROL_GROUP_RECALL], [1]]))
                two_action.append(sc2_actions.FunctionCall(331, [[_NOT_QUEUED], [int(x1[env_num]), y1[env_num]]]))
            elif base_actions[env_num] == 2:
                two_action.append(sc2_actions.FunctionCall(0, []))
                two_action.append(sc2_actions.FunctionCall(0, []))

            actions.append(two_action)

        return actions

    def run(self):
        mb_obs, mb_td_targets, mb_base_actions, mb_xy0, mb_xy1, mb_values, mb_dones = [], [], [], [], [], [], []

        mb_states = self.states
        for n in range(self.nsteps):
            pi1, pi_xy0, pi_xy1, values, states = self.model.step(self.obs, self.states, self.dones)

            pi1_noise = np.random.random_sample((self.nenv, 3)) * 0.3

            base_actions = np.argmax(pi1 * self.base_act_mask + pi1_noise, axis=1)
            xy0 = np.argmax(pi_xy0, axis=1)

            x0 = (xy0 % 32).astype(int)
            y0 = (xy0 / 32).astype(int)

            xy1 = np.argmax(pi_xy1, axis=1)
            x1 = (xy1 % 32).astype(int)
            y1 = (xy1 / 32).astype(int)

            base_actions = self.valid_base_action(base_actions)
            new_base_actions = self.trans_base_actions(base_actions)

            base_action_spec = self.env.action_spec(new_base_actions)
            actions = self.construct_action(base_actions, base_action_spec, x0, y0, x1, y1)

            mb_obs.append(np.copy(self.obs))
            mb_base_actions.append(base_actions)

            mb_xy0.append(xy0)
            mb_xy1.append(xy1)
            mb_values.append(values)
            mb_dones.append(self.dones)

            obs, rewards, dones, available_actions, army_counts, control_groups, selected, xy_per_marine = self.env.step(actions=actions)
            self.army_counts = army_counts
            self.control_groups = control_groups
            self.selected = selected
            for env_num, data in enumerate(xy_per_marine):
                self.xy_per_marine[env_num] = data
            self.update_available(available_actions)

            self.states = states
            self.dones = dones
            mean_100ep_reward_a2c = 0
            for n, done in enumerate(dones):
                self.total_reward[n] += float(rewards[n])
                if done:
                    self.obs[n] = self.obs[n] * 0
                    self.episodes += 1
                    num_episodes = self.episodes
                    self.episode_rewards.append(self.total_reward[n])

                    model = self.model
                    mean_100ep_reward = round(
                        np.mean(self.episode_rewards[-101:]), 1)
                    if n < self.nscripts:  # scripted agents
                        self.episode_rewards_script.append(
                            self.total_reward[n])
                        mean_100ep_reward_script = round(
                            np.mean(self.episode_rewards_script[-101:]), 1)
                        nsml.report(
                            reward_script=self.total_reward[n],
                            mean_reward_script=mean_100ep_reward_script,
                            reward=self.total_reward[n],
                            mean_100ep_reward=mean_100ep_reward,
                            episodes=self.episodes,
                            step=self.episodes,
                            scope=locals()
                        )
                    else:
                        self.episode_rewards_a2c.append(self.total_reward[n])
                        mean_100ep_reward_a2c = round(
                            np.mean(self.episode_rewards_a2c[-101:]), 1)
                        nsml.report(
                            reward_a2c=self.total_reward[n],
                            mean_reward_a2c=mean_100ep_reward_a2c,
                            reward=self.total_reward[n],
                            mean_100ep_reward=mean_100ep_reward,
                            episodes=self.episodes,
                            step=self.episodes,
                            scope=locals()
                        )
                        print("mean_100ep_reward_a2c", mean_100ep_reward_a2c)

                    if self.callback is not None:
                        self.callback(locals(), globals())
                    self.total_reward[n] = 0
                    self.group_list[n] = []

            self.update_obs(obs)
            mb_td_targets.append(rewards)
        mb_dones.append(self.dones)
        # batch of steps to batch of rollouts
        mb_obs = np.asarray(
            mb_obs, dtype=np.uint8).swapaxes(1, 0).reshape(
            self.batch_ob_shape)
        mb_td_targets = np.asarray(mb_td_targets, dtype=np.float32).swapaxes(1, 0)
        mb_base_actions = np.asarray(
            mb_base_actions, dtype=np.int32).swapaxes(1, 0)

        mb_xy0 = np.asarray(mb_xy0, dtype=np.int32).swapaxes(1, 0)
        mb_xy1 = np.asarray(mb_xy1, dtype=np.int32).swapaxes(1, 0)

        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        mb_masks = mb_dones[:, :-1]
        mb_dones = mb_dones[:, 1:]
        last_values = self.model.value(self.obs, self.states,
                                       self.dones).tolist()
        # discount/bootstrap off value fn
        for n, (rewards, dones, value) in enumerate(
                zip(mb_td_targets, mb_dones, last_values)):
            rewards = rewards.tolist()
            dones = dones.tolist()
            if dones[-1] == 0:
                rewards = discount_with_dones(rewards + [value], dones + [0],
                                              self.gamma)[:-1]
            else:
                rewards = discount_with_dones(rewards, dones, self.gamma)
            mb_td_targets[n] = rewards
        mb_td_targets = mb_td_targets.flatten()
        mb_base_actions = mb_base_actions.flatten()
        mb_xy0 = mb_xy0.flatten()
        mb_xy1 = mb_xy1.flatten()

        mb_values = mb_values.flatten()
        mb_masks = mb_masks.flatten()
        return mb_obs, mb_states, mb_td_targets, mb_masks, mb_base_actions, mb_xy0, mb_xy1, mb_values


_CONTROL_GROUP_RECALL = 0
_NOT_QUEUED = 0