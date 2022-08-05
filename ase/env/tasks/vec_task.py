# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from gym import spaces

from isaacgym import gymtorch
from isaacgym.torch_utils import to_torch
import torch
import numpy as np


# VecEnv Wrapper for RL training
class VecTask():
    def __init__(self, task, rl_device, clip_observations=5.0, clip_actions=1.0):
        self.task = task

        self.num_environments = task.num_envs
        self.num_agents = 1  # used for multi-agent environments
        self.num_observations = task.num_obs
        self.num_states = task.num_states
        self.num_actions = task.num_actions

        self.obs_space = spaces.Box(np.ones(self.num_obs) * -np.Inf, np.ones(self.num_obs) * np.Inf)
        self.state_space = spaces.Box(np.ones(self.num_states) * -np.Inf, np.ones(self.num_states) * np.Inf)
        self.act_space = spaces.Box(np.ones(self.num_actions) * -1., np.ones(self.num_actions) * 1.)

        self.clip_obs = clip_observations
        self.clip_actions = clip_actions
        self.rl_device = rl_device

        print("RL device: ", rl_device)

    def step(self, actions):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def get_number_of_agents(self):
        return self.num_agents

    @property
    def observation_space(self):
        return self.obs_space

    @property
    def action_space(self):
        return self.act_space

    @property
    def num_envs(self):
        return self.num_environments

    @property
    def num_acts(self):
        return self.num_actions

    @property
    def num_obs(self):
        return self.num_observations


# C++ CPU Class
class VecTaskCPU(VecTask):
    def __init__(self, task, rl_device, sync_frame_time=False, clip_observations=5.0, clip_actions=1.0):
        super().__init__(task, rl_device, clip_observations=clip_observations, clip_actions=clip_actions)
        self.sync_frame_time = sync_frame_time

    def step(self, actions):
        actions = actions.cpu().numpy()
        self.task.render(self.sync_frame_time)

        obs, rewards, resets, extras = self.task.step(np.clip(actions, -self.clip_actions, self.clip_actions))

        return (to_torch(np.clip(obs, -self.clip_obs, self.clip_obs), dtype=torch.float, device=self.rl_device),
                to_torch(rewards, dtype=torch.float, device=self.rl_device),
                to_torch(resets, dtype=torch.uint8, device=self.rl_device), [])

    def reset(self):
        actions = 0.01 * (1 - 2 * np.random.rand(self.num_envs, self.num_actions)).astype('f')

        # step the simulator
        obs, rewards, resets, extras = self.task.step(actions)

        return to_torch(np.clip(obs, -self.clip_obs, self.clip_obs), dtype=torch.float, device=self.rl_device)


# C++ GPU Class
class VecTaskGPU(VecTask):
    def __init__(self, task, rl_device, clip_observations=5.0, clip_actions=1.0):
        super().__init__(task, rl_device, clip_observations=clip_observations, clip_actions=clip_actions)

        self.obs_tensor = gymtorch.wrap_tensor(self.task.obs_tensor, counts=(self.task.num_envs, self.task.num_obs))
        self.rewards_tensor = gymtorch.wrap_tensor(self.task.rewards_tensor, counts=(self.task.num_envs,))
        self.resets_tensor = gymtorch.wrap_tensor(self.task.resets_tensor, counts=(self.task.num_envs,))

    def step(self, actions):
        self.task.render(False)
        actions_clipped = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        actions_tensor = gymtorch.unwrap_tensor(actions_clipped)

        self.task.step(actions_tensor)

        return torch.clamp(self.obs_tensor, -self.clip_obs, self.clip_obs), self.rewards_tensor, self.resets_tensor, []

    def reset(self):
        actions = 0.01 * (1 - 2 * torch.rand([self.task.num_envs, self.task.num_actions], dtype=torch.float32, device=self.rl_device))
        actions_tensor = gymtorch.unwrap_tensor(actions)

        # step the simulator
        self.task.step(actions_tensor)

        return torch.clamp(self.obs_tensor, -self.clip_obs, self.clip_obs)


# Python CPU/GPU Class
class VecTaskPython(VecTask):

    def get_state(self):
        return torch.clamp(self.task.states_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

    def step(self, actions):
        actions_tensor = torch.clamp(actions, -self.clip_actions, self.clip_actions)

        self.task.step(actions_tensor)

        return torch.clamp(self.task.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device), self.task.rew_buf.to(self.rl_device), self.task.reset_buf.to(self.rl_device), self.task.extras

    def reset(self):
        actions = 0.01 * (1 - 2 * torch.rand([self.task.num_envs, self.task.num_actions], dtype=torch.float32, device=self.rl_device))

        # step the simulator
        self.task.step(actions)

        return torch.clamp(self.task.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)
