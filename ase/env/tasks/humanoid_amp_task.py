# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch

import env.tasks.humanoid_amp as humanoid_amp

class HumanoidAMPTask(humanoid_amp.HumanoidAMP):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self._enable_task_obs = cfg["env"]["enableTaskObs"]

        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        return

    
    def get_obs_size(self):
        obs_size = super().get_obs_size()
        if (self._enable_task_obs):
            task_obs_size = self.get_task_obs_size()
            obs_size += task_obs_size
        return obs_size

    def get_task_obs_size(self):
        return 0

    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        self._update_task()
        return

    def render(self, sync_frame_time=False):
        super().render(sync_frame_time)

        if self.viewer:
            self._draw_task()
        return

    def _update_task(self):
        return

    def _reset_envs(self, env_ids):
        super()._reset_envs(env_ids)
        self._reset_task(env_ids)
        return

    def _reset_task(self, env_ids):
        return

    def _compute_observations(self, env_ids=None):
        humanoid_obs = self._compute_humanoid_obs(env_ids)
        
        if (self._enable_task_obs):
            task_obs = self._compute_task_obs(env_ids)
            obs = torch.cat([humanoid_obs, task_obs], dim=-1)
        else:
            obs = humanoid_obs

        if (env_ids is None):
            self.obs_buf[:] = obs
        else:
            self.obs_buf[env_ids] = obs
        return

    def _compute_task_obs(self, env_ids=None):
        return NotImplemented

    def _compute_reward(self, actions):
        return NotImplemented

    def _draw_task(self):
        return