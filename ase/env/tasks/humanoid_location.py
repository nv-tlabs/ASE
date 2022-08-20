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

import env.tasks.humanoid as humanoid
import env.tasks.humanoid_amp as humanoid_amp
import env.tasks.humanoid_amp_task as humanoid_amp_task
from utils import torch_utils

from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *

class HumanoidLocation(humanoid_amp_task.HumanoidAMPTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self._tar_speed = cfg["env"]["tarSpeed"]
        self._tar_change_steps_min = cfg["env"]["tarChangeStepsMin"]
        self._tar_change_steps_max = cfg["env"]["tarChangeStepsMax"]
        self._tar_dist_max = cfg["env"]["tarDistMax"]

        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        
        self._tar_change_steps = torch.zeros([self.num_envs], device=self.device, dtype=torch.int64)
        self._prev_root_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)
        self._tar_pos = torch.zeros([self.num_envs, 2], device=self.device, dtype=torch.float)

        if (not self.headless):
            self._build_marker_state_tensors()

        return

    def get_task_obs_size(self):
        obs_size = 0
        if (self._enable_task_obs):
            obs_size = 2
        return obs_size

    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        self._prev_root_pos[:] = self._humanoid_root_states[..., 0:3]
        return
    
    def _update_marker(self):
        self._marker_pos[..., 0:2] = self._tar_pos
        self._marker_pos[..., 2] = 0.0

        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(self._marker_actor_ids), len(self._marker_actor_ids))
        return

    def _create_envs(self, num_envs, spacing, num_per_row):
        if (not self.headless):
            self._marker_handles = []
            self._load_marker_asset()

        super()._create_envs(num_envs, spacing, num_per_row)
        return

    def _load_marker_asset(self):
        asset_root = "ase/data/assets/mjcf/"
        asset_file = "location_marker.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.density = 1.0
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        self._marker_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        return

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)
        
        if (not self.headless):
            self._build_marker(env_id, env_ptr)

        return

    def _build_marker(self, env_id, env_ptr):
        col_group = env_id
        col_filter = 2
        segmentation_id = 0
        default_pose = gymapi.Transform()
        
        marker_handle = self.gym.create_actor(env_ptr, self._marker_asset, default_pose, "marker", col_group, col_filter, segmentation_id)
        self.gym.set_rigid_body_color(env_ptr, marker_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.8, 0.0, 0.0))
        self._marker_handles.append(marker_handle)

        return

    def _build_marker_state_tensors(self):
        num_actors = self._root_states.shape[0] // self.num_envs
        self._marker_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., 1, :]
        self._marker_pos = self._marker_states[..., :3]
        
        self._marker_actor_ids = self._humanoid_actor_ids + 1

        return

    def _update_task(self):
        reset_task_mask = self.progress_buf >= self._tar_change_steps
        rest_env_ids = reset_task_mask.nonzero(as_tuple=False).flatten()
        if len(rest_env_ids) > 0:
            self._reset_task(rest_env_ids)
        return

    def _reset_task(self, env_ids):
        n = len(env_ids)

        char_root_pos = self._humanoid_root_states[env_ids, 0:2]
        rand_pos = self._tar_dist_max * (2.0 * torch.rand([n, 2], device=self.device) - 1.0)

        change_steps = torch.randint(low=self._tar_change_steps_min, high=self._tar_change_steps_max,
                                     size=(n,), device=self.device, dtype=torch.int64)

        self._tar_pos[env_ids] = char_root_pos + rand_pos
        self._tar_change_steps[env_ids] = self.progress_buf[env_ids] + change_steps
        return

    def _compute_task_obs(self, env_ids=None):
        if (env_ids is None):
            root_states = self._humanoid_root_states
            tar_pos = self._tar_pos
        else:
            root_states = self._humanoid_root_states[env_ids]
            tar_pos = self._tar_pos[env_ids]
        
        obs = compute_location_observations(root_states, tar_pos)
        return obs

    def _compute_reward(self, actions):
        root_pos = self._humanoid_root_states[..., 0:3]
        root_rot = self._humanoid_root_states[..., 3:7]
        self.rew_buf[:] = compute_location_reward(root_pos, self._prev_root_pos, root_rot,
                                                 self._tar_pos, self._tar_speed,
                                                 self.dt)
        return

    def _draw_task(self):
        self._update_marker()
        
        cols = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)

        self.gym.clear_lines(self.viewer)

        starts = self._humanoid_root_states[..., 0:3]
        ends = self._marker_pos

        verts = torch.cat([starts, ends], dim=-1).cpu().numpy()

        for i, env_ptr in enumerate(self.envs):
            curr_verts = verts[i]
            curr_verts = curr_verts.reshape([1, 6])
            self.gym.add_lines(self.viewer, env_ptr, curr_verts.shape[0], curr_verts, cols)

        return

#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_location_observations(root_states, tar_pos):
    # type: (Tensor, Tensor) -> Tensor
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]

    tar_pos3d = torch.cat([tar_pos, torch.zeros_like(tar_pos[..., 0:1])], dim=-1)
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
    
    local_tar_pos = quat_rotate(heading_rot, tar_pos3d - root_pos)
    local_tar_pos = local_tar_pos[..., 0:2]

    obs = local_tar_pos
    return obs

@torch.jit.script
def compute_location_reward(root_pos, prev_root_pos, root_rot, tar_pos, tar_speed, dt):
    # type: (Tensor, Tensor, Tensor, Tensor, float, float) -> Tensor
    dist_threshold = 0.5

    pos_err_scale = 0.5
    vel_err_scale = 4.0

    pos_reward_w = 0.5
    vel_reward_w = 0.4
    face_reward_w = 0.1
    
    pos_diff = tar_pos - root_pos[..., 0:2]
    pos_err = torch.sum(pos_diff * pos_diff, dim=-1)
    pos_reward = torch.exp(-pos_err_scale * pos_err)

    tar_dir = tar_pos - root_pos[..., 0:2]
    tar_dir = torch.nn.functional.normalize(tar_dir, dim=-1)
    
    
    delta_root_pos = root_pos - prev_root_pos
    root_vel = delta_root_pos / dt
    tar_dir_speed = torch.sum(tar_dir * root_vel[..., :2], dim=-1)
    tar_vel_err = tar_speed - tar_dir_speed
    tar_vel_err = torch.clamp_min(tar_vel_err, 0.0)
    vel_reward = torch.exp(-vel_err_scale * (tar_vel_err * tar_vel_err))
    speed_mask = tar_dir_speed <= 0
    vel_reward[speed_mask] = 0


    heading_rot = torch_utils.calc_heading_quat(root_rot)
    facing_dir = torch.zeros_like(root_pos)
    facing_dir[..., 0] = 1.0
    facing_dir = quat_rotate(heading_rot, facing_dir)
    facing_err = torch.sum(tar_dir * facing_dir[..., 0:2], dim=-1)
    facing_reward = torch.clamp_min(facing_err, 0.0)


    dist_mask = pos_err < dist_threshold
    facing_reward[dist_mask] = 1.0
    vel_reward[dist_mask] = 1.0

    reward = pos_reward_w * pos_reward + vel_reward_w * vel_reward + face_reward_w * facing_reward

    return reward