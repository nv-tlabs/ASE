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

from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *

import env.tasks.humanoid_amp as humanoid_amp
import env.tasks.humanoid_amp_getup as humanoid_amp_getup
import env.tasks.humanoid_strike as humanoid_strike
import env.tasks.humanoid_location as humanoid_location
from utils import torch_utils

PERTURB_OBJS = [
    ["small", 60],
    ["small", 7],
    ["small", 10],
    ["small", 35],
    ["small", 2],
    ["small", 2],
    ["small", 3],
    ["small", 2],
    ["small", 2],
    ["small", 3],
    ["small", 2],
    ["large", 60],
    ["small", 300],
]

class HumanoidPerturb(humanoid_amp.HumanoidAMP):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        
        self._proj_dist_min = 4
        self._proj_dist_max = 5
        self._proj_h_min = 0.25
        self._proj_h_max = 2
        self._proj_steps = 150
        self._proj_warmup_steps = 1
        self._proj_speed_min = 30
        self._proj_speed_max = 40
        assert(self._proj_warmup_steps < self._proj_steps)

        self._build_proj_tensors()
        self._calc_perturb_times()

        return
    
    def _create_envs(self, num_envs, spacing, num_per_row):
        self._proj_handles = []
        self._load_proj_asset()

        super()._create_envs(num_envs, spacing, num_per_row)
        return

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)
        self._build_proj(env_id, env_ptr)
        return

    def _load_proj_asset(self):
        asset_root = "ase/data/assets/mjcf/"

        small_asset_file = "block_projectile.urdf"
        small_asset_options = gymapi.AssetOptions()
        small_asset_options.angular_damping = 0.01
        small_asset_options.linear_damping = 0.01
        small_asset_options.max_angular_velocity = 100.0
        small_asset_options.density = 200.0
        small_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        self._small_proj_asset = self.gym.load_asset(self.sim, asset_root, small_asset_file, small_asset_options)
        
        large_asset_file = "block_projectile_large.urdf"
        large_asset_options = gymapi.AssetOptions()
        large_asset_options.angular_damping = 0.01
        large_asset_options.linear_damping = 0.01
        large_asset_options.max_angular_velocity = 100.0
        large_asset_options.density = 100.0
        large_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        self._large_proj_asset = self.gym.load_asset(self.sim, asset_root, large_asset_file, large_asset_options)
        return

    def _build_proj(self, env_id, env_ptr):
        col_group = env_id
        col_filter = 0
        segmentation_id = 0

        for i, obj in enumerate(PERTURB_OBJS):
            default_pose = gymapi.Transform()
            default_pose.p.x = 200 + i
            default_pose.p.z = 1
            obj_type = obj[0]
            if (obj_type == "small"):
                proj_asset = self._small_proj_asset
            elif (obj_type == "large"):
                proj_asset = self._large_proj_asset

            proj_handle = self.gym.create_actor(env_ptr, proj_asset, default_pose, "proj{:d}".format(i), col_group, col_filter, segmentation_id)
            self._proj_handles.append(proj_handle)

        return

    def _build_body_ids_tensor(self, env_ptr, actor_handle, body_names):
        env_ptr = self.envs[0]
        actor_handle = self.humanoid_handles[0]
        body_ids = []

        for body_name in body_names:
            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
            assert(body_id != -1)
            body_ids.append(body_id)

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def _build_proj_tensors(self):
        num_actors = self.get_num_actors_per_env()
        num_objs = self._get_num_objs()
        self._proj_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., (num_actors - num_objs):, :]
        
        self._proj_actor_ids = num_actors * np.arange(self.num_envs)
        self._proj_actor_ids = np.expand_dims(self._proj_actor_ids, axis=-1)
        self._proj_actor_ids = self._proj_actor_ids + np.reshape(np.array(self._proj_handles), [self.num_envs, num_objs])
        self._proj_actor_ids = self._proj_actor_ids.flatten()
        self._proj_actor_ids = to_torch(self._proj_actor_ids, device=self.device, dtype=torch.int32)
        
        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
        self._proj_contact_forces = contact_force_tensor.view(self.num_envs, bodies_per_env, 3)[..., (num_actors - num_objs):, :]
        
        return

    def _calc_perturb_times(self):
        self._perturb_timesteps = []
        total_steps = 0
        for i, obj in enumerate(PERTURB_OBJS):
            curr_time = obj[1]
            total_steps += curr_time
            self._perturb_timesteps.append(total_steps)

        self._perturb_timesteps = np.array(self._perturb_timesteps)

        return
    
    def _reset_env_tensors(self, env_ids):
        super()._reset_env_tensors(env_ids)

        env_ids_int32 = self._proj_actor_ids[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        return

    def _compute_reset(self):
        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(self.reset_buf, self.progress_buf,
                                                               self._contact_forces, self._contact_body_ids,
                                                               self._rigid_body_pos, self.max_episode_length,
                                                               self._enable_early_termination, self._termination_heights)
        return

    def post_physics_step(self):
        self._update_proj()
        super().post_physics_step()
        return
    
    def _get_num_objs(self):
        return len(PERTURB_OBJS)

    def _update_proj(self):
        
        curr_timestep = self.progress_buf.cpu().numpy()[0]
        curr_timestep = curr_timestep % (self._perturb_timesteps[-1] + 1)
        perturb_step = np.where(self._perturb_timesteps == curr_timestep)[0]
        
        if (len(perturb_step) > 0):
            perturb_id = perturb_step[0]
            n = self.num_envs
            humanoid_root_pos = self._humanoid_root_states[..., 0:3]

            rand_theta = torch.rand([n], dtype=self._proj_states.dtype, device=self._proj_states.device)
            rand_theta *= 2 * np.pi
            rand_dist = (self._proj_dist_max - self._proj_dist_min) * torch.rand([n], dtype=self._proj_states.dtype, device=self._proj_states.device) + self._proj_dist_min
            pos_x = rand_dist * torch.cos(rand_theta)
            pos_y = -rand_dist * torch.sin(rand_theta)
            pos_z = (self._proj_h_max - self._proj_h_min) * torch.rand([n], dtype=self._proj_states.dtype, device=self._proj_states.device) + self._proj_h_min
            
            self._proj_states[..., perturb_id, 0] = humanoid_root_pos[..., 0] + pos_x
            self._proj_states[..., perturb_id, 1] = humanoid_root_pos[..., 1] + pos_y
            self._proj_states[..., perturb_id, 2] = pos_z
            self._proj_states[..., perturb_id, 3:6] = 0.0
            self._proj_states[..., perturb_id, 6] = 1.0
            
            tar_body_idx = np.random.randint(self.num_bodies)
            tar_body_idx = 1

            launch_tar_pos = self._rigid_body_pos[..., tar_body_idx, :]
            launch_dir = launch_tar_pos - self._proj_states[..., perturb_id, 0:3]
            launch_dir += 0.1 * torch.randn_like(launch_dir)
            launch_dir = torch.nn.functional.normalize(launch_dir, dim=-1)
            launch_speed = (self._proj_speed_max - self._proj_speed_min) * torch.rand_like(launch_dir[:, 0:1]) + self._proj_speed_min
            launch_vel = launch_speed * launch_dir
            launch_vel[..., 0:2] += self._rigid_body_vel[..., tar_body_idx, 0:2]
            self._proj_states[..., perturb_id, 7:10] = launch_vel
            self._proj_states[..., perturb_id, 10:13] = 0.0

            self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                         gymtorch.unwrap_tensor(self._proj_actor_ids),
                                                         len(self._proj_actor_ids))

        return

    def _draw_task(self):
        super()._draw_task()
        
        cols = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

        self.gym.clear_lines(self.viewer)

        starts = self._humanoid_root_states[..., 0:3]
        ends = self._proj_states[..., 0:3]
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
def compute_humanoid_reset(reset_buf, progress_buf, contact_buf, contact_body_ids, rigid_body_pos,
                           max_episode_length, enable_early_termination, termination_heights):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, bool, Tensor) -> Tuple[Tensor, Tensor]
    
    terminated = torch.zeros_like(reset_buf)
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)

    return reset, terminated
