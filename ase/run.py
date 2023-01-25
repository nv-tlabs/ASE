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

import os

from utils.config import set_np_formatting, set_seed, get_args, parse_sim_params, load_cfg
from utils.parse_task import parse_task

from rl_games.algos_torch import players
from rl_games.algos_torch import torch_ext
from rl_games.common import env_configurations, experiment, vecenv
from rl_games.common.algo_observer import AlgoObserver
from rl_games.torch_runner import Runner

import numpy as np
import copy
import torch

from learning import deepmm_agent
from learning import deepmm_players
from learning import deepmm_models
from learning import deepmm_network_builder

from learning import amp_agent
from learning import amp_players
from learning import amp_models
from learning import amp_network_builder

# from learning import ase_agent
# from learning import ase_players
# from learning import ase_models
# from learning import ase_network_builder

# from learning import hrl_agent
# from learning import hrl_players
# from learning import hrl_models
# from learning import hrl_network_builder

args = None
cfg = None
cfg_train = None

def create_rlgpu_env(**kwargs):
    use_horovod = cfg_train['params']['config'].get('multi_gpu', False)
    
    #! For multi-GPU setting
    if use_horovod:
        import horovod.torch as hvd

        rank = hvd.rank()
        print("Horovod rank: ", rank)

        cfg_train['params']['seed'] = cfg_train['params']['seed'] + rank

        #! args는 get_args()로부터 얻어올 수 있음. defined in ase>utils.config.py
        args.device = 'cuda'
        args.device_id = rank
        args.rl_device = 'cuda:' + str(rank)

        cfg['rank'] = rank
        cfg['rl_device'] = 'cuda:' + str(rank)

    #! from ase>utils.config.py (기본 sim_params 설정 + config yaml 파일의 [sim]에 적힌 info로도 설정)
    sim_params = parse_sim_params(args, cfg, cfg_train)
    #! from ase>utils.parse_task.py
    # #! 여기서 class Humanoid가 지정됌!         
    task, env = parse_task(args, cfg, cfg_train, sim_params)    #! parse_task > VecTaskPythonWrapper(VecTaskPython) > amp_obs_space도 wrapping 해줌.

    #? why! 이거 어떻게 되는지 나중에 알아보기
    print('num_envs: {:d}'.format(env.num_envs))
    print('num_actions: {:d}'.format(env.num_actions))
    print('num_obs: {:d}'.format(env.num_obs))
    print('num_states: {:d}'.format(env.num_states))
    
    #! pop: removes and returns value: value which is to be returned when the key is not in the dictionary 
    frames = kwargs.pop('frames', 1)
    if frames > 1:
        #? why? 다시 보기
        env = wrappers.FrameStack(env, frames, False)
    return env


class RLGPUAlgoObserver(AlgoObserver):
    def __init__(self, use_successes=True):
        self.use_successes = use_successes
        return

    def after_init(self, algo):
        self.algo = algo
        self.consecutive_successes = torch_ext.AverageMeter(1, self.algo.games_to_track).to(self.algo.ppo_device)
        self.writer = self.algo.writer
        return

    #! Used in amp_agent.py
    def process_infos(self, infos, done_indices):
        if isinstance(infos, dict):
            if (self.use_successes == False) and 'consecutive_successes' in infos:
                cons_successes = infos['consecutive_successes'].clone()
                self.consecutive_successes.update(cons_successes.to(self.algo.ppo_device))
            if self.use_successes and 'successes' in infos:
                successes = infos['successes'].clone()
                self.consecutive_successes.update(successes[done_indices].to(self.algo.ppo_device))
        return

    def after_clear_stats(self):
        self.mean_scores.clear()
        return

    def after_print_stats(self, frame, epoch_num, total_time):
        if self.consecutive_successes.current_size > 0:
            mean_con_successes = self.consecutive_successes.get_mean()
            self.writer.add_scalar('successes/consecutive_successes/mean', mean_con_successes, frame)
            self.writer.add_scalar('successes/consecutive_successes/iter', mean_con_successes, epoch_num)
            self.writer.add_scalar('successes/consecutive_successes/time', mean_con_successes, total_time)
        return


class RLGPUEnv(vecenv.IVecEnv):
    def __init__(self, config_name, num_actors, **kwargs):
        #! config_name: rlgpu 
        #! env_configurations.configurations[config_name]: {'env_creator': <function <lambda> at 0x7f80565f43b0>, 'vecenv_type': 'RLGPU'}
        self.env = env_configurations.configurations[config_name]['env_creator'](**kwargs)  #! env.tasks.vec_task_wrappers.VecTaskPythonWrapper
        self.use_global_obs = (self.env.num_states > 0)

        self.full_state = {}
        self.full_state["obs"] = self.reset()
        if self.use_global_obs:
            self.full_state["states"] = self.env.get_state()
        return

    def step(self, action):
        next_obs, reward, is_done, info = self.env.step(action)

        # todo: improve, return only dictinary
        self.full_state["obs"] = next_obs
        if self.use_global_obs:
            self.full_state["states"] = self.env.get_state()
            return self.full_state, reward, is_done, info
        else:
            return self.full_state["obs"], reward, is_done, info

    def reset(self, env_ids=None):
        self.full_state["obs"] = self.env.reset(env_ids)
        if self.use_global_obs:
            self.full_state["states"] = self.env.get_state()
            return self.full_state
        else:
            return self.full_state["obs"]

    def get_number_of_agents(self):
        return self.env.get_number_of_agents()

    def get_env_info(self):
        info = {}
        info['action_space'] = self.env.action_space
        info['observation_space'] = self.env.observation_space
        info['amp_observation_space'] = self.env.amp_observation_space

        if self.use_global_obs:
            info['state_space'] = self.env.state_space
            print(info['action_space'], info['observation_space'], info['state_space'])
        else:
            print(info['action_space'], info['observation_space'])

        return info

vecenv.register('RLGPU', lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))
env_configurations.register('rlgpu', {
    'env_creator': lambda **kwargs: create_rlgpu_env(**kwargs),
    'vecenv_type': 'RLGPU'})

def build_alg_runner(algo_observer):
    runner = Runner(algo_observer)  #! make runner which control all isaacGym work

    """ 
        Factory is consisted of class of builder(algo : trainer, player : tester, model : )
        In this fn, runner register the following classes additionally
        In call runner.load(cfg_train) step, runner makes corresponding class objects written as cfg_train file(builder)
    """

    runner.algo_factory.register_builder('deepmm', lambda **kwargs : deepmm_agent.DeepmmAgent(**kwargs))
    runner.player_factory.register_builder('deepmm', lambda **kwargs : deepmm_players.DeepmmPlayerContinuous(**kwargs))
    runner.model_builder.model_factory.register_builder('deepmm', lambda network, **kwargs : deepmm_models.ModelDeepmmContinuous(network))  
    runner.model_builder.network_factory.register_builder('deepmm', lambda **kwargs : deepmm_network_builder.DeepmmBuilder())
    
    runner.algo_factory.register_builder('amp', lambda **kwargs : amp_agent.AMPAgent(**kwargs))
    runner.player_factory.register_builder('amp', lambda **kwargs : amp_players.AMPPlayerContinuous(**kwargs))
    runner.model_builder.model_factory.register_builder('amp', lambda network, **kwargs : amp_models.ModelAMPContinuous(network))  
    runner.model_builder.network_factory.register_builder('amp', lambda **kwargs : amp_network_builder.AMPBuilder())
    
    # runner.algo_factory.register_builder('ase', lambda **kwargs : ase_agent.ASEAgent(**kwargs))
    # runner.player_factory.register_builder('ase', lambda **kwargs : ase_players.ASEPlayer(**kwargs))
    # runner.model_builder.model_factory.register_builder('ase', lambda network, **kwargs : ase_models.ModelASEContinuous(network))  
    # runner.model_builder.network_factory.register_builder('ase', lambda **kwargs : ase_network_builder.ASEBuilder())
    
    # runner.algo_factory.register_builder('hrl', lambda **kwargs : hrl_agent.HRLAgent(**kwargs))
    # runner.player_factory.register_builder('hrl', lambda **kwargs : hrl_players.HRLPlayer(**kwargs))
    # runner.model_builder.model_factory.register_builder('hrl', lambda network, **kwargs : hrl_models.ModelHRLContinuous(network))  
    # runner.model_builder.network_factory.register_builder('hrl', lambda **kwargs : hrl_network_builder.HRLBuilder())
    
    return runner

def main():
    global args
    global cfg
    global cfg_train

    set_np_formatting()                             #! set np print option for debugging
    args = get_args()                               #! parse args and use it for isaacGym setting
    cfg, cfg_train, logdir = load_cfg(args)         #! divide args to cfg, cfg_train(dictionary from yml file loading), respectively

                                                    #! set seed, and cuDnn non-deterministic property
    cfg_train['params']['seed'] = set_seed(cfg_train['params'].get("seed", -1), cfg_train['params'].get("torch_deterministic", False))


    if args.horovod:                                #! manipulate Mutli-GPU case
        cfg_train['params']['config']['multi_gpu'] = args.horovod

    if args.horizon_length != -1:                   #! total simulation length regardless of env reset
        cfg_train['params']['config']['horizon_length'] = args.horizon_length

    if args.minibatch_size != -1:                   #! ppo optimization minibatch epoch update
        cfg_train['params']['config']['minibatch_size'] = args.minibatch_size
        
    if args.motion_file:                            #! set designated motion file
        cfg['env']['motion_file'] = args.motion_file
    
    # Create default directories for weights and statistics
    cfg_train['params']['config']['train_dir'] = args.output_path
    
 
    vargs = vars(args)                              #! convert args to dictionary

    algo_observer = RLGPUAlgoObserver()             #! make RLGPU env observer

    runner = build_alg_runner(algo_observer)        #! make Runner for RLtask with algo observer
    runner.load(cfg_train)                          #! set config in Runner to cfg_train
    runner.reset()                                  #! reset Runner, yet implemented nothing in this fn 
    runner.run(vargs)                               #! start run Runner with vargs file -> train / play w.r.t --train in args

    return

if __name__ == '__main__':
    main()
