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
import sys
import yaml

from isaacgym import gymapi
from isaacgym import gymutil

import numpy as np
import random
import torch

SIM_TIMESTEP = 1.0 / 60.0

def set_np_formatting():
    np.set_printoptions(edgeitems=30, infstr='inf',
                        linewidth=4000, nanstr='nan', precision=2,
                        suppress=False, threshold=10000, formatter=None)


def warn_task_name():
    raise Exception(
        "Unrecognized task!\nTask should be one of: [BallBalance, Cartpole, CartpoleYUp, Ant, Humanoid, Anymal, FrankaCabinet, Quadcopter, ShadowHand, ShadowHandLSTM, ShadowHandFFOpenAI, ShadowHandFFOpenAITest, ShadowHandOpenAI, ShadowHandOpenAITest, Ingenuity]")


def set_seed(seed, torch_deterministic=False):
    if seed == -1 and torch_deterministic:
        seed = 42
    elif seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch_deterministic:
        # refer to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.set_deterministic(True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    return seed


def load_cfg(args):
    with open(os.path.join(os.getcwd(), args.cfg_train), 'r') as f:
        cfg_train = yaml.load(f, Loader=yaml.SafeLoader)

    with open(os.path.join(os.getcwd(), args.cfg_env), 'r') as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    # Override number of environments if passed on the command line
    if args.num_envs > 0:
        cfg["env"]["numEnvs"] = args.num_envs

    if args.episode_length > 0:
        cfg["env"]["episodeLength"] = args.episode_length

    cfg["name"] = args.task                             #! augment task name == agent class name (eg, humanoidAMP)
    cfg["headless"] = args.headless

    # Set physics domain randomization
    if "task" in cfg:                                   #! not task key in all cfg file
        if "randomize" not in cfg["task"]:
            cfg["task"]["randomize"] = args.randomize
        else:
            cfg["task"]["randomize"] = args.randomize or cfg["task"]["randomize"]
    else:
        cfg["task"] = {"randomize": False}              #! cfg[task][randomize] = False

    logdir = args.logdir
    
    # Set deterministic mode
    if args.torch_deterministic:                            #! Default torch deterministic set as False
        cfg_train["params"]["torch_deterministic"] = True    

    exp_name = cfg_train["params"]["config"]['name']        #! Humanoid (for all ase, amp train yaml)

    if args.experiment != 'Base':                           #! Default is Base -> experiment name is Humanoid
        if args.metadata:
            exp_name = "{}_{}_{}_{}".format(args.experiment, args.task_type, args.device, str(args.physics_engine).split("_")[-1])

            if cfg["task"]["randomize"]:
                exp_name += "_DR"
        else:
             exp_name = args.experiment                     

    # Override config name
    cfg_train["params"]["config"]['name'] = exp_name

    if args.resume > 0:
        cfg_train["params"]["load_checkpoint"] = True

    if args.checkpoint != "Base":
        cfg_train["params"]["load_path"] = args.checkpoint
        
    if args.llc_checkpoint != "":
        cfg_train["params"]["config"]["llc_checkpoint"] = args.llc_checkpoint

    # Set maximum number of training iterations (epochs)
    if args.max_iterations > 0:
        cfg_train["params"]["config"]['max_epochs'] = args.max_iterations

    cfg_train["params"]["config"]["num_actors"] = cfg["env"]["numEnvs"]         #! numEnvs = num_actors(not agents)

    seed = cfg_train["params"].get("seed", -1)
    if args.seed is not None:
        seed = args.seed
    cfg["seed"] = seed
    cfg_train["params"]["seed"] = seed                                          #! synchronize seed for cfg and cfg_train

    cfg["args"] = args                                                          #! append args dictornary to cfg

    return cfg, cfg_train, logdir


def parse_sim_params(args, cfg, cfg_train):
    # initialize sim
    sim_params = gymapi.SimParams()
    sim_params.dt = SIM_TIMESTEP
    sim_params.num_client_threads = args.slices

    if args.physics_engine == gymapi.SIM_FLEX:
        if args.device != "cpu":
            print("WARNING: Using Flex with GPU instead of PHYSX!")
        sim_params.flex.shape_collision_margin = 0.01
        sim_params.flex.num_outer_iterations = 4
        sim_params.flex.num_inner_iterations = 10
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 0
        sim_params.physx.num_threads = 4
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.physx.num_subscenes = args.subscenes
        sim_params.physx.max_gpu_contact_pairs = 8 * 1024 * 1024

    sim_params.use_gpu_pipeline = args.use_gpu_pipeline
    sim_params.physx.use_gpu = args.use_gpu

    # if sim options are provided in cfg, parse them and update/override above:
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # Override num_threads if passed on the command line
    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads

    return sim_params


def get_args(benchmark=False):
    custom_parameters = [
        {"name": "--test", "action": "store_true", "default": False,
            "help": "Run trained policy, no training"},
        {"name": "--play", "action": "store_true", "default": False,
            "help": "Run trained policy, the same as test, can be used only by rl_games RL library"},
        {"name": "--resume", "type": int, "default": 0,
            "help": "Resume training or start testing from a checkpoint"},
        {"name": "--checkpoint", "type": str, "default": "Base",
            "help": "Path to the saved weights, only for rl_games RL library"},
        {"name": "--headless", "action": "store_true", "default": False,
            "help": "Force display off at all times"},
        {"name": "--horovod", "action": "store_true", "default": False,
            "help": "Use horovod for multi-gpu training, have effect only with rl_games RL library"},
        {"name": "--task", "type": str, "default": "Humanoid",
            "help": "Can be BallBalance, Cartpole, CartpoleYUp, Ant, Humanoid, Anymal, FrankaCabinet, Quadcopter, ShadowHand, Ingenuity"},
        {"name": "--task_type", "type": str,
            "default": "Python", "help": "Choose Python or C++"},
        {"name": "--rl_device", "type": str, "default": "cuda:0",
            "help": "Choose CPU or GPU device for inferencing policy network"},
        {"name": "--logdir", "type": str, "default": "logs/"},
        {"name": "--experiment", "type": str, "default": "Base",
            "help": "Experiment name. If used with --metadata flag an additional information about physics engine, sim device, pipeline and domain randomization will be added to the name"},
        {"name": "--metadata", "action": "store_true", "default": False,
            "help": "Requires --experiment flag, adds physics engine, sim device, pipeline info and if domain randomization is used to the experiment name provided by user"},
        {"name": "--cfg_env", "type": str, "default": "Base", "help": "Environment configuration file (.yaml)"},
        {"name": "--cfg_train", "type": str, "default": "Base", "help": "Training configuration file (.yaml)"},
        {"name": "--motion_file", "type": str,
            "default": "", "help": "Specify reference motion file"},
        {"name": "--num_envs", "type": int, "default": 0,
            "help": "Number of environments to create - override config file"},
        {"name": "--episode_length", "type": int, "default": 0,
            "help": "Episode length, by default is read from yaml config"},
        {"name": "--seed", "type": int, "help": "Random seed"},
        {"name": "--max_iterations", "type": int, "default": 0,
            "help": "Set a maximum number of training iterations"},
        {"name": "--horizon_length", "type": int, "default": -1,
            "help": "Set number of simulation steps per 1 PPO iteration. Supported only by rl_games. If not -1 overrides the config settings."},
        {"name": "--minibatch_size", "type": int, "default": -1,
            "help": "Set batch size for PPO optimization step. Supported only by rl_games. If not -1 overrides the config settings."},
        {"name": "--randomize", "action": "store_true", "default": False,
            "help": "Apply physics domain randomization"},
        {"name": "--torch_deterministic", "action": "store_true", "default": False,
            "help": "Apply additional PyTorch settings for more deterministic behaviour"},
        {"name": "--output_path", "type": str, "default": "output/", "help": "Specify output directory"},
        {"name": "--llc_checkpoint", "type": str, "default": "",
            "help": "Path to the saved weights for the low-level controller of an HRL agent."}]

    if benchmark:
        custom_parameters += [{"name": "--num_proc", "type": int, "default": 1, "help": "Number of child processes to launch"},
                              {"name": "--random_actions", "action": "store_true",
                                  "help": "Run benchmark with random actions instead of inferencing"},
                              {"name": "--bench_len", "type": int, "default": 10,
                                  "help": "Number of timing reports"},
                              {"name": "--bench_file", "action": "store", "help": "Filename to store benchmark results"}]

    # parse arguments
    args = gymutil.parse_arguments(
        description="RL Policy",
        custom_parameters=custom_parameters)                                #! parse args augmented with gym setting, default engine = PHYSX

    # allignment with examples
    args.device_id = args.compute_device_id                                 #! 0    in cuda:0
    args.device = args.sim_device_type if args.use_gpu_pipeline else 'cpu'  #! cuda in cuda:0

    if args.test:
        args.play = args.test
        args.train = False
    elif args.play:
        args.train = False
    else:
        args.train = True

    return args
