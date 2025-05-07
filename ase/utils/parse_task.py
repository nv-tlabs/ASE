from env.tasks.humanoid import Humanoid
from env.tasks.humanoid_amp import HumanoidAMP
from env.tasks.humanoid_amp_getup import HumanoidAMPGetup
from env.tasks.humanoid_heading import HumanoidHeading
from env.tasks.humanoid_location import HumanoidLocation
from env.tasks.humanoid_strike import HumanoidStrike
from env.tasks.humanoid_reach import HumanoidReach
from env.tasks.humanoid_perturb import HumanoidPerturb
from env.tasks.humanoid_view_motion import HumanoidViewMotion
from env.tasks.vec_task_wrappers import VecTaskPythonWrapper

from isaacgym import rlgpu

import json
import numpy as np


def warn_task_name():
    raise Exception(
        "Unrecognized task!\nTask should be one of: [BallBalance, Cartpole, CartpoleYUp, Ant, Humanoid, Anymal, FrankaCabinet, Quadcopter, ShadowHand, ShadowHandLSTM, ShadowHandFFOpenAI, ShadowHandFFOpenAITest, ShadowHandOpenAI, ShadowHandOpenAITest, Ingenuity]")

def parse_task(args, cfg, cfg_train, sim_params):

    # create native task and pass custom config
    device_id = args.device_id
    rl_device = args.rl_device

    cfg["seed"] = cfg_train.get("seed", -1)
    cfg_task = cfg["env"]
    cfg_task["seed"] = cfg["seed"]

    try:
        task = eval(args.task)(
            cfg=cfg,
            sim_params=sim_params,
            physics_engine=args.physics_engine,
            device_type=args.device,
            device_id=device_id,
            headless=args.headless)
    except NameError as e:
        print(e)
        warn_task_name()
    env = VecTaskPythonWrapper(task, rl_device, cfg_train.get("clip_observations", np.inf), cfg_train.get("clip_actions", 1.0))

    return task, env
