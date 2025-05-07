import os
import json

from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
from poselib.visualization.common import plot_skeleton_state, plot_skeleton_motion_interactive

# source fbx file path
fbx_file = "data/01_01_cmu.fbx"

# import fbx file - make sure to provide a valid joint name for root_joint
motion = SkeletonMotion.from_fbx(
    fbx_file_path=fbx_file,
    root_joint="Hips",
    fps=60
)

# save motion in npy format
motion.to_file("data/01_01_cmu.npy")

# visualize motion
plot_skeleton_motion_interactive(motion)
