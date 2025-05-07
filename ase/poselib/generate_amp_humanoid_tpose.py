import torch

from poselib.core.rotation3d import *
from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState
from poselib.visualization.common import plot_skeleton_state

"""
This scripts imports a MJCF XML file and converts the skeleton into a SkeletonTree format.
It then generates a zero rotation pose, and adjusts the pose into a T-Pose.
"""

# import MJCF file
xml_path = "../../../../assets/mjcf/amp_humanoid.xml"
skeleton = SkeletonTree.from_mjcf(xml_path)

# generate zero rotation pose
zero_pose = SkeletonState.zero_pose(skeleton)

# adjust pose into a T Pose
local_rotation = zero_pose.local_rotation
local_rotation[skeleton.index("left_upper_arm")] = quat_mul(
    quat_from_angle_axis(angle=torch.tensor([90.0]), axis=torch.tensor([1.0, 0.0, 0.0]), degree=True), 
    local_rotation[skeleton.index("left_upper_arm")]
)
local_rotation[skeleton.index("right_upper_arm")] = quat_mul(
    quat_from_angle_axis(angle=torch.tensor([-90.0]), axis=torch.tensor([1.0, 0.0, 0.0]), degree=True), 
    local_rotation[skeleton.index("right_upper_arm")]
)
translation = zero_pose.root_translation
translation += torch.tensor([0, 0, 0.9])

# save and visualize T-pose
zero_pose.to_file("data/amp_humanoid_tpose.npy")
plot_skeleton_state(zero_pose)