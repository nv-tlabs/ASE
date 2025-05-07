from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState
from poselib.visualization.common import plot_skeleton_state

# load in XML mjcf file and save zero rotation pose in npy format
xml_path = "../../../../assets/mjcf/nv_humanoid.xml"
skeleton = SkeletonTree.from_mjcf(xml_path)
zero_pose = SkeletonState.zero_pose(skeleton)
zero_pose.to_file("data/nv_humanoid.npy")

# visualize zero rotation pose
plot_skeleton_state(zero_pose)