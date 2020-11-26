import numpy as np

config = {
    "odom_noise": np.array([1, 1, 1, 0.5, 0.5, 0.5]),
    "prior_noise": np.array([0.3, 0.3, 0.3, 0.1, 0.1, 0.1]),
    "gps_noise": np.array([1, 1, 1]),
    "initial_pose": np.array([0, 0, 0, 1, 0, 0, 0]),
    "visualize":True,
    "weight_paths": ['./CMRNet/checkpoints/iter1.tar','./CMRNet/checkpoints/iter2.tar','./CMRNet/checkpoints/iter3.tar'],
    "path_to_map": "./map-00_0.1_0-300.pcd",
    "path_to_dataset": "./CMRNet/KITTI_ODOMETRY",
    "sequence": "00"
}