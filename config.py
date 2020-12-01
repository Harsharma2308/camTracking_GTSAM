import numpy as np

config = {
    "odom_noise": np.array([1, 1, 1, 0.5, 0.5, 0.5]),
    "prior_noise": np.array([0.3, 0.3, 0.3, 0.1, 0.1, 0.1]),
    "gps_noise": np.array([1, 1, 1]),
    "initial_pose": np.array([0, 0, 0, 1, 0, 0, 0]),
    "visualize": False,
    "weight_paths": ["./CMRNet/checkpoints/iter1.tar", "./CMRNet/checkpoints/iter2.tar", "./CMRNet/checkpoints/iter3.tar"],
    "path_to_map_folder": "./Maps",
    "path_to_dataset": "./CMRNet/KITTI_ODOMETRY",
    "plot_vo": False,
    "sequence": "00",
    "dataset_image_dir": "/home/stars/Documents/sof/slam/project/datasets/kitti/sequences/",
    "dataset_gt_poses_dir": "/home/stars/Documents/sof/slam/project/datasets/kitti/poses/",
    "length_traj": 300,
    "dataset_path": "/home/stars/Documents/sof/slam/project/datasets/kitti/",
    "seq": "00",
    "log_dir": "./Logs",
    "start_frame_num":1200
}
