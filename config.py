import numpy as np

config = {
    "odom_noise": np.array([1, 1, 1, 0.5, 0.5, 0.5]),
    "prior_noise": np.array([0.3, 0.3, 0.3, 0.1, 0.1, 0.1]),
    "gps_noise": np.array([1, 1, 1]),
    "initial_pose": np.array([0, 0, 0, 1, 0, 0, 0]),
    "visualize": True,
    "weight_paths": ["./CMRNet/checkpoints/iter1.tar", "./CMRNet/checkpoints/iter2.tar", "./CMRNet/checkpoints/iter3.tar"],
    "path_to_map_folder": "./Maps",
    "path_to_dataset": "./CMRNet/KITTI_ODOMETRY",
    "plot_vo": False,
    "sequence": "00",
    "dataset_image_dir": "/home/arcot/Projects/SLAM_Project/dataset/sequences/",
    "dataset_gt_poses_dir": "/home/arcot/Projects/SLAM_Project/dataset/poses/",
    "length_traj": 50,
    "dataset_path": "/home/arcot/Projects/SLAM_Project/dataset",
    "seq": "00",
    "log_dir": "./Logs",
    "start_frame_num":1200
}
