from cmrnet_inference import RefineEstimate
from graph import FactorGraph
from config import config
import pykitti
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R_scipy
from visual_odometry import VisualOdometryManager
from tqdm import tqdm
import matplotlib.pyplot as plt
from logger import Logger
from factor_graph_utils import X
from utils import matrix2posevec

if __name__ == "__main__":
    # create a kitti reader
    kitti = pykitti.odometry(config["dataset_path"], config["seq"])
    vo_pose_init = kitti.poses[config["start_frame_num"]]
    fg_pose_init = kitti.poses[config["start_frame_num"]+1]
    skip_num = config["skip_num"]
    # create vo inference class
    vo_manager = VisualOdometryManager(config, vo_pose_init)
    vo_manager.initialize()

    # create cmrnet inference class
    cmr_manager = RefineEstimate(config)

    # create factor graph object with a prior as the second pose
    fg = FactorGraph(config, fg_pose_init)

    # create a logger
    logger = Logger(config)
    fg_logger = Logger(config, prefix="factor_")

    # make axes
    if config["plot_vo"]:
        axes = [plt.subplot(3, 1, i + 1) for i in range(3)]

    # some initialisations
    start_frame_id = config["start_frame_num"]+2
    end_frame_id = config["start_frame_num"]+config["length_traj"]
    logger.write_record(vo_manager.vo.get_current_transform())
    current_transform = kitti.poses[start_frame_id]
    # the main loop
    try:
        for img_id in tqdm(range(start_frame_id, end_frame_id)):
            delta_skip_odom = delta_skip_odom_other = None
            img_rgb = kitti.get_cam2(img_id)

            gps_pos, current_transform, images = cmr_manager.update(
                img_id, current_transform, img_rgb
            )
            logger.write_record(current_transform)

            print("##########################")
            print("gps_estimate:", gps_pos)
            print("gt:          ", kitti.poses[img_id][:3, -1])

            if config["plot_cmr"]:
                for i in range(len(images)):
                    axes[i].imshow(images[i])
                plt.pause(0.5)

        if config["plot_fg"]:
            fg.plot()
        # print(fg.graph)
        logger.close()
        fg_logger.close()
    except:
        logger.close()
        fg_logger.close()
