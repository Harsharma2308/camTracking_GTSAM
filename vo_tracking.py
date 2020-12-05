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
    # create vo inference class
    vo_manager = VisualOdometryManager(config, vo_pose_init)
    vo_manager.initialize()

    # create a logger
    logger = Logger(config)

    # some initialisations
    start_frame_id = config["start_frame_num"]+2
    end_frame_id = config["start_frame_num"]+config["length_traj"]
    logger.write_record(vo_manager.vo.get_current_transform())

    # the main loop
    try:
        for img_id in tqdm(range(start_frame_id, end_frame_id)):
            current_pose, current_transform, delta_odom = vo_manager.update(
                img_id)
            logger.write_record(current_transform)
            # vo_manager.refine_pose(cmr_global_transform_estimate)
            print("##########################")
            print("vo_estimate", current_pose[:3])
            print("gt", kitti.poses[img_id][:3, -1])

            # visualisation code
            if config["plot_vo"]:
                vo_manager.plot(img_id)

        logger.close()
    except:
        logger.close()
