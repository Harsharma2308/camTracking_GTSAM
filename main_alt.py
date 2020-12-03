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

if __name__ == "__main__":
    # create a kitti reader
    kitti = pykitti.odometry(config["dataset_path"], config["seq"])
    #pose_init = kitti.poses[2]
    pose_init = kitti.poses[config["start_frame_num"]]
    # create vo inference class
    vo_manager = VisualOdometryManager(config,pose_init)
    vo_manager.initialize()
    # create cmrnet inference class
    cmr_manager = RefineEstimate(config)

    

    # create factor graph object with a prior as the third pose
    #TODO: channge this
    # fg = FactorGraph(config, pose_init)
    
    # create a logger
    logger = Logger(config["log_dir"])
    # the main loop
    axes = [plt.subplot(3, 1, i + 1) for i in range(3)]
    start_frame_id = config["start_frame_num"]+2
    end_frame_id = config["start_frame_num"]+config["length_traj"]
    for img_id in tqdm(range(start_frame_id,end_frame_id )):
        img_rgb = kitti.get_cam2(img_id)
        current_pose, current_transform, delta_odom = vo_manager.update(img_id)
        # call cmrnet inference
        gps_pos, cmr_global_transform_estimate, images = cmr_manager.update(
            img_id, current_transform, img_rgb
        )
        logger.write_record(cmr_global_transform_estimate)
        vo_manager.refine_pose(cmr_global_transform_estimate)
        print("##########################")
        print("vo_estimate", current_pose[:3])
        print("gps_estimate", gps_pos)
        print("gt", kitti.poses[img_id][:3, -1])
        # create a state
        state = {
            "delta_odom": delta_odom,
            "cur_pose_estimate": current_pose,
            "cur_pose_gps": gps_pos,
        }
        # fg.update(state)

        # visualisation code
        if config["plot_vo"]:
            for i in range(len(images)):
                axes[i].imshow(images[i])
                plt.pause(0.2)
            vo_manager.plot(img_id)

    if config["visualize"]:
        fg.plot()
    # print(fg.current_estimate)
    logger.close()
