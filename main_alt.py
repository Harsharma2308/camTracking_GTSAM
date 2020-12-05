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
    vo_manager = VisualOdometryManager(config,vo_pose_init)
    vo_manager.initialize()

    # create cmrnet inference class
    cmr_manager = RefineEstimate(config)

    # create factor graph object with a prior as the second pose
    fg = FactorGraph(config, fg_pose_init)
    
    # create a logger
    logger = Logger(config)
    fg_logger = Logger(config, prefix="factor_")

    # make axes
    if config["plot_cmr"]:
        axes = [plt.subplot(3, 1, i + 1) for i in range(3)]

    # some initialisations
    start_frame_id = config["start_frame_num"]+2
    end_frame_id = config["start_frame_num"]+config["length_traj"]
    logger.write_record(vo_manager.vo.get_current_transform())

    # the main loop
    for img_id in tqdm(range(start_frame_id,end_frame_id )):
        delta_skip_odom=delta_skip_odom_other=None
        img_rgb = kitti.get_cam2(img_id)
        
        current_pose, current_transform, delta_odom = vo_manager.update(img_id)
        #if((img_id-config["start_frame_num"]-1)%skip_num==0):
        #    prev_frame_id = img_id - skip_num
        #    other_prev_frame_id = img_id - skip_num + 2
        #    new_frame_id = img_id
        #    # new_frame = kitti.get_cam0(img_id)
        #    # prev_frame = kitti.get_cam0(img_id-skip_num)
        #    # delta_skip_odom = vo_manager.get_skip_delta_pose(new_frame_id,prev_frame_id)
        #    ##############################################################################
        #    # import ipdb; ipdb.set_trace()
        #    fg_idx = prev_frame_id - start_frame_id + 1
        #    prev_pose3 = fg.current_estimate.atPose3(X(fg_idx))
        #    cur_img = kitti.get_cam2(new_frame_id)
        #    delta_skip_transform, _ = cmr_manager.refine_pose_estimate(prev_pose3.matrix(), cur_img)
        #    # delta_skip_transform = np.linalg.inv(delta_skip_transform)
        #    delta_skip_odom = matrix2posevec(delta_skip_transform)
        #    delta_skip_odom[3:] = np.roll(delta_skip_odom[3:], 1)
        #    ##############################################################################
        #    delta_skip_odom_other = vo_manager.get_skip_delta_pose(new_frame_id,other_prev_frame_id)
        #    print("Skipping frames!", delta_skip_odom)
        gps_pos = cmr_global_transform_estimate = images = None
        # call cmrnet inference
        if (img_id%10 == 0):
            gps_pos, cmr_global_transform_estimate, images = cmr_manager.update(
                img_id, current_transform, img_rgb
            )
            if config["plot_cmr"]:
                for i in range(len(images)):
                    axes[i].imshow(images[i])
            plt.pause(0.5)

        # gps_pos = current_transform[:3,-1]
        logger.write_record(current_transform)
        # vo_manager.refine_pose(cmr_global_transform_estimate)
        print("##########################")
        print("vo_estimate", current_pose[:3])
        print("gps_estimate", gps_pos)
        print("gt", kitti.poses[img_id][:3, -1])
        # create a state
        state = {
            "delta_odom": delta_odom,
            "cur_pose_estimate": current_pose,
            "cur_pose_gps": gps_pos,
            "cmr_global_transform": cmr_global_transform_estimate,
            "delta_skip_odom": delta_skip_odom,
            "delta_skip_odom_other": delta_skip_odom_other
        }

        # update factor graph
        fg.update(state, resize = (img_id%100 == 0))
        vo_manager.refine_pose(fg.last_transform)

        # visualisation code
        if config["plot_vo"]:
            vo_manager.plot(img_id)

    if config["plot_fg"]:
        fg.plot()
    # print(fg.current_estimate)

    # wrte the factor graph into log and close
    import ipdb; ipdb.set_trace()
    fg_logger.write_factor_graph(fg.current_estimate)
    # print(fg.graph)
    logger.close()
    fg_logger.close()
