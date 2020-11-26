from cmrnet_inference import RefineEstimate
from graph import FactorGraph
from config import config
import pykitti
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R_scipy
from visual_odometry import PinholeCamera, VisualOdometry
from tqdm import tqdm

if __name__ == '__main__':
    # create the config dictionary
    cmrnet_config = {
        "weight_paths": ['./CMRNet/checkpoints/iter1.tar','./CMRNet/checkpoints/iter2.tar','./CMRNet/checkpoints/iter3.tar'],
        "path_to_map": "./map-00_0.1_0-300.pcd",
        "path_to_dataset": "./CMRNet/KITTI_ODOMETRY",
        "sequence": "00"
    }
    # create cmrnet inference class
    # cmr_manager = RefineEstimate(config)
    kitti = pykitti.odometry("/home/arcot/Projects/SLAM_Project/dataset", "00")
    # create vo inference class
    dataset_image_dir = '/home/arcot/Projects/SLAM_Project/dataset/sequences/'
    dataset_gt_poses_dir = '/home/arcot/Projects/SLAM_Project/dataset/poses/'

    cam = PinholeCamera(1241.0, 376.0, 718.8560, 718.8560, 607.1928, 185.2157)
    vo = VisualOdometry(cam, dataset_gt_poses_dir+'00.txt')
    traj = np.zeros((600, 600, 3), dtype=np.uint8)

    length_traj = 4541
    trajectory = np.empty((length_traj, 3))
    gt_trajectory = np.empty((length_traj, 3))

    # create factor graph object
    pose_3rd = kitti.poses[2]
    quat = R_scipy.from_matrix(pose_3rd[:3,:3]).as_quat()
    t = pose_3rd[:3,-1]
    initial_pose =  np.array([*t, *quat])
    fg = FactorGraph(config, initial_pose)
    # the main loop
    for img_id in tqdm(range(length_traj)):
        img_rgb = cv2.imread(dataset_image_dir+'00/image_2/' +
                        str(img_id).zfill(6)+'.png')
        img=cv2.imread(dataset_image_dir+'00/image_0/' +
                        str(img_id).zfill(6)+'.png',0)

        vo.update(img, img_id)
        cur_t = vo.cur_t
        if img_id >= 2:
            x, y, z = cur_t[0], cur_t[1], cur_t[2]
            trajectory[img_id] = cur_t.flatten()
            gt_trajectory[img_id] = vo.trueX, vo.trueY, vo.trueZ
            # call cmrnet inference
            current_transform = vo.get_current_transform()
            current_pose = vo.get_current_pose()
            # refinement, _ = cmr_manager.refine_pose_estimate(current_transform, img_rgb)
            cmr_global_transform_estimate = current_transform # @ refinement
            gps_pos = cmr_global_transform_estimate[:3,-1]
            state = {
                "delta_odom":vo.delta_odom,
                "cur_pose_estimate":current_pose,
                "cur_pose_gps":gps_pos
            }
            fg.update(state)
        else:
            x, y, z = 0., 0., 0.
        


        # visualisation code
        # draw_x, draw_y = int(x)+290, int(z)+90
        # true_x, true_y = int(vo.trueX)+290, int(vo.trueZ)+90

        # cv2.circle(traj, (draw_x, draw_y), 1, 
        #         (img_id*255/(length_traj-1), 255-img_id*255/(length_traj-1), 0), 1)
        # cv2.circle(traj, (true_x, true_y), 1, (0, 0, 255), 2)
        # cv2.rectangle(traj, (10, 20), (600, 60), (0, 0, 0), -1)
        # text = "Coordinates: x=%2fm y=%2fm z=%2fm" % (x, y, z)
        # cv2.putText(traj, text, (20, 40), cv2.FONT_HERSHEY_PLAIN,
        #             1, (255, 255, 255), 1, 8)

        # cv2.imshow('Road facing camera', img)
        # cv2.imshow('Trajectory', traj)

        # cv2.waitKey(1)
