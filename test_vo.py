import numpy as np
import cv2

from visual_odometry import PinholeCamera, VisualOdometry

dataset_image_dir = '/home/arcot/Projects/SLAM_Project/dataset/sequences/'
dataset_gt_poses_dir = '/home/arcot/Projects/SLAM_Project/dataset/poses/'

cam = PinholeCamera(1241.0, 376.0, 718.8560, 718.8560, 607.1928, 185.2157)
vo = VisualOdometry(cam, dataset_gt_poses_dir+'00.txt')
traj = np.zeros((600, 600, 3), dtype=np.uint8)

length_traj = 4541
trajectory = np.empty((length_traj, 3))
gt_trajectory = np.empty((length_traj, 3))

for img_id in np.arange(length_traj):
    img = cv2.imread(dataset_image_dir+'00/image_0/' +
                     str(img_id).zfill(6)+'.png', 0)

    vo.update(img, img_id)
    cur_t = vo.cur_t
    if(img_id >= 2):
        x, y, z = cur_t[0], cur_t[1], cur_t[2]
        trajectory[img_id] = cur_t.flatten()
        gt_trajectory[img_id] = vo.trueX, vo.trueY, vo.trueZ
    else:
        x, y, z = 0., 0., 0.
    draw_x, draw_y = int(x)+290, int(z)+90
    true_x, true_y = int(vo.trueX)+290, int(vo.trueZ)+90

    cv2.circle(traj, (draw_x, draw_y), 1, 
               (img_id*255/(length_traj-1), 255-img_id*255/(length_traj-1), 0), 1)
    cv2.circle(traj, (true_x, true_y), 1, (0, 0, 255), 2)
    cv2.rectangle(traj, (10, 20), (600, 60), (0, 0, 0), -1)
    text = "Coordinates: x=%2fm y=%2fm z=%2fm" % (x, y, z)
    cv2.putText(traj, text, (20, 40), cv2.FONT_HERSHEY_PLAIN,
                1, (255, 255, 255), 1, 8)

    cv2.imshow('Road facing camera', img)
    cv2.imshow('Trajectory', traj)

    cv2.waitKey(1)

np.save("trajectory.npy", trajectory)
np.save("gt_trajectory.npy", gt_trajectory)
cv2.imwrite('map.png', traj)
