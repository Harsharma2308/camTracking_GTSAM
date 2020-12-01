import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R_scipy
from numpy.linalg import inv
from utils import *

STAGE_FIRST_FRAME = 0
STAGE_SECOND_FRAME = 1
STAGE_DEFAULT_FRAME = 2
kMinNumFeature = 1500

lk_params = dict(
    winSize=(21, 21),
    # maxLevel = 3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)


def featureTracking(image_ref, image_cur, px_ref):
    kp2, st, err = cv2.calcOpticalFlowPyrLK(
        image_ref, image_cur, px_ref, None, **lk_params
    )  # shape: [k,2] [k,1] [k,1]

    st = st.reshape(st.shape[0])
    kp1 = px_ref[st == 1]
    kp2 = kp2[st == 1]

    return kp1, kp2


class PinholeCamera:
    def __init__(
        self, width, height, fx, fy, cx, cy, k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0
    ):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.distortion = abs(k1) > 0.0000001
        self.d = [k1, k2, p1, p2, k3]


class VisualOdometry:
    def __init__(self, cam, annotations, init_T):
        self.frame_stage = 0
        self.cam = cam
        self.new_frame = None
        self.last_frame = None
        self.cur_R = None
        self.cur_t = None
        # 7x1 delta pose vector between consecutive frames represented in prev frame
        self.delta_odom = np.empty(7,)
        self.px_ref = None
        self.px_cur = None
        self.focal = cam.fx
        self.init_R = init_T[:3,:3]
        self.init_t = init_T[:3,-1]
        self.pp = (cam.cx, cam.cy)
        self.trueX, self.trueY, self.trueZ = 0, 0, 0
        self.detector = cv2.FastFeatureDetector_create(
            threshold=25, nonmaxSuppression=True
        )
        with open(annotations) as f:
            self.annotations = f.readlines()

    def getAbsoluteScale(self, frame_id):  # specialized for KITTI odometry dataset
        ss = self.annotations[frame_id - 1].strip().split()
        x_prev = float(ss[3])
        y_prev = float(ss[7])
        z_prev = float(ss[11])
        ss = self.annotations[frame_id].strip().split()
        x = float(ss[3])
        y = float(ss[7])
        z = float(ss[11])
        self.trueX, self.trueY, self.trueZ = x, y, z
        return np.sqrt(
            (x - x_prev) * (x - x_prev)
            + (y - y_prev) * (y - y_prev)
            + (z - z_prev) * (z - z_prev)
        )

    def processFirstFrame(self):
        self.px_ref = self.detector.detect(self.new_frame)
        self.px_ref = np.array([x.pt for x in self.px_ref], dtype=np.float32)
        self.frame_stage = STAGE_SECOND_FRAME

    def processSecondFrame(self):
        self.px_ref, self.px_cur = featureTracking(
            self.last_frame, self.new_frame, self.px_ref
        )
        E, mask = cv2.findEssentialMat(
            self.px_cur,
            self.px_ref,
            focal=self.focal,
            pp=self.pp,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0,
        )
        _, self.cur_R, self.cur_t, mask = cv2.recoverPose(
            E, self.px_cur, self.px_ref, focal=self.focal, pp=self.pp
        )
        
        self.cur_t += (self.cur_R @ self.init_t).reshape(3,1)
        self.cur_R = self.cur_R @ self.init_R
        
        self.frame_stage = STAGE_DEFAULT_FRAME
        self.px_ref = self.px_cur

    def processFrame(self, frame_id):
        self.px_ref, self.px_cur = featureTracking(
            self.last_frame, self.new_frame, self.px_ref
        )
        E, mask = cv2.findEssentialMat(
            self.px_cur,
            self.px_ref,
            focal=self.focal,
            pp=self.pp,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0,
        )
        _, R, t, mask = cv2.recoverPose(
            E, self.px_cur, self.px_ref, focal=self.focal, pp=self.pp
        )
        absolute_scale = self.getAbsoluteScale(frame_id)

        prev_R = self.cur_R
        prev_t = self.cur_t
        T_prev_w = np.eye(4)
        T_prev_w[:3, :4] = np.concatenate((self.cur_R, self.cur_t), axis=1)

        if absolute_scale > 0.1:
            self.cur_t = self.cur_t + absolute_scale * self.cur_R.dot(t)
            self.cur_R = R.dot(self.cur_R)
        if self.px_ref.shape[0] < kMinNumFeature:
            self.px_cur = self.detector.detect(self.new_frame)
            self.px_cur = np.array([x.pt for x in self.px_cur], dtype=np.float32)
        self.px_ref = self.px_cur
        T_cur_w = np.eye(4)
        T_cur_w[:3, :4] = np.concatenate((self.cur_R, self.cur_t), axis=1)

        # transformation representing movement from previous frame to current frame, in previous frame
        self.delta_odom = inv(T_prev_w) @ T_cur_w
        self.delta_odom = matrix2posevec(self.delta_odom)
        # quat stored as qw,qx,qy,qz
        self.delta_odom[3:] = np.roll(self.delta_odom[3:], 1)

    def update(self, img, frame_id):
        assert (
            img.ndim == 2
            and img.shape[0] == self.cam.height
            and img.shape[1] == self.cam.width
        ), "Frame: provided image has not the same size as the camera model or image is not grayscale"
        self.new_frame = img
        if self.frame_stage == STAGE_DEFAULT_FRAME:
            self.processFrame(frame_id)
        elif self.frame_stage == STAGE_SECOND_FRAME:
            self.processSecondFrame()
        elif self.frame_stage == STAGE_FIRST_FRAME:
            self.processFirstFrame()
        self.last_frame = self.new_frame

    def get_current_pose(self):
        if self.cur_t is None:
            return None
        quat = R_scipy.from_matrix(self.cur_R).as_quat()  # x,y,z,w
        quat = np.roll(quat, 1)
        t = self.cur_t.flatten()
        return np.array([*t, *quat])

    def get_current_transform(self):
        if self.cur_t is None:
            return None
        R = np.zeros((4, 4))
        R[-1, -1] = 1
        R[:3, :3] = self.cur_R
        T = np.eye(4)
        T[:3, -1] = self.cur_t.flatten()
        return T @ R


class VisualOdometryManager(object):
    def __init__(self, config, init_T):
        # create vo inference class
        self.dataset_image_dir = config["dataset_image_dir"]
        self.dataset_gt_poses_dir = config["dataset_gt_poses_dir"]
        self.start_frame_num=config["start_frame_num"]
        cam = PinholeCamera(1241.0, 376.0, 718.8560, 718.8560, 607.1928, 185.2157)
        self.vo = VisualOdometry(cam, self.dataset_gt_poses_dir + "00.txt",init_T)
        self.traj_bg = np.zeros((600, 600, 3), dtype=np.uint8)

        self.length_traj = config["length_traj"]
        self.trajectory = np.empty((self.start_frame_num+self.length_traj, 3))
        self.gt_trajectory = np.empty((self.start_frame_num+self.length_traj, 3))

    def initialize(self):
        self.update(0)
        self.update(1)

    def refine_pose(self, gps_pose_transformation):
        self.vo.cur_t = gps_pose_transformation[:3, -1].reshape(3,1)
        self.vo.cur_R = gps_pose_transformation[:3, :3]

    def update(self, img_id):
        img = cv2.imread(
            self.dataset_image_dir + "00/image_0/" + str(img_id).zfill(6) + ".png", 0
        )

        self.vo.update(img, img_id)

        current_transform = None
        current_pose = None

        if img_id >= self.start_frame_num+2:
            self.trajectory[img_id] = self.vo.cur_t.flatten()
            self.gt_trajectory[img_id] = self.vo.trueX, self.vo.trueY, self.vo.trueZ
            current_transform = self.vo.get_current_transform()
            current_pose = self.vo.get_current_pose()

        return current_pose, current_transform, self.vo.delta_odom

    def plot(self, img_id):
        x, y, z = 0, 0, 0
        if self.vo.cur_t is not None:
            x, y, z = self.vo.cur_t[0], self.vo.cur_t[1], self.vo.cur_t[2]
        draw_x, draw_y = int(x) + 290, int(z) + 90
        true_x, true_y = int(self.vo.trueX) + 290, int(self.vo.trueZ) + 90

        cv2.circle(
            self.traj_bg,
            (draw_x, draw_y),
            1,
            (
                img_id * 255 / (self.length_traj - 1),
                255 - img_id * 255 / (self.length_traj - 1),
                0,
            ),
            1,
        )
        cv2.circle(self.traj_bg, (true_x, true_y), 1, (0, 0, 255), 2)
        cv2.rectangle(self.traj_bg, (10, 20), (600, 60), (0, 0, 0), -1)
        text = "Coordinates: x=%2fm y=%2fm z=%2fm" % (x, y, z)
        cv2.putText(
            self.traj_bg,
            text,
            (20, 40),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            (255, 255, 255),
            1,
            8,
        )

        cv2.imshow("Trajectory", self.traj_bg)

        cv2.waitKey(1)

