from scipy.spatial.transform import Rotation as R_scipy
import numpy as np


def matrix2posevec(T_mat):
    t = T_mat[:3, -1]
    rot_mat = T_mat[:3, :3]
    quat = R_scipy.from_matrix(rot_mat).as_quat()  # qx,qy,qz,qw
    return np.array([*t, *quat])


def posevec2matrix(pose_vec):
    t = pose_vec[:3]
    rot_mat = R_scipy.from_quat(rot_mat).as_matrix()
    T = np.eye(4)
    T[:3, :3] = rot_mat
    T[:3, -1] = t
    return T
