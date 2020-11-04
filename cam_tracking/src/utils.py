from scipy.spatial.transform import Rotation as R
import numpy as np

def tf2matrix(trans,quat):
    T=np.eye(4)
    rot_mat = R.from_quat(quat).as_matrix()
    trans=np.array(trans).reshape(3,1)
    T[:3,:4] = np.hstack((rot_mat,trans))
    return T


def matrix2tf(T):
    rot_mat = T[:3,:3]
    quat = R.from_matrix(rot_mat).as_quat()
    trans=T[:3,3]
    return trans,quat
