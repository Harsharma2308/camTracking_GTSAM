from scipy.spatial.transform import Rotation as R_scipy
import numpy as np
def matrix2posevec(T_mat):
    t=T_mat[:3,-1]
    rot_mat=T_mat[:3,:3]
    quat = R_scipy.from_matrix(rot_mat).as_quat() #qx,qy,qz,qw
    return np.array([*t,*quat])
