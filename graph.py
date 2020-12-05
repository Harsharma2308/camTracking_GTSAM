import gtsam
import numpy as np
import matplotlib.pyplot as plt
import gtsam.utils.plot as gtsam_plot
from gtsam.utils.circlePose3 import *
from factor_graph_utils import *
from utils import matrix2posevec
from config import config
from scipy.spatial.transform import Rotation as R_scipy


class FactorGraph(object):
    '''
    A class to add factors and update the estimates
    '''
    def __init__(self,config,pose_3rd):
        quat = R_scipy.from_matrix(pose_3rd[:3, :3]).as_quat()
        quat = np.roll(quat, 1)
        t = pose_3rd[:3, -1]
        init_pose = np.array([*t, *quat])

        self.result = None
        self.marginals = None
        self.node_idx = 0

        self.odom_noise = gtsam.noiseModel.Diagonal.Sigmas(config["odom_noise"])
        self.skip_noise = gtsam.noiseModel.Diagonal.Sigmas(config["skip_noise"])
        self.prior_noise = gtsam.noiseModel.Diagonal.Sigmas(config["prior_noise"])
        self.gps_pose_noise = gtsam.noiseModel.Diagonal.Sigmas(config["gps_pose_noise"])
        self.gps_noise = gtsam.noiseModel.Isotropic.Sigmas(config["gps_noise"])
        self.skip_num = config["skip_num"]


        # self.visualize=config["visualize"]
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_estimate = gtsam.Values()
        self.init_pose = gen_pose(init_pose) #gtsam.Pose3(gtsam.Rot3.Rodrigues(*init_pose[3:]), gtsam.Point3(*init_pose[:3]))#gen_pose([0,0,0])

        self.parameters = gtsam.ISAM2Params()
        self.parameters.setRelinearizeThreshold(0.01)
        self.parameters.setRelinearizeSkip(10)
        self.isam = gtsam.ISAM2(self.parameters)

        #! Add a prior on pose x0
        self.graph.push_back(gtsam.PriorFactorPose3(X(self.node_idx), self.init_pose, self.prior_noise))
        self.initial_estimate.insert(X(self.node_idx), self.init_pose)
        self.current_estimate = None

        self.last_transform = None

    def add_skip_odom(self, delta_odom, prev_node_idx, cur_node_idx):
        '''
        params:
        delta_odom = 1*7 np.array(tx,ty,tz,w,x,y,z)
        '''
        delta = gen_pose(delta_odom)
        self.graph.add(gtsam.BetweenFactorPose3(X(prev_node_idx), X(cur_node_idx), delta, self.skip_noise))

    
    def add_odom(self, delta_odom, cur_pose_estimate):
        '''
        params:
        delta_odom = 1*7 np.array(tx,ty,tz,w,x,y,z)
        '''
        delta = gen_pose(delta_odom)
        cur_pose = gen_pose(cur_pose_estimate)
        self.graph.add(gtsam.BetweenFactorPose3(X(self.node_idx), X(self.node_idx+1), delta, self.odom_noise))
        self.initial_estimate.insert(X(self.node_idx+1), cur_pose)

    def add_gps(self, cur_pose):
        # self.graph.add(gtsam.GPSFactor(X(self.node_idx+1), gps_pose, self.gps_noise))
        gps_pose = gen_pose(cur_pose)
        self.graph.add(gtsam.PriorFactorPose3(X(self.node_idx+1), gps_pose, self.gps_pose_noise))

    def optimize(self):
        # print(self.graph)
        # print("#########################")
        # print(self.initial_estimate)
        # import ipdb; ipdb.set_trace()
        self.isam.update(self.graph, self.initial_estimate)
        self.current_estimate = self.isam.calculateEstimate()
        # print("###########################")
        # print(self.current_estimate)
        # self.graph.resize(0)
        self.initial_estimate.clear() # = self.current_estimate
        
    def update(self,state):
        delta_odom = state['delta_odom']
        delta_skip_odom = state['delta_skip_odom']
        delta_skip_odom_other = state['delta_skip_odom_other']
        cur_pose_estimate = state['cur_pose_estimate']
        cur_pose_gps = state['cur_pose_gps']
        
        #######################################################
        self.add_odom(delta_odom,cur_pose_estimate)
        
        #######################################################
        if(delta_skip_odom is not None and delta_skip_odom_other is not None):
            self.add_skip_odom(delta_skip_odom,self.node_idx+1-self.skip_num,self.node_idx+1)
            self.add_skip_odom(delta_skip_odom_other,self.node_idx+1-self.skip_num+2,self.node_idx+1)
        
        # self.add_gps(cur_pose_gps)
        #######################################################
        cmr_pose = state["cmr_global_transform"]
        cmr_pose_vec = matrix2posevec(cmr_pose)
        cmr_pose_vec[3:] = np.roll(cmr_pose_vec[3:],1)
        self.add_gps(cmr_pose_vec)
        #######################################################
        # if(self.visualize and self.initial_estimate is not None):
        #     self.plot()
        self.optimize()
        last_pose3 = self.current_estimate.atPose3(X(self.node_idx+1))
        self.last_transform = last_pose3.matrix()[:3,:]
        print("fg estimate:",self.last_transform[:3,-1].flatten())
        self.node_idx += 1
    
    def plot(self):
        visual_ISAM2_plot(self.current_estimate)


if __name__ == "__main__":
    graph_obj = FactorGraph(config,np.array([0,0,0,1,0,0,0]))
