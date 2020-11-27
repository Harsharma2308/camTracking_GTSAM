import gtsam
import numpy as np
import matplotlib.pyplot as plt
import gtsam.utils.plot as gtsam_plot
from gtsam.utils.circlePose3 import *
from factor_graph_utils import *
from config import config
class FactorGraph(object):
    '''
    A class to add factors and update the estimates
    '''
    def __init__(self,config,init_pose):
        self.result = None
        self.marginals = None
        self.node_idx = 0

        self.odom_noise = gtsam.noiseModel.Diagonal.Sigmas(config["odom_noise"])
        self.prior_noise = gtsam.noiseModel.Diagonal.Sigmas(config["prior_noise"])
        self.gps_noise = gtsam.noiseModel.Isotropic.Sigmas(config["gps_noise"])
        self.visualize=config["visualize"]
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_estimate = gtsam.Values()
        self.init_pose = gen_pose(init_pose) #gtsam.Pose3(gtsam.Rot3.Rodrigues(*init_pose[3:]), gtsam.Point3(*init_pose[:3]))#gen_pose([0,0,0])

        self.parameters = gtsam.ISAM2Params()
        self.parameters.setRelinearizeThreshold(0.01)
        self.parameters.setRelinearizeSkip(1)
        self.isam = gtsam.ISAM2(self.parameters)

        #! Add a prior on pose x0
        self.graph.push_back(gtsam.PriorFactorPose3(X(self.node_idx), self.init_pose, self.prior_noise))
        self.initial_estimate.insert(X(self.node_idx), self.init_pose)
        self.current_estimate = None

        self.count_index = 0
        self.est_pose = []
        self.pre_pose = None

    def add_odom(self, delta_odom, cur_pose_estimate):
        '''
        params:
        delta_odom = 1*7 np.array(tx,ty,tz,w,x,y,z)
        '''
        delta = gen_pose(delta_odom)#gtsam.Pose3(r = gtsam.Rot3.Quaternion(*delta_odom[3:]), t = delta_odom[:3])
        cur_pose = gen_pose(cur_pose_estimate)
        self.graph.add(gtsam.BetweenFactorPose3(X(self.node_idx), X(self.node_idx+1), delta, self.odom_noise))
        # if self.count_index == 0:
        self.initial_estimate.insert(X(self.node_idx+1), cur_pose)
        # else:
        #     last_id = self.current_estimate.size()-1
        #     self.initial_estimate.insert(X(self.node_idx+1), self.current_estimate.atPose3(X(last_id)).compose(delta))
        
        # self.count_index += 1

    def add_gps(self, cur_pose):
        gps_pose = gtsam.Point3(*cur_pose)
        self.graph.add(gtsam.GPSFactor(X(self.node_idx), gps_pose, self.gps_noise))
        
    def optimize(self):
        self.isam.update(self.graph, self.initial_estimate)
        self.current_estimate = self.isam.calculateEstimate()
        self.graph.resize(0)
        self.initial_estimate.clear()
        
    def update(self,state):
        delta_odom = state['delta_odom']
        cur_pose_estimate = state['cur_pose_estimate']
        cur_pose_gps = state['cur_pose_gps']
        self.add_odom(delta_odom,cur_pose_estimate)
        self.add_gps(cur_pose_gps)
        # if(self.visualize and self.initial_estimate is not None):
        #     self.plot()
        self.optimize()
        self.node_idx += 1
    
    def plot(self):
        visual_ISAM2_plot(self.current_estimate)


if __name__ == "__main__":
    graph_obj = FactorGraph(config,np.array([0,0,0,1,0,0,0]))
