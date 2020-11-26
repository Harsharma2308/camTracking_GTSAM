"""
GTSAM Copyright 2010-2018, Georgia Tech Research Corporation,
Atlanta, Georgia 30332-0415
All Rights Reserved
Authors: Frank Dellaert, et al. (see THANKS for the full author list)
See LICENSE for the license information
Simple robot motion example, with prior and two odometry measurements
Author: Frank Dellaert
"""
# pylint: disable=invalid-name, E1101

from __future__ import print_function

import numpy as np

import gtsam

import matplotlib.pyplot as plt
import gtsam.utils.plot as gtsam_plot
from gtsam.utils.circlePose3 import *

# Create noise models
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([1, 1, 1, 0.5, 0.5, 0.5]))
PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([10, 10, 1]))
GPS_NOISE = gtsam.noiseModel.Isotropic.Sigmas(np.array([1, 1, 1]))

def X(i):
    """Create key for pose i."""
    return int(gtsam.symbol('x', i))

def L(j):
    """Create key for landmark j."""
    return int(gtsam.symbol('l', j))

def gen_pose(point):
    return gtsam.Pose3(gtsam.Rot3.Rodrigues(0.0,0.0,0.0), gtsam.Point3(point[0], point[1], point[2]))

def visual_ISAM2_plot(result, est_pose, loam_traj, cov):
    # Declare an id for the figure
    fignum = 0
    point = []
    fig = plt.figure(fignum)
    # axes = fig.gca(projection='3d')
    plt.cla()

    if loam_traj is not None:
        plt.plot(np.array(loam_traj)[:, 0], np.array(loam_traj)[:, 1], 'r')

    # gtsam_plot.plot_3d_points(fignum, result, 'rx')
    if len(est_pose)>1:
        if cov >0.4:
            plt.plot(np.array(est_pose)[-5:, 0], np.array(est_pose)[-5:, 1], 'yx')
        else:
            plt.plot(np.array(est_pose)[-5:, 0], np.array(est_pose)[-5:, 1], 'bx')
    i = 0
    while result.exists(X(i)):
        pose_i = result.atPose3(X(i))
        # gtsam_plot.plot_pose3(fignum, pose_i, 10)
        point.append(np.array([pose_i.x(), pose_i.y(), 0]))
        plt.plot(np.array(point)[:, 0], np.array(point)[:, 1], 'g')
        # gtsam_plot.plot_pose2(0, gtsam.Pose2(pose_i.x(), pose_i.y(), 0))
        i += 1
    plt.axis('equal')
    plt.pause(0.0000001)
    return point

class Factor(object):
    def __init__(self):
        # Create an empty nonlinear factor graph
        # self.delta = p0.between(p1)

        # self.graph = gtsam.NonlinearFactorGraph()
        # self.initial = gtsam.Values()
        # init_pose = gen_pose([0,0,0])
        # self.graph.add(gtsam.NonlinearEqualityPose3(1, init_pose))
        # self.initial.insert(1, init_pose)

        self.result = None
        self.marginals = None
        self.node_idx = 0

        #! Define Gtsam2
        parameters = gtsam.ISAM2Params()
        parameters.setRelinearizeThreshold(0.01)
        parameters.setRelinearizeSkip(1)
        self.isam = gtsam.ISAM2(parameters)

        #! Create a Factor Graph and Values to hold the new data
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_estimate = gtsam.Values()
        self.init_pose = gen_pose([0,0,0])

        #! Add a prior on pose x0
        pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.3, 0.3, 0.3, 0.1, 0.1, 0.1]))
        self.graph.push_back(gtsam.PriorFactorPose3(X(self.node_idx), self.init_pose, pose_noise))
        self.initial_estimate.insert(X(self.node_idx), self.init_pose)
        self.current_estimate = None

        self.count_index = 0
        self.est_pose = []
        self.pre_pose = None

    def cur_index(self):
        return self.node_idx

    def add_odom(self, delta, cur_pose):
        # TODO Current with fixed angle
        self.graph.add(gtsam.BetweenFactorPose3(X(self.node_idx), X(self.node_idx+1), delta, ODOMETRY_NOISE))
        if self.count_index == 0:
            self.initial_estimate.insert(X(self.node_idx+1), cur_pose)
        else:
            last_id = self.current_estimate.size()-1
            self.initial_estimate.insert(X(self.node_idx+1), self.current_estimate.atPose3(X(last_id)).compose(delta))
        self.node_idx += 1
        self.count_index += 1

        # TODO add jump connections

        if self.node_idx >=6:
            pose1 = self.current_estimate.atPose3(X(self.node_idx-6))
            pose2 = self.current_estimate.atPose3(X(self.node_idx-5))
            pose3 = self.current_estimate.atPose3(X(self.node_idx-4))
            pose4 = self.current_estimate.atPose3(X(self.node_idx-3))
            pose5 = self.current_estimate.atPose3(X(self.node_idx-2))
            pose6 = self.current_estimate.atPose3(X(self.node_idx-1))
            #* 1->3 2->4 3->5 4->6
            self.graph.add(gtsam.BetweenFactorPose3(X(self.node_idx-6), X(self.node_idx-4), pose1.between(pose3), ODOMETRY_NOISE))
            self.graph.add(gtsam.BetweenFactorPose3(X(self.node_idx-5), X(self.node_idx-3), pose2.between(pose4), ODOMETRY_NOISE))
            self.graph.add(gtsam.BetweenFactorPose3(X(self.node_idx-4), X(self.node_idx-2), pose3.between(pose5), ODOMETRY_NOISE))
            self.graph.add(gtsam.BetweenFactorPose3(X(self.node_idx-3), X(self.node_idx-1), pose4.between(pose6), ODOMETRY_NOISE))
            #* 1->4 2->5 3->6
            self.graph.add(gtsam.BetweenFactorPose3(X(self.node_idx-6), X(self.node_idx-3), pose1.between(pose4), ODOMETRY_NOISE))                        
            self.graph.add(gtsam.BetweenFactorPose3(X(self.node_idx-5), X(self.node_idx-2), pose2.between(pose5), ODOMETRY_NOISE))                        
            self.graph.add(gtsam.BetweenFactorPose3(X(self.node_idx-4), X(self.node_idx-1), pose3.between(pose6), ODOMETRY_NOISE))                        
            #* 1->5 2->6
            self.graph.add(gtsam.BetweenFactorPose3(X(self.node_idx-6), X(self.node_idx-2), pose1.between(pose5), ODOMETRY_NOISE))                        
            self.graph.add(gtsam.BetweenFactorPose3(X(self.node_idx-5), X(self.node_idx-1), pose2.between(pose6), ODOMETRY_NOISE))
            #* 1->6
            self.graph.add(gtsam.BetweenFactorPose3(X(self.node_idx-6), X(self.node_idx-1), pose1.between(pose6), ODOMETRY_NOISE))

    def add_vpr(self, pose_vpr, cov):
        GPS_NOISE = gtsam.noiseModel_Isotropic.Sigmas(np.array([20, 20, 10])*cov)
        gps = gtsam.Point3(pose_vpr[0], pose_vpr[1], 0.0)
        self.graph.add(gtsam.GPSFactor(X(self.node_idx), gps, GPS_NOISE))
        self.est_pose.append(pose_vpr)

    def optimize(self):
        # import pdb; pdb.set_trace()
        self.isam.update(self.graph, self.initial_estimate)
        self.isam.update()
        self.isam.update()
        self.isam.update()
        self.isam.update()
        self.isam.update()
        self.current_estimate = self.isam.calculateEstimate()

        self.graph.resize(0)
        self.initial_estimate.clear()

    def get_cur_estimate(self):
        i = 0
        points = []
        while self.current_estimate.exists(X(i)):
            pose_i = self.current_estimate.atPose3(X(i))
            # gtsam_plot.plot_pose3(fignum, pose_i, 10)
            points.append(np.array([pose_i.x(), pose_i.y(), 0]))
            i += 1
        return points

    def plot(self,loam_traj, cov):
        # fig = plt.figure(0)
        # point = []

        # for i in range(1, self.node_idx):
        #     pose = self.result.atPose3(i)
        #     point.append(np.array([pose.x(), pose.y(), 0]))
        #     gtsam_plot.plot_pose2(0, gtsam.Pose2(pose.x(), pose.y(), 0))
        # plt.axis('equal')
        # plt.show()
        # return point
        visual_ISAM2_plot(self.current_estimate, self.est_pose, loam_traj, cov)
