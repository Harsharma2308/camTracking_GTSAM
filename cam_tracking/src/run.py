import sys
import rospy
import math
import tf
import geometry_msgs.msg
import tf2_ros
import pcl_ros
import sensor_msgs.msg
import numpy as np
import open3d as o3d
import ros_numpy
from utils import tf2matrix, matrix2tf


class TrackerNode:
    def __init__(self, name):
        rospy.init_node(name)
        self._orb_pose = None
        self._lidar_pose = None
        self._refined_pose = None
        self._map_frameid = "map"
        self._camera_frameid = "camera_link"
        self._tf_listener = tf.TransformListener()
        self._tf_broadcaster = tf2_ros.TransformBroadcaster()
        # self._orb_pose_sub = rospy.Subscriber('/orb_slam2_mono/pose',geometry_msgs.msg.TransformStamped,self._orb_pose_grabber)
        self._orb_pcd_sub = rospy.Subscriber(
            "/orb_slam2_mono/map_points",
            sensor_msgs.msg.PointCloud2,
            self._lidar_matching,
        )

        self._track()

    def _lidar_matching(self, sparse_orb_pcd):
        ### Added in one function to avoid asynchronous callbacks; TODO to fix this
        self._orb_pose = None
        try:
            (trans, rot) = self._tf_listener.lookupTransform(
                self._map_frameid, self._camera_frameid, rospy.Time(0)
            )
            self._orb_pose = tf2matrix(trans, rot)
            # print("Transformation found:{},{}".format(trans,rot))
        except (
            tf.LookupException,
            tf.ConnectivityException,
            tf.ExtrapolationException,
        ):
            pass
        if self._orb_pose is not None:

            rospy.loginfo("Got sparse pcd, performing icp ")

            # Dummy pcd as same pcd
            pc = ros_numpy.numpify(sparse_orb_pcd)
            points = np.zeros((pc.shape[0], 3))

            points[:, 0] = pc["x"]
            points[:, 1] = pc["y"]
            points[:, 2] = pc["z"]

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)

            source = pcd
            target = pcd

            reg_p2p = o3d.registration.registration_icp(
                source,
                target,
                0.02,
                np.eye(4),
                o3d.registration.TransformationEstimationPointToPoint(),
            )

            # print("Transformation is:")
            # print(reg_p2p.transformation)
            lidar_cam_pose = reg_p2p.transformation
            self._refined_pose = lidar_cam_pose @ self._orb_pose
            self._publish_pose()

    def _publish_pose(self):
        (x, y, z), q = matrix2tf(self._refined_pose)
        msg_tf = geometry_msgs.msg.TransformStamped()
        msg_tf.header.stamp = rospy.Time.now()
        msg_tf.header.frame_id = self._map_frameid
        msg_tf.child_frame_id = "camera_link_refined"

        msg_tf.transform.translation.x = x
        msg_tf.transform.translation.y = y
        msg_tf.transform.translation.z = z
        msg_tf.transform.rotation.x = q[0]
        msg_tf.transform.rotation.y = q[1]
        msg_tf.transform.rotation.z = q[2]
        msg_tf.transform.rotation.w = q[3]

        self._tf_broadcaster.sendTransform(msg_tf)
        rospy.loginfo("Refined pose published")

    # def _orb_pose_grabber(self,msg):

    #     t = geometry_msgs.msg.TransformStamped()

    #     x = msg.position.x
    #     y = msg.position.y
    #     z = msg.position.z

    #     q = (msg.orientation.x,msg.orientation.y,msg.orientation.z,msg.orientation.w)

    #     self._refine_pose()

    def _refine_pose(self, name):
        self._refined_pose = self._orb_pose

    def _track(self):

        while not rospy.is_shutdown():
            ## on synchornous fix, main loop
            # try:
            #     (trans,rot) = self._tf_listener.lookupTransform(self._map_frameid, self._camera_frameid, rospy.Time(0))
            #     self._orb_pose=tf2matrix(trans,rot)
            #     # print("Transformation found:{},{}".format(trans,rot))
            # except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue


if __name__ == "__main__":
    tracker = TrackerNode("tracking_node")

