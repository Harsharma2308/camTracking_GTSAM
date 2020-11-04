import sys
paths = ['/home/stars/Documents/sof/slam/project/ros_ws/devel/lib/python3/dist-packages', '/home/stars/anaconda3/envs/slam/lib/python37.zip', '/home/stars/anaconda3/envs/slam/lib/python3.7', '/home/stars/anaconda3/envs/slam/lib/python3.7/lib-dynload', '/home/stars/anaconda3/envs/slam/lib/python3.7/site-packages']
sys.path.extend(paths);sys.path.reverse()
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

class TrackerNode:
    def __init__(self,name):
        rospy.init_node(name)
        self._orb_pose=None
        self._lidar_pose=None
        self._refined_pose=None
        self._map_frameid='map'
        self._camera_frameid='camera'
        # self._orb_pose_sub = rospy.Subscriber('/orb_slam2_mono/pose',geometry_msgs.msg.TransformStamped,self._orb_pose_grabber)
        self._orb_pcd_sub = rospy.Subscriber('/orb_slam2_mono/map_points',sensor_msgs.msg.PointCloud2,self._lidar_matching)
        self._track()
    
    def _lidar_matching(self,pcd_msg):
        rospy.loginfo("Got scan, performing icp")
        if(self._orb_pose is not None):
            rospy.loginfo("Got scan, performing icp")

        #Dummy pcd as same pcd
        pc = ros_numpy.numpify(pcd_msg)
        points=np.zeros((pc.shape[0],3))
        
        points[:,0]=pc['x']
        points[:,1]=pc['y']
        points[:,2]=pc['z']
        
        pcd = o3d.geometry.PointCloud()
        pcd.points =o3d.utility.Vector3dVector(points)
        
        source=pcd;target=pcd
        
        reg_p2p = o3d.registration.registration_icp(
        source, target, 0.02, np.eye(4),
        o3d.registration.TransformationEstimationPointToPoint())
        print(reg_p2p)
        print("Transformation is:")
        print(reg_p2p.transformation)
        self._lidar_pose=reg_p2p.transformation
        



    def _orb_pose_grabber(self,msg):
        
        br = tf2_ros.TransformBroadcaster()
        t = geometry_msgs.msg.TransformStamped()

        t.header.stamp = rospy.Time.now()
        t.header.frame_id = self._map_frameid
        t.child_frame_id = self._camera_frameid
        x = msg.position.x
        y = msg.position.y
        z = msg.position.z
        
        q = (msg.orientation.x,msg.orientation.y,msg.orientation.z,msg.orientation.w)

        
        self._orb_pose = (x,y,z,q)
        
        self._refine_pose()
        
        
        (x,y,z,q)=self._refined_pose
        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.translation.z = z
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        br.sendTransform(t)
    
    def _refine_pose(self,name):
        self._refined_pose = self._orb_pose
    
    
    
    def _track(self):

        while not rospy.is_shutdown():
            continue
            ## if we want to listen to tf
            # try:
            #     (trans,rot) = listener.lookupTransform(self._map_frameid, self._camera_frameid, rospy.Time(0))
            # except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            #     continue




if __name__ == '__main__':
    tracker=TrackerNode('tracking_node')