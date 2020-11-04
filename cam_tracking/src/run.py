import rospy
import math
import tf
import geometry_msgs.msg
import tf2_ros
import ipdb; ipdb.set_trace()


class TrackerNode:
    def __init__(self,name):
        rospy.init_node(name)
        self._orb_pose=None
        self._lidar_pose=None
        self._refined_pose=None
        self._map_frameid='map'
        self._camera_frameid='camera'
        self._orb_pose_subscriber = rospy.Subscriber('/orb_slam2_mono/pose',geometry_msgs.msg.TransformStamped,self._orb_pose_grabber)
        self._track()
    
    
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