

# camTracking
Tracking only using cameras with a prior lidar map. We pose the problem on a factor graph.

### ROS Melodic Ubuntu 18.04

### Create ROS-workspace
```
mkdir -p ros_ws/src
cd ros_ws/src
```
```
git clone https://github.com/Harsharma2308/camTracking_GTSAM.git
cd ..
catkin_make

```


### Usage (Make sure to match topics)
```
rosbag play --loop MH_01_easy.bag
roslaunch orb_slam_2_ros test.launch
```