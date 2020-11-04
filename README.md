

# camTracking
Tracking only using cameras with a prior lidar map. We pose the problem on a factor graph.

### ROS Melodic Ubuntu 18.04

## Installation
### Conda environment
```
conda env create -f conda_env.yml --name slam
```

### Create ROS-workspace
```
mkdir -p ros_ws/src
cd ros_ws/src
```

### For ROS Melodic with Python 2, compile ros/geometry with python3-
```
cd ros_ws
wstool init
wstool set -y src/geometry2 --git https://github.com/ros/geometry2 -v 0.6.5
wstool up
rosdep install --from-paths src --ignore-src -y -r
catkin_make --cmake-args -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=~/anaconda3/envs/slam/bin/python3  -DPYTHON_INCLUDE_DIR=~/anaconda3/envs/slam/include/python3.7m    -DPYTHON_LIBRARY=~/anaconda3/envs/slam/lib/libpython3.7m.so
```

### Compile this package
```
cd ros_ws/src
git clone https://github.com/Harsharma2308/camTracking_GTSAM.git
cd ..
catkin_make --cmake-args -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=~/anaconda3/envs/slam/bin/python3  -DPYTHON_INCLUDE_DIR=~/anaconda3/envs/slam/include/python3.7m    -DPYTHON_LIBRARY=~/anaconda3/envs/slam/lib/libpython3.7m.so
```


### Usage (Make sure to match topics)
```
rosbag play --loop kitti.bag
roslaunch cam_tracking run.launch
python run.py
```


