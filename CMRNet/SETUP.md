# Steps to setup CMRNet
* git clone https://github.com/cattaneod/CMRNet.git
## Setup Virtual Env
```
conda create -n 'cmrnet' python=3.6
conda activate cmrnet
```
## Install blender-utils
```
cd ~
git clone https://gitlab.com/ideasman42/blender-mathutils.git
cd blender-mathutils
python setup.py build
sudo python setup.py install
```
## Install dependencies from requirements.txt

* Remove the "git" line and "-e" lines from the requirements.txt
* cd <Path to CMR repo>
* pip install -r requirements.txt
* pip install -e ./
* pip install -e ./models/CMRNet/correlation_package

## To create h5 files
 python preprocess/kitti_maps.py --sequence 00 --kitti_folder ./KITTI_ODOMETRY/ --start 300

## To run
 python evaluate_iterative_single_CALIB.py with test_sequence=00 maps_folder=local_maps data_folder=./KITTI_ODOMETRY/sequences weight="['./checkpoints/iter1.tar','./checkpoints/iter2.tar','./checkpoints/iter3.tar']"

