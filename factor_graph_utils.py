import gtsam
import gtsam.utils.plot as gtsam_plot
from matplotlib import pyplot as plt
import numpy as np

def visual_ISAM2_plot(result):
    # Declare an id for the figure
    fignum = 0

    fig = plt.figure(fignum)
    axes = fig.gca(projection='3d')
    plt.cla()

    gtsam_plot.plot_3d_points(fignum, result, 'rx')

    # Plot cameras
    i = 0
    while result.exists(X(i)):
        pose_i = result.atPose3(X(i))
        gtsam_plot.plot_pose3(fignum, pose_i, 10)
        i += 1

    # draw
    axes.set_xlim3d(-40, 40)
    axes.set_ylim3d(-40, 40)
    axes.set_zlim3d(-40, 40)
    plt.pause(1)


def X(i):
    """Create key for pose i."""
    return int(gtsam.symbol('x', i))

def gen_pose(pose:np.ndarray):
    return gtsam.Pose3(r=gtsam.Rot3.Quaternion(*pose[3:]), t=gtsam.Point3(*pose[:3]))


