import gtsam
import gtsam.utils.plot as gtsam_plot
from matplotlib import pyplot as plt
import numpy as np

import pickle



def visual_ISAM2_plot(result):
    # Declare an id for the figure
    fignum = 0
    fig = None
    fig = plt.figure(fignum)
    axes = fig.gca(projection='3d')
    plt.cla()

    gtsam_plot.plot_trajectory(fignum, result)

    # Plot cameras
    i = 0
    while result.exists(X(i)):
        pose_i = result.atPose3(X(i))
        gtsam_plot.plot_pose3(fignum, pose_i, 1)
        i += 1

    # draw
    # axes.set_xlim3d(-40, 40)
    # axes.set_ylim3d(-40, 40)
    # axes.set_zlim3d(-40, 40)
    # plt.show()
    axes.view_init(elev=180,azim=90)
    plt.savefig("Figures/traj.png")
    pickle.dump(fig, open("Figures/fig.pickle","wb"))
    plt.show()


def X(i):
    """Create key for pose i."""
    return int(gtsam.symbol('x', i))

def gen_pose(pose:np.ndarray):
    return gtsam.Pose3(r=gtsam.Rot3.Quaternion(*pose[3:]), t=gtsam.Point3(*pose[:3]))
