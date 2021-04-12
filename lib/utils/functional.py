from math import atan2

import numpy as np
import scipy.io as sio


def read_mat(mat_path, pt3d=False):
    # Get facebox, 2D landmarks and Euler angles from .mat files
    mat = sio.loadmat(mat_path)
    if pt3d:
        pt = mat['pt3d_68']
    else:
        pt = mat['pt2d']

    landmarks = np.array(list(zip(pt[0], pt[1])))

    pre_pose_params = mat['Pose_Para'][0]
    # Get [pitch, yaw, roll]
    pose = pre_pose_params[:3]

    x_min = min(pt[0, :])
    y_min = min(pt[1, :])
    x_max = max(pt[0, :])
    y_max = max(pt[1, :])

    pitch = pose[0] * 180 / np.pi
    yaw = pose[1] * 180 / np.pi
    roll = pose[2] * 180 / np.pi

    return np.array(landmarks), np.array((yaw, pitch, roll)), (x_min, y_min, x_max, y_max)
