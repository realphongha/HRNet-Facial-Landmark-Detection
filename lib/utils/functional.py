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


def conv_98p_to_68p(l98):
    # converts 98 points landmarks annotation to 68 points type
    # mapping = list(range(0, 33, 2)) + list(range(33, 38)) + list(range(42, 47)) + list(range(51, 60)) \
    #           + list(range(60, 62)) + list(range(63, 66)) + [67] + list(range(68, 70)) \
    #           + list(range(71, 74)) + list(range(75, 96))
    mapping = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 33, 34, 35, 36, 37, 42, 43, 44, 45, 46,
               51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 63, 64, 65, 67, 68, 69, 71, 72, 73, 75, 76, 77, 78, 79, 80,
               81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95]
    return l98[mapping] if len(l98.shape) == 2 else l98[:, mapping]
