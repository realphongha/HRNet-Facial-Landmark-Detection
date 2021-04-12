import cv2
import numpy as np

from lib.datasets.ds_300w_lp import DS_300W_LP
from lib.datasets.wflw import WFLW
from lib.utils.visualize import draw_axes_euler, draw_marks, draw_point

if __name__ == '__main__':
    from lib.config import config
    test_ds = "300w_lp"
    test_sample_range = range(100, 120)
    if test_ds == "300w_lp":
        config.DATASET.TRAINSET = r"E:\Workspace\data\face-direction\300W_LP\300w_lp_train.txt"
        config.DATASET.ROOT = r"E:\Workspace\data\face-direction\300W_LP"
        ds = DS_300W_LP(config)
        for i in test_sample_range:
            img, target, pose, meta = ds[i]
            img = img.transpose(1, 2, 0)
            center = meta["center"].numpy()
            marks = meta["tpts"].numpy()
            marks *= 4.0
            draw_marks(img, marks)
            draw_point(img, center)
            draw_axes_euler(img, pose[0], pose[1], pose[2], marks[33][0], marks[33][1])
            # print(target)
            # print(pose)
            # print(meta)
            cv2.imshow("Blah", img)
            cv2.waitKey()
    elif test_ds == "wflw":
        config.DATASET.TRAINSET = r"E:\Workspace\data\face-direction\WFLW\data\data\wflw\face_landmarks_wflw_train.csv"
        config.DATASET.ROOT = r"E:\Workspace\data\face-direction\WFLW\WFLW_images"
        ds = WFLW(config)
        for i in test_sample_range:
            img, target, meta = ds[i]
            img = img.transpose(1, 2, 0)
            center = meta["center"].numpy()
            marks = meta["tpts"].numpy()
            marks *= 4.0
            draw_marks(img, marks)
            draw_point(img, center)
            print(target)
            print(meta)
            cv2.imshow("Blah", img)
            cv2.waitKey()

