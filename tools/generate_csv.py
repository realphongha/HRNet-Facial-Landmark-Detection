import math

import pandas as pd
import numpy as np
import argparse
import os


parser = argparse.ArgumentParser(description="Generate csv file from datasets.")
parser.add_argument("--dat", dest="dat", type=str, help="Dataset name")
parser.add_argument("--path", dest="path", type=str, help="Path to dataset")
parser.add_argument("--opath", dest="output_path", type=str, help="Path to save output csv files")
args = parser.parse_args()


def read_annotation_file_wflw(path, annotation_list):
    with open(path, "r") as file:
        for line in file.read().splitlines():
            line = line.split()
            landmarks = list(map(float, line[:196]))
            np_landmarks = np.array(list(zip(landmarks[0::2], landmarks[1::2])))
            x_min = math.floor(np.min(np_landmarks[:, 0]))
            x_max = math.ceil(np.max(np_landmarks[:, 0]))
            y_min = math.floor(np.min(np_landmarks[:, 1]))
            y_max = math.ceil(np.max(np_landmarks[:, 1]))
            filename = line[-1]
            scale = max(x_max - x_min, y_max - y_min) / 200.0
            center_w = (x_max + x_min) / 2.0
            center_h = (y_max + y_min) / 2.0
            sample = [filename, scale, center_w, center_h]
            sample.extend(landmarks)
            annotation_list.append(sample)


def wflw_generator(dataset_path, output_path):
    path = os.path.join(dataset_path, "WFLW_annotations")
    path = os.path.join(path, "list_98pt_rect_attr_train_test")
    train_path = os.path.join(path, "list_98pt_rect_attr_train.txt")
    test_path = os.path.join(path, "list_98pt_rect_attr_test.txt")
    train = list()
    test = list()
    title = ["image_name", "scale", "center_w", "center_h"]
    for i in range(98):
        title.extend(["original_%d_x" % i, "original_%d_y" % i])
    read_annotation_file_wflw(train_path, train)
    read_annotation_file_wflw(test_path, test)
    train_pd = pd.DataFrame(train, columns=title)
    test_pd = pd.DataFrame(test, columns=title)
    output_train = os.path.join(output_path, "face_landmarks_wflw_train.csv")
    train_pd.to_csv(output_train, index=False)
    output_test = os.path.join(output_path, "face_landmarks_wflw_test.csv")
    test_pd.to_csv(output_test, index=False)


if __name__ == "__main__":
    if args.dat.lower() == "wflw":
        wflw_generator(args.path, args.output_path)
    else:
        raise Exception("Dataset type not supported!")