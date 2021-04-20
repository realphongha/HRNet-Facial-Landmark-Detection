import os
import sys
import random
import numpy as np
import argparse
from os.path import splitext, join


# sets up system arguments:
parser = argparse.ArgumentParser(description="Generate a single file containing all the filenames in a dataset.")
parser.add_argument("--dat", dest="dat", type=str, help="Dataset name")
parser.add_argument("--path", dest="path", type=str, help="Path to dataset")
parser.add_argument("--split", action="store_true", default=False, help="Split or not?")
parser.add_argument("--flip", action="store_true", default=False, help="Flip or not?")
parser.add_argument("--aug", action="store_true", default=False, help="Augmentation or not? (300W-LP)")
args = parser.parse_args()


def list_images_aflw2000(path):
    result_filename = "aflw2000.txt"
    dir_files = os.listdir(path)
    filenames = [file for file in dir_files if splitext(file)[1] == ".jpg"]
    with open(join(path, result_filename), "w") as result:
        for filename in filenames:
            result.write(filename + "\n")
        result.close()


def list_images_300w_lp(path, aug, flip):
    result_filename = "300w_lp_%s_aug.txt" if aug else "300w_lp_%s.txt"
    sub_folders = ['AFW', 'HELEN', 'LFPW', 'IBUG', 'AFW_Flip', 'HELEN_Flip', 'LFPW_Flip', 'IBUG_Flip'] if flip \
        else ['AFW', 'HELEN', 'LFPW', 'IBUG']
    files = []
    for folder in sub_folders:
        sub_folder_path = os.path.join(path, folder)
        dir_files = os.listdir(sub_folder_path)
        # only files with "_0.jpg" have right landmarks point:
        filenames = [file for file in dir_files if file[-4:] == ".jpg"] if aug else [file for file in dir_files if file[-6:] == "_0.jpg"]
        for filename in filenames:
            files.append(folder + "/" + filename)
        if args.split:
            random.shuffle(files)
            train, validate, test = files[:int(len(files) * 0.6)], files[int(len(files) * 0.6):int(len(files) * 0.8)], \
                                    files[int(len(files) * 0.8):]
            with open(join(path, result_filename % "train"), "w") as result:
                for filename in train:
                    result.write(filename + "\n")
                result.close()
            with open(join(path, result_filename % "val"), "w") as result:
                for filename in validate:
                    result.write(filename + "\n")
                result.close()
            with open(join(path, result_filename % "test"), "w") as result:
                for filename in test:
                    result.write(filename + "\n")
                result.close()
        else:
            with open(join(path, "300w_lp.txt"), "w") as result:
                for filename in train:
                    result.write(filename + "\n")
                result.close()


def list_mix_300w_lp_aflw2000(path, aug, flip):
    result_filename = "300w_lp_%s_aug.txt" if aug else "300w_lp_%s.txt"
    path_300w_lp = join(path, "300W_LP")
    path_aflw2000 = join(path, "AFLW2000")
    files = []

    sub_folders = ['AFW', 'HELEN', 'LFPW', 'IBUG', 'AFW_Flip', 'HELEN_Flip', 'LFPW_Flip', 'IBUG_Flip'] if flip \
        else ['AFW', 'HELEN', 'LFPW', 'IBUG']
    for folder in sub_folders:
        sub_folder_path = os.path.join(path_300w_lp, folder)
        dir_files = os.listdir(sub_folder_path)
        # only files with "_0.jpg" have right landmarks point:
        filenames = [file for file in dir_files if file[-4:] == ".jpg"] if aug else [file for file in dir_files if file[-6:] == "_0.jpg"]
        for filename in filenames:
            files.append(join("300W_LP", join(folder, filename)))
            # result.write(join("300W_LP", join(folder, filename)) + "\n")

    dir_files = os.listdir(path_aflw2000)
    filenames = [file for file in dir_files if splitext(file)[1] == ".jpg"]
    for filename in filenames:
        files.append(join("AFLW2000", filename))
        # result.write(join("AFLW2000", filename) + "\n")
    random.shuffle(files)
    train, validate, test = files[:int(len(files)*0.6)], files[int(len(files)*0.6):int(len(files)*0.8)], \
                            files[int(len(files)*0.8):]
    with open(join(path, result_filename % "train"), "w") as result:
        for filename in train:
            result.write(filename + "\n")
        result.close()
    with open(join(path, result_filename % "val"), "w") as result:
        for filename in validate:
            result.write(filename + "\n")
        result.close()
    with open(join(path, result_filename % "test"), "w") as result:
        for filename in test:
            result.write(filename + "\n")
        result.close()


def list_mix_300w_lp_wflw(path, aug, flip, ratio=(8, 2, 0)):
    result_filename = "300w_lp_%s_aug.txt" if aug else "300w_lp_%s.txt"
    path_300w_lp = join(path, "300W_LP")
    path_wflw = join(path, "WFLW/WFLW_annotations/list_98pt_rect_attr_train_test")
    files = []

    sub_folders = ['AFW', 'HELEN', 'LFPW', 'IBUG', 'AFW_Flip', 'HELEN_Flip', 'LFPW_Flip', 'IBUG_Flip'] if flip \
        else ['AFW', 'HELEN', 'LFPW', 'IBUG']
    for folder in sub_folders:
        sub_folder_path = join(path_300w_lp, folder)
        dir_files = os.listdir(sub_folder_path)
        # only files with "_0.jpg" have right landmarks point:
        filenames = [file for file in dir_files if file[-4:] == ".jpg"] if aug else [file for file in dir_files if file[-6:] == "_0.jpg"]
        for filename in filenames:
            files.append(join("300W_LP", join(folder, filename)))

    dir_files = ["list_98pt_rect_attr_test.txt", "list_98pt_rect_attr_train.txt"]
    for filename in dir_files:
        file = open(os.path.join(path_wflw, filename), "r")
        for line in file.read().splitlines():
            line = line.split()
            img_path = os.path.join("WFLW/WFLW_images", line[-1])
            marks = line[:196]
            line = [img_path]
            line.extend(marks)
            files.append(" ".join(line))
        file.close()
    random.shuffle(files)
    train, test, validate = files[:int(len(files)*ratio[0]/10)], files[int(len(files)*ratio[0]/10):int(len(files)*(ratio[0]+ratio[1])/10)], \
                            files[int(len(files)*(ratio[0]+ratio[1])/10):]
    with open(join(path, result_filename % "train"), "w") as result:
        for filename in train:
            result.write(filename + "\n")
        result.close()
    with open(join(path, result_filename % "val"), "w") as result:
        for filename in validate:
            result.write(filename + "\n")
        result.close()
    with open(join(path, result_filename % "test"), "w") as result:
        for filename in test:
            result.write(filename + "\n")
        result.close()


if __name__ == "__main__":
    if args.dat.upper() == "AFLW2000":
        list_images_aflw2000(args.path)
    elif args.dat.upper() == "300W_LP":
        list_images_300w_lp(args.path, args.aug, args.flip)
    elif args.dat.upper() == "MIX_300WLP_AFLW2000":
        list_mix_300w_lp_aflw2000(args.path, args.aug, args.flip)
    elif args.dat.upper() == "MIX_300WLP_WFLW":
        list_mix_300w_lp_wflw(args.path, args.aug, args.flip)
    else:
        print("This type of dataset is not supported.")
