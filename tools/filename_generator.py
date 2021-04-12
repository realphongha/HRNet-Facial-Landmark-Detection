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
args = parser.parse_args()


def list_images_aflw2000(path):
    result_filename = "aflw2000.txt"
    dir_files = os.listdir(path)
    filenames = [file for file in dir_files if splitext(file)[1] == ".jpg"]
    with open(join(path, result_filename), "w") as result:
        for filename in filenames:
            result.write(filename + "\n")
        result.close()


def list_images_300w_lp(path):
    result_filename = "300w_lp_%s.txt"
    result = open(join(path, result_filename), "w")
    sub_folders = ['AFW', 'HELEN', 'LFPW', 'IBUG', 'AFW_Flip', 'HELEN_Flip', 'LFPW_Flip', 'IBUG_Flip']
    files = []
    for folder in sub_folders:
        sub_folder_path = os.path.join(path, folder)
        dir_files = os.listdir(sub_folder_path)
        # only files with "_0.jpg" have right landmarks point:
        filenames = [file for file in dir_files if file[-6:] == "_0.jpg"]
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


def list_mix_300w_lp_aflw2000(path):
    result_filename = "mix_%s.txt"
    path_300w_lp = join(path, "300W_LP")
    path_aflw2000 = join(path, "AFLW2000")
    files = []

    sub_folders = ['AFW', 'HELEN', 'LFPW', 'IBUG', 'AFW_Flip', 'HELEN_Flip', 'LFPW_Flip', 'IBUG_Flip']
    for folder in sub_folders:
        sub_folder_path = os.path.join(path_300w_lp, folder)
        dir_files = os.listdir(sub_folder_path)
        # only files with "_0.jpg" have right landmarks point:
        filenames = [file for file in dir_files if file[-6:] == "_0.jpg"]
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


if __name__ == "__main__":
    if args.dat.upper() == "AFLW2000":
        list_images_aflw2000(args.path)
    elif args.dat.upper() == "300W_LP":
        list_images_300w_lp(args.path)
    elif args.dat.upper() == "MIX":
        list_mix_300w_lp_aflw2000(args.path)
    else:
        print("This type of dataset is not supported.")
