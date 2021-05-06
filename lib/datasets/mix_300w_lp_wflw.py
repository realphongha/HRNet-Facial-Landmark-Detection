import math
import os
import random

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image

from lib.utils.functional import read_mat, conv_98p_to_68p
from ..utils.transforms import fliplr_joints, crop, generate_target, transform_pixel


class Mix300WLP_WFLW(data.Dataset):
    def __init__(self, cfg, ds_type="train", transform=None):
        # specify annotation file for dataset
        if ds_type == "train":
            self.filenames = cfg.DATASET.TRAINSET
        elif ds_type == "val":
            self.filenames = cfg.DATASET.VALSET
        elif ds_type == "test":
            self.filenames = cfg.DATASET.TESTSET
        else:
            raise NotImplementedError("Dataset type %s is not implemented!" % ds_type)

        self.is_train = (ds_type == "train")
        self.transform = transform
        self.data_root = cfg.DATASET.ROOT
        self.input_size = cfg.MODEL.IMAGE_SIZE
        self.output_size = cfg.MODEL.HEATMAP_SIZE
        self.sigma = cfg.MODEL.SIGMA
        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rot_factor = cfg.DATASET.ROT_FACTOR
        self.label_type = cfg.MODEL.TARGET_TYPE
        self.flip = cfg.DATASET.FLIP

        # load annotations
        self.images = []
        self.landmarks = []
        for line in open(self.filenames, "r").read().splitlines():
            if "300W_LP" in line:
                filename = line
                file_path = os.path.join(self.data_root, filename)
                mat_path = file_path.replace("jpg", "mat")
                landmarks, _, _ = read_mat(mat_path, pt3d=False)
                self.images.append(file_path)
                self.landmarks.append(landmarks)
            elif "WFLW" in line:
                line = line.split()
                filename = line[0]
                file_path = os.path.join(self.data_root, filename)
                self.images.append(file_path)
                raw_landmarks = np.array(list(map(float, line[1:])))
                self.landmarks.append(conv_98p_to_68p(raw_landmarks.reshape((-1, 2))))

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        pose = self.pose[idx]
        x_min = math.floor(np.min(self.landmarks[idx][:, 0]))
        x_max = math.ceil(np.max(self.landmarks[idx][:, 0]))
        y_min = math.floor(np.min(self.landmarks[idx][:, 1]))
        y_max = math.ceil(np.max(self.landmarks[idx][:, 1]))
        scale = max(x_max - x_min, y_max - y_min) / 200.0
        center_w = (x_max + x_min) / 2.0
        center_h = (y_max + y_min) / 2.0
        center = torch.Tensor([center_w, center_h])

        pts = self.landmarks[idx]

        scale *= 1.25
        nparts = pts.shape[0]
        img = np.array(Image.open(image_path).convert('RGB'), dtype=np.float32)

        r = 0
        if self.is_train:
            scale = scale * (random.uniform(1 - self.scale_factor,
                                            1 + self.scale_factor))
            r = random.uniform(-self.rot_factor, self.rot_factor) \
                if random.random() <= 0.6 else 0
            if random.random() <= 0.5 and self.flip:
                img = np.fliplr(img)
                pts = fliplr_joints(pts, width=img.shape[1], dataset='300W')
                center[0] = img.shape[1] - center[0]
                if self.return_pose:
                    # flips Euler angles:
                    pose[0], pose[2] = -pose[0], -pose[2]
            if r != 0:
                if self.return_pose:
                    # rotates Euler angles:
                    pose[2] -= r

        img = crop(img, center, scale, self.input_size, rot=r)

        target = np.zeros((nparts, self.output_size[0], self.output_size[1]))
        tpts = pts.copy()

        for i in range(nparts):
            if tpts[i, 1] > 0:
                tpts[i, 0:2] = transform_pixel(tpts[i, 0:2] + 1, center,
                                               scale, self.output_size, rot=r)
                target[i] = generate_target(target[i], tpts[i] - 1, self.sigma,
                                            label_type=self.label_type)
        img = img.astype(np.float32)
        img = (img / 255.0 - self.mean) / self.std
        img = img.transpose([2, 0, 1])
        target = torch.Tensor(target)
        tpts = torch.Tensor(tpts)
        center = torch.Tensor(center)

        meta = {'index': idx, 'center': center, 'scale': scale,
                'pts': torch.Tensor(pts), 'tpts': tpts}
        return img, target, meta


if __name__ == "__main__":
    pass
