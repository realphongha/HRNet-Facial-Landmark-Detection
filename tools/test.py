# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com)
# ------------------------------------------------------------------------------

import os
import pprint
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import lib.models as models
from lib.config import config, update_config
from lib.utils import utils
from lib.datasets import get_dataset
from lib.core import function


def parse_args():

    parser = argparse.ArgumentParser(description='Train Face Alignment')

    parser.add_argument('--cfg', help='experiment configuration filename',
                        required=True, type=str)
    parser.add_argument('--model-file', help='model parameters', required=True, type=str)

    args = parser.parse_args()
    update_config(config, args)
    return args


def main():

    args = parse_args()

    logger, final_output_dir, tb_log_dir = \
        utils.create_logger(config, args.cfg, 'test')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.determinstic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    config.defrost()
    config.MODEL.INIT_WEIGHTS = False
    config.freeze()
    model = models.hrnet_pose(config) if config.MODEL.RETURN_POSE else models.get_face_alignment_net(config)

    gpus = list(config.GPUS)
    os.environ["CUDA_VISIBLE_DEVICES"] = ", ".join(list(map(str, gpus)))
    model = nn.DataParallel(model, device_ids=list(range(len(gpus)))).cuda()

    # load model
    state_dict = torch.load(args.model_file)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if "module." in k:
            name = k[7:]  # remove "module"
        else:
            name = k
        new_state_dict[name] = v
    state_dict = new_state_dict
    if 'state_dict' in state_dict.keys():
        state_dict = state_dict['state_dict'].module
        model.load_state_dict(state_dict)
    else:
        model.module.load_state_dict(state_dict)

    dataset_type = get_dataset(config)

    test_loader = DataLoader(
        dataset=dataset_type(config,
                             ds_type="test", return_pose=config.MODEL.RETURN_POSE),
        batch_size=config.TEST.BATCH_SIZE_PER_GPU*len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    if config.MODEL.RETURN_POSE:
        nme, predictions = function.test_pose(config, test_loader, model)
    else:
        nme, predictions = function.test(config, test_loader, model)

    torch.save(predictions, os.path.join(final_output_dir, 'predictions.pth'))


if __name__ == '__main__':
    main()
