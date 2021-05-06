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
import torch.optim as optim
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import lib.models as models
from lib.config import config, update_config
from lib.datasets import get_dataset
from lib.core import function
from lib.utils import utils


def parse_args():

    parser = argparse.ArgumentParser(description='Train Face Alignment')

    parser.add_argument('--cfg', help='experiment configuration filename',
                        required=True, type=str)

    args = parser.parse_args()
    update_config(config, args)
    return args


def main():

    args = parse_args()

    logger, final_output_dir, tb_log_dir = \
        utils.create_logger(config, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.determinstic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    gpus = list(config.GPUS)

    model = models.hrnet_pose(config) if config.MODEL.RETURN_POSE else models.get_face_alignment_net(config)

    # copy model files
    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    os.environ["CUDA_VISIBLE_DEVICES"] = ", ".join(list(map(str, gpus)))
    model = nn.DataParallel(model, device_ids=list(range(len(gpus)))).cuda()

    # loss
    criterion = torch.nn.MSELoss(reduction="mean").cuda()

    optimizer = utils.get_optimizer(config, model)
    best_nme = 100
    last_epoch = config.TRAIN.BEGIN_EPOCH
    if config.TRAIN.RESUME:
        checkpoint = "latest.pth" if not config.TRAIN.CHECKPOINT else config.TRAIN.CHECKPOINT
        model_state_file = os.path.join(final_output_dir, checkpoint)
        print("Checkpoint file: %s" % model_state_file)
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file)
            last_epoch = checkpoint['epoch']
            best_nme = checkpoint['best_nme']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint (epoch {})"
                  .format(checkpoint['epoch']))
        else:
            print("=> no checkpoint found")

    if config.MODEL.PRETRAINED:
        print("Loading pretrained model from %s..." % config.MODEL.PRETRAINED)
        state_dict = torch.load(config.MODEL.PRETRAINED)
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

    if isinstance(config.TRAIN.LR_STEP, list):
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, config.TRAIN.LR_STEP,
            config.TRAIN.LR_FACTOR, last_epoch-1
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, config.TRAIN.LR_STEP,
            config.TRAIN.LR_FACTOR, last_epoch-1
        )
    dataset_type = get_dataset(config)

    train_loader = DataLoader(
        dataset=dataset_type(config,
                             ds_type="train", return_pose=config.MODEL.RETURN_POSE),
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU*len(gpus),
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY)

    val_loader = DataLoader(
        dataset=dataset_type(config,
                             ds_type="val", return_pose=config.MODEL.RETURN_POSE),
        batch_size=config.TEST.BATCH_SIZE_PER_GPU*len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY
    )

    # freezes weight:
    if config.MODEL.RETURN_POSE:
        model.module.freeze_weights(config.MODEL.FREEZE_BACKBONE, config.MODEL.FREEZE_CLF)

    for epoch in range(last_epoch, config.TRAIN.END_EPOCH):
        lr_scheduler.step()

        if config.MODEL.RETURN_POSE:
            function.train_pose(config, train_loader, model, criterion,
                           optimizer, epoch, writer_dict)
            # evaluate
            nme, predictions = function.validate_pose(config, val_loader, model,
                                                 criterion, epoch, writer_dict)
        else:
            function.train(config, train_loader, model, criterion,
                           optimizer, epoch, writer_dict)
            # evaluate
            nme, predictions = function.validate(config, val_loader, model,
                                                 criterion, epoch, writer_dict)



        is_best = nme < best_nme
        best_nme = min(nme, best_nme)

        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        print("best:", is_best)
        utils.save_checkpoint(
            {"state_dict": model,
             "epoch": epoch + 1,
             "best_nme": best_nme,
             "optimizer": optimizer.state_dict(),
             }, predictions, is_best, final_output_dir, 'checkpoint_{}.pth'.format(epoch))

    final_model_state_file = os.path.join(final_output_dir,
                                          'final_state.pth')
    logger.info('saving final model state to {}'.format(
        final_model_state_file))
    torch.save(model.module.state_dict(), final_model_state_file)
    writer_dict['writer'].close()


if __name__ == '__main__':
    main()
