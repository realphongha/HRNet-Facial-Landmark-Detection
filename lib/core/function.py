# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from PIL import Image

import time
import logging

import torch
import numpy as np
import cv2

from .evaluation import decode_preds, compute_nme
from lib.utils import visualize

logger = logging.getLogger(__name__)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(config, train_loader, model, critertion, optimizer,
          epoch, writer_dict):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()
    nme_count = 0
    nme_batch_sum = 0

    end = time.time()

    for i, (inp, target, meta) in enumerate(train_loader):
        # measure data time
        data_time.update(time.time()-end)

        # compute the output
        output = model(inp)
        target = target.cuda(non_blocking=True)

        score_map = output.data.cpu()
        loss = critertion(output, target)

        preds = decode_preds(score_map, meta['center'], meta['scale'], [64, 64])

        nme_batch = compute_nme(preds, meta)
        nme_batch_sum += np.sum(nme_batch)
        nme_count += preds.size(0)

        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), inp.size(0))

        batch_time.update(time.time()-end)
        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=inp.size(0)/batch_time.val,
                      data_time=data_time, loss=losses)
            logger.info(msg)

            if writer_dict:
                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_loss', losses.val, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1

        end = time.time()
    nme = nme_batch_sum / nme_count
    msg = 'Train Epoch {} time:{:.4f} loss:{:.4f} nme:{:.4f}'\
        .format(epoch, batch_time.avg, losses.avg, nme)
    logger.info(msg)


def train_pose(config, train_loader, model, criterion, optimizer,
          epoch, writer_dict):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()
    mae_count = 0
    mae_sum = 0

    end = time.time()

    for i, (inp, _, pose, meta) in enumerate(train_loader):
        # measure data time
        data_time.update(time.time()-end)

        # compute the output
        output = model(inp)
        pose = pose.cuda(non_blocking=True).float()

        loss = criterion(output, pose)

        # MAE
        mae = torch.nn.L1Loss(reduction="mean").cuda()(output, pose)
        mae_sum += (mae * output.size(0))
        mae_count += output.size(0)

        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), inp.size(0))

        batch_time.update(time.time()-end)
        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=inp.size(0)/batch_time.val,
                      data_time=data_time, loss=losses)
            logger.info(msg)

            if writer_dict:
                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_loss', losses.val, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1

        end = time.time()
    msg = 'Train Epoch {} time:{:.4f} loss:{:.4f} mae:{:.4f}'\
        .format(epoch, batch_time.avg, losses.avg, mae_sum/mae_count)
    logger.info(msg)


def validate(config, val_loader, model, criterion, epoch, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = AverageMeter()

    num_classes = config.MODEL.NUM_JOINTS
    predictions = torch.zeros((len(val_loader.dataset), num_classes, 2))

    model.eval()

    nme_count = 0
    nme_batch_sum = 0
    count_failure_008 = 0
    count_failure_010 = 0
    end = time.time()

    with torch.no_grad():
        for i, (inp, target, meta) in enumerate(val_loader):
            data_time.update(time.time() - end)
            output = model(inp)
            target = target.cuda(non_blocking=True)

            score_map = output.data.cpu()
            # loss
            loss = criterion(output, target)

            preds = decode_preds(score_map, meta['center'], meta['scale'], [64, 64])
            # NME
            nme_temp = compute_nme(preds, meta)
            # Failure Rate under different threshold
            failure_008 = (nme_temp > 0.08).sum()
            failure_010 = (nme_temp > 0.10).sum()
            count_failure_008 += failure_008
            count_failure_010 += failure_010

            nme_batch_sum += np.sum(nme_temp)
            nme_count += preds.size(0)
            for n in range(score_map.size(0)):
                predictions[meta['index'][n], :, :] = preds[n, :, :]

            losses.update(loss.item(), inp.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    nme = nme_batch_sum / nme_count
    failure_008_rate = count_failure_008 / nme_count
    failure_010_rate = count_failure_010 / nme_count

    msg = 'Test Epoch {} time:{:.4f} loss:{:.4f} nme:{:.4f} [008]:{:.4f} ' \
          '[010]:{:.4f}'.format(epoch, batch_time.avg, losses.avg, nme,
                                failure_008_rate, failure_010_rate)
    logger.info(msg)

    if writer_dict:
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('valid_loss', losses.avg, global_steps)
        writer.add_scalar('valid_nme', nme, global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1

    return nme, predictions


def validate_pose(config, val_loader, model, criterion, epoch, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = AverageMeter()

    num_classes = config.MODEL.NUM_JOINTS
    predictions = torch.zeros((len(val_loader.dataset), num_classes, 2))

    model.eval()

    mae_count = 0
    mae_sum = 0
    end = time.time()

    with torch.no_grad():
        for i, (inp, _, pose, meta) in enumerate(val_loader):
            data_time.update(time.time() - end)

            # compute the output
            output = model(inp)
            pose = pose.cuda(non_blocking=True).float()

            loss = criterion(output, pose)

            # MAE
            mae = torch.nn.L1Loss(reduction="mean").cuda()(output, pose)
            mae_sum += (mae * output.size(0))
            mae_count += output.size(0)

            losses.update(loss.item(), inp.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    mae = mae_sum / mae_count

    msg = 'Test Epoch {} time:{:.4f} loss:{:.4f} mae:{:.4f}'.format(epoch, batch_time.avg, losses.avg, mae)
    logger.info(msg)

    if writer_dict:
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('valid_loss', losses.avg, global_steps)
        writer.add_scalar('valid_mae', mae, global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1

    return mae, predictions


def test(config, data_loader, model):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    num_classes = config.MODEL.NUM_JOINTS
    predictions = torch.zeros((len(data_loader.dataset), num_classes, 2))

    model.eval()

    nme_count = 0
    nme_batch_sum = 0
    count_failure_008 = 0
    count_failure_010 = 0
    end = time.time()

    with torch.no_grad():
        for i, (inp, target, meta) in enumerate(data_loader):
            data_time.update(time.time() - end)
            output = model(inp)
            score_map = output.data.cpu()
            preds = decode_preds(score_map, meta['center'], meta['scale'], [64, 64])

            # NME
            nme_temp = compute_nme(preds, meta)

            failure_008 = (nme_temp > 0.08).sum()
            failure_010 = (nme_temp > 0.10).sum()
            count_failure_008 += failure_008
            count_failure_010 += failure_010

            nme_batch_sum += np.sum(nme_temp)
            nme_count = nme_count + preds.size(0)
            for n in range(score_map.size(0)):
                predictions[meta['index'][n], :, :] = preds[n, :, :]

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    nme = nme_batch_sum / nme_count
    failure_008_rate = count_failure_008 / nme_count
    failure_010_rate = count_failure_010 / nme_count

    msg = 'Test Results time:{:.4f} loss:{:.4f} nme:{:.4f} [008]:{:.4f} ' \
          '[010]:{:.4f}'.format(batch_time.avg, losses.avg, nme,
                                failure_008_rate, failure_010_rate)
    logger.info(msg)

    return nme, predictions


def test_pose(config, data_loader, model):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    num_classes = config.MODEL.NUM_JOINTS
    predictions = torch.zeros((len(data_loader.dataset), 3))

    model.eval()

    sample_count = 0
    mae_yaw = 0
    mae_pitch = 0
    mae_roll = 0
    end = time.time()

    with torch.no_grad():
        for i, (inp, _, pose, meta) in enumerate(data_loader):
            data_time.update(time.time() - end)
            output = model(inp)
            pose = pose.cuda(non_blocking=True).float()

            # MAE
            mae_yaw += torch.nn.L1Loss(reduction="mean").cuda()(output[0], pose[0]) * output.size(0)
            mae_pitch += torch.nn.L1Loss(reduction="mean").cuda()(output[1], pose[1]) * output.size(0)
            mae_roll += torch.nn.L1Loss(reduction="mean").cuda()(output[2], pose[2]) * output.size(0)
            sample_count += output.size(0)

            for n in range(pose.size(0)):
                predictions[meta['index'][n], :] = pose[n, :]

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    mae_yaw /= sample_count
    mae_pitch /= sample_count
    mae_roll /= sample_count
    mae = (mae_yaw + mae_pitch + mae_roll) / 3

    msg = 'Test Results time:{:.4f}, yaw:{:.4f}, pitch:{:.4f}, roll:{:.4f}, mae:{:.4f}'.format(batch_time.avg,
                                                                                               mae_yaw, mae_pitch,
                                                                                               mae_roll, mae)
    logger.info(msg)

    return mae, predictions

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def process_img(raw_img):
    img = raw_img.astype(np.float32)
    img = (img / 255.0 - MEAN) / STD
    img = img.transpose([2, 0, 1])
    img = torch.Tensor(img)
    return torch.unsqueeze(img, 0) # add batch dimension

def inference_img_pose(config, model, args):
    input_size = (config.MODEL.IMAGE_SIZE[0], config.MODEL.IMAGE_SIZE[1])
    raw_img = np.array(Image.open(args.img).convert('RGB'), dtype=np.float32)
    raw_img = cv2.resize(raw_img, input_size)
    img = process_img(raw_img)
    model.eval()
    with torch.no_grad():
        output = model(img)
        y, p, r = output.detach().cpu().numpy()[0]
    if args.show or args.store:
        visualize.draw_axes_euler(raw_img, y, p, r)
    if args.show:
        cv2.imshow("Result", raw_img)
        cv2.waitKey()
    if args.store:
        filename = "output_inference/%d.jpg" % int(time.time())
        cv2.imwrite(filename, raw_img)
        print("Saving file to %s" % filename)
    return y, p, r