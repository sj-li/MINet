#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
# import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import imp
import yaml
import time
import collections
import copy
import cv2
import os
import os.path as osp
import numpy as np
from matplotlib import pyplot as plt

# from .utils.logger import Logger
from .utils.avgmeter import *
from .utils.sync_batchnorm.batchnorm import convert_model
from .utils.warmupLR import *
from .utils.ioueval import *
from .dataset.Parser import Parser
from .models import *
from .losses import *

def set_tensorboard(path):
    writer = SummaryWriter(path)
    return writer

class Trainer():
  def __init__(self, ARCH, DATA, datadir, logdir, logger, pretrained=None, use_mps=True):
    # parameters
    self.ARCH = ARCH
    self.DATA = DATA
    self.datadir = datadir
    self.log = logdir
    self.logger = logger
    self.pretrained = pretrained
    self.use_mps = use_mps

    self.writer = set_tensorboard(osp.join(logdir, 'tfrecord'))

    # get the data
    self.parser = Parser(root=self.datadir,
                         data_cfg = DATA,
                         arch_cfg = ARCH,
                         gt=True,
                         shuffle_train=True)

    # weights for loss (and bias)
    # weights for loss (and bias)
    epsilon_w = self.ARCH["train"]["epsilon_w"]
    content = torch.zeros(self.parser.get_n_classes(), dtype=torch.float)
    for cl, freq in DATA["content"].items():
      x_cl = self.parser.to_xentropy(cl)  # map actual class to xentropy class
      content[x_cl] += freq
    self.loss_w = 1 / (content + epsilon_w)   # get weights
    for x_cl, w in enumerate(self.loss_w):  # ignore the ones necessary to ignore
      if DATA["learning_ignore"][x_cl]:
        # don't weigh
        self.loss_w[x_cl] = 0

    content = torch.zeros(3, dtype=torch.float)
    for cl, freq in DATA["content"].items():
      if cl == 0:
          content[0] += freq
      elif cl > 250:
          content[2] += freq
      else:
          content[1] += freq

    self.loss_w_m = 1 / (content + epsilon_w)   # get weights

    self.logger.info("Loss weights from content: ", self.loss_w.data)

    self.model = get_model(ARCH['model']['name'])(ARCH['model']['in_channels'], self.parser.get_n_classes(), ARCH["model"]["dropout"])

    # GPU?
    self.gpu = False
    self.multi_gpu = False
    self.n_gpus = 0
    # self.model_single = self.model
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.logger.info("Training in device: ", self.device)
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
      cudnn.benchmark = True
      cudnn.fastest = True
      self.gpu = True
      self.n_gpus = 1
      self.model.cuda()
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
      self.logger.info("Let's use", torch.cuda.device_count(), "GPUs!")
      self.model = nn.DataParallel(self.model)   # spread in gpus
      self.model = convert_model(self.model).cuda()  # sync batchnorm
      # self.model_single = self.model.module  # single model to get weight names
      self.multi_gpu = True
      self.n_gpus = torch.cuda.device_count()

    weights_total = sum(p.numel() for p in self.model.parameters())
    weights_grad = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    self.logger.info("Total number of parameters: " + str(weights_total))
    self.logger.info("Total number of parameters requires_grad: " + str(weights_grad))


    # loss
    if "loss" in self.ARCH["train"].keys() and self.ARCH["train"]["loss"] == "xentropy":
      self.criterion = nn.NLLLoss(weight=self.loss_w).to(self.device)
      self.ls = Lovasz_softmax(ignore=0).to(self.device)
    else:
      raise Exception('Loss not defined in config file')

    if self.use_mps:
      self.criterion_e = nn.BCEWithLogitsLoss().to(self.device)
      self.criterion_m = nn.NLLLoss(weight=self.loss_w_m).to(self.device)
      self.criterion_d = Depth_Loss().to(self.device)


    # loss as dataparallel too (more images in batch)
    if self.n_gpus > 1:
      self.criterion = nn.DataParallel(self.criterion).cuda()  # spread in gpu
      self.ls = nn.DataParallel(self.ls).cuda()
      if self.use_mps:
        self.criterion_e = nn.DataParallel(self.criterion_e).cuda()
        self.criterion_m = nn.DataParallel(self.criterion_m).cuda()
        self.criterion_d = nn.DataParallel(self.criterion_d).cuda()

    # Use SGD optimizer to train
    # self.optimizer = optim.SGD(self.model_single.parameters(),
    self.optimizer = optim.SGD(self.model.parameters(),
                               lr=self.ARCH["train"]["lr"],
                               momentum=self.ARCH["train"]["momentum"],
                               weight_decay=self.ARCH["train"]["w_decay"])

    # Use warmup learning rate
    # post decay and step sizes come in epochs and we want it in steps
    steps_per_epoch = self.parser.get_train_size()
    up_steps = int(self.ARCH["train"]["wup_epochs"] * steps_per_epoch)
    final_decay = self.ARCH["train"]["lr_decay"] ** (1/steps_per_epoch)
    self.scheduler = warmupLR(optimizer=self.optimizer,
                              lr=self.ARCH["train"]["lr"],
                              warmup_steps=up_steps,
                              momentum=self.ARCH["train"]["momentum"],
                              decay=final_decay)

    self.start_epoch = 0
    if self.pretrained is not None:
      try:
        w_dict = torch.load(self.pretrained)
        self.model.load_state_dict(w_dict['model'])
        self.optimizer.load_state_dict(w_dict['optim'])
        self.scheduler.load_state_dict(w_dict['scheduler'])
        self.start_epoch = w_dict['epoch']
        self.logger.info("Successfully loaded model weights")
      except Exception as e:
        self.logger.warning()
        self.logger.warning("Couldn't load parameters, using random weights. Error: ", e)
        raise e


  def train(self):

    # accuracy and IoU stuff
    best_train_iou = 0.0
    best_val_iou = 0.0

    self.ignore_class = []
    for i, w in enumerate(self.loss_w):
      if w < 1e-10:
        self.ignore_class.append(i)
        self.logger.info("Ignoring class ", i, " in IoU evaluation")
    self.evaluator = iouEval(self.parser.get_n_classes(),
                             self.device, self.ignore_class)

    # train for n epochs
    for epoch in range(self.start_epoch, self.ARCH["train"]["max_epochs"]):
      # get info for learn rate currently

      # train for 1 epoch
      acc, iou, loss, update_mean = self.train_epoch(train_loader=self.parser.get_train_set(),
                                                     model=self.model,
                                                     optimizer=self.optimizer,
                                                     epoch=epoch,
                                                     evaluator=self.evaluator,
                                                     scheduler=self.scheduler,
                                                     color_fn=self.parser.to_color,
                                                     report=self.ARCH["train"]["report_batch"])
      
      self.writer.add_scalar('training/acc', acc, epoch)
      self.writer.add_scalar('training/mIoU', iou, epoch)
      self.writer.add_scalar('training/loss', loss, epoch)

      # remember best iou and save checkpoint
      if iou > best_train_iou:
        self.logger.info("Best mean iou in training set so far, save model!")
        best_train_iou = iou
        torch.save({ 'epoch': epoch, 
             'optim': self.optimizer.state_dict(), 
             'scheduler': self.scheduler.state_dict(), 
             'model': self.model.state_dict() }, 
             osp.join(self.log, 'epoch-' + str(epoch).zfill(4) + '.path'))


      if epoch % self.ARCH["train"]["report_epoch"] == 0:
        # evaluate on validation set
        self.logger.info("*" * 80)
        acc, iou, loss = self.validate(val_loader=self.parser.get_valid_set(),
                                                 model=self.model,
                                                 evaluator=self.evaluator,
                                                 class_func=self.parser.get_xentropy_class_string)
      
        self.writer.add_scalar('validating/acc', acc, epoch)
        self.writer.add_scalar('validating/mIoU', iou, epoch)
        self.writer.add_scalar('validating/loss', loss, epoch)

        # remember best iou and save checkpoint
        if iou > best_val_iou:
          self.logger.info("Best mean iou in validation so far, save model!")
          self.logger.info("*" * 80)
          best_val_iou = iou

          # save the weights!
          torch.save({ 'epoch': epoch, 
               'optim': self.optimizer.state_dict(), 
               'scheduler': self.scheduler.state_dict(), 
               'model': self.model.state_dict() }, 
               osp.join(self.log, 'best_val-epoch-' + str(epoch).zfill(4) + '.path'))

        self.logger.info("*" * 80)

    self.logger.info('Finished Training')

    return

  def train_epoch(self, train_loader, model, optimizer, epoch, evaluator, scheduler, color_fn, report=10):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    fslosses = AverageMeter()
    slosses = AverageMeter()
    mlosses = AverageMeter()
    elosses = AverageMeter()
    acc = AverageMeter()
    iou = AverageMeter()
    update_ratio_meter = AverageMeter()

    # empty the cache to train now
    if self.gpu:
      torch.cuda.empty_cache()

    # switch to train mode
    model.train()

    end = time.time()

    for i, (in_vol, proj_mask, proj_labels, _, path_seq, path_name, _, _, proj_range, _, _, _, _, _, _, edge) in enumerate(train_loader):
        # measure data loading time
      data_time.update(time.time() - end)
      if not self.multi_gpu and self.gpu:
        in_vol = in_vol.cuda()
        proj_mask = proj_mask.cuda()
      if self.gpu:
        proj_labels = proj_labels.cuda(non_blocking=True).long()

      # compute output
      output, skips = model(in_vol)
      loss = self.criterion(torch.log(output.clamp(min=1e-8)), proj_labels) + self.ls(output, proj_labels.long())

      if self.use_mps:
        fsloss = loss

        sloss = 0
        for j, s in enumerate(skips['seg']):
          proj_labels_small = F.interpolate(proj_labels.unsqueeze(1).float(), size=(skips['seg'][j].size(2), skips['seg'][j].size(3)), mode='nearest').long().squeeze()
          l = self.criterion(torch.log(F.softmax(s, dim=1).clamp(min=1e-8)), proj_labels_small)
          sloss = sloss + l
        sloss *= 0.1

        mloss = 0
        for j, s in enumerate(skips['mot']):
          mot_labels_small = F.interpolate((proj_labels > 250).unsqueeze(1).float(), size=(skips['mot'][j].size(2), skips['mot'][j].size(3)), mode='nearest').long().squeeze()
          l = self.criterion_m(torch.log(F.softmax(s, dim=1).clamp(min=1e-8)), mot_labels_small)
          mloss = mloss + l
        mloss *= 0.1

        edge = edge.cuda()
        eloss = 0
        for j, e in enumerate(skips['edge']):
          edge_small = F.interpolate(edge.unsqueeze(1).float(), size=(skips['edge'][j].size(2), skips['edge'][j].size(3)), mode='nearest')
          l = self.criterion_e(e, edge_small.float())
          # l[l > 1] = 0
          eloss = eloss + l
  
        fslosses.update(fsloss.mean().item(), in_vol.size(0))
        slosses.update(sloss.mean().item(), in_vol.size(0))
        mlosses.update(mloss.mean().item(), in_vol.size(0))
        elosses.update(eloss.mean().item(), in_vol.size(0))

        loss = fsloss + sloss + eloss  + mloss

      # compute gradient and do SGD step
      optimizer.zero_grad()
      if self.n_gpus > 1:
        idx = torch.ones(self.n_gpus).cuda()
        loss.backward(idx)
      else:
        loss.backward()
      optimizer.step()

      # measure accuracy and record loss
      loss = loss.mean()
      with torch.no_grad():
        evaluator.reset()
        argmax = output.argmax(dim=1)
        evaluator.addBatch(argmax, proj_labels)
        accuracy = evaluator.getacc()
        jaccard, class_jaccard = evaluator.getIoU()
      losses.update(loss.item(), in_vol.size(0))
      acc.update(accuracy.item(), in_vol.size(0))
      iou.update(jaccard.item(), in_vol.size(0))


      # measure elapsed time
      batch_time.update(time.time() - end)
      end = time.time()

      # get gradient updates and weights, so I can print the relationship of
      # their norms
      update_ratios = []
      for g in self.optimizer.param_groups:
        lr = g["lr"]
        for value in g["params"]:
          if value.grad is not None:
            w = np.linalg.norm(value.data.cpu().numpy().reshape((-1)))
            update = np.linalg.norm(-max(lr, 1e-10) *
                                    value.grad.cpu().numpy().reshape((-1)))
            update_ratios.append(update / max(w, 1e-10))
      update_ratios = np.array(update_ratios)
      update_mean = update_ratios.mean()
      update_std = update_ratios.std()
      update_ratio_meter.update(update_mean)  # over the epoch


      if i % self.ARCH["train"]["report_batch"] == 0:
        self.logger.info('Lr: {lr:.3e} | '
              'Update: {umean:.3e} mean,{ustd:.3e} std | '
              'Epoch: [{0}][{1}/{2}] | '
              'Loss {loss.val:.4f} ({loss.avg:.4f}) | '
              'floss {loss.val:.4f} ({fsloss.avg:.4f}) | '
              'sloss {loss.val:.4f} ({sloss.avg:.4f}) | '
              'eloss {loss.val:.4f} ({eloss.avg:.4f}) | '
              'mloss {loss.val:.4f} ({mloss.avg:.4f}) | '
              'acc {acc.val:.3f} ({acc.avg:.3f}) | '
              'IoU {iou.val:.3f} ({iou.avg:.3f})'.format(
                  epoch, i, len(train_loader), batch_time=batch_time,
                  data_time=data_time, loss=losses, fsloss=fslosses, sloss=slosses, eloss=elosses, mloss=mlosses, acc=acc, iou=iou, lr=lr,
                  umean=update_mean, ustd=update_std))

      # step scheduler
      scheduler.step()

    if self.use_mps:
      self.writer.add_scalar('training/fsloss', fslosses.avg, epoch)
      self.writer.add_scalar('training/sloss', slosses.avg, epoch)
      self.writer.add_scalar('training/mloss', mlosses.avg, epoch)
      self.writer.add_scalar('training/eloss', elosses.avg, epoch)

    return acc.avg, iou.avg, losses.avg, update_ratio_meter.avg

  def validate(self, val_loader, model, evaluator, class_func):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    iou = AverageMeter()

    # switch to evaluate mode
    model.eval()
    evaluator.reset()

    # empty the cache to infer in high res
    if self.gpu:
      torch.cuda.empty_cache()

    with torch.no_grad():
      end = time.time()
      for i, (in_vol, proj_mask, proj_labels, _, path_seq, path_name, _, _, proj_range, _, _, _, _, _, _, edge) in enumerate(val_loader):
        if not self.multi_gpu and self.gpu:
          in_vol = in_vol.cuda()
          proj_mask = proj_mask.cuda()
        if self.gpu:
          proj_labels = proj_labels.cuda(non_blocking=True).long()

        # compute output
        output, skips = model(in_vol)
        loss = self.criterion(torch.log(output.clamp(min=1e-8)), proj_labels)

        # measure accuracy and record loss
        argmax = output.argmax(dim=1)
        evaluator.addBatch(argmax, proj_labels)
        losses.update(loss.mean().item(), in_vol.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

      accuracy = evaluator.getacc()
      jaccard, class_jaccard = evaluator.getIoU()
      acc.update(accuracy.item(), in_vol.size(0))
      iou.update(jaccard.item(), in_vol.size(0))

      self.logger.info('Validation set:\n'
            'Time avg per batch {batch_time.avg:.3f}\n'
            'Loss avg {loss.avg:.4f}\n'
            'Acc avg {acc.avg:.3f}\n'
            'IoU avg {iou.avg:.3f}'.format(batch_time=batch_time,
                                           loss=losses,
                                           acc=acc, iou=iou))
      # print also classwise
      for i, jacc in enumerate(class_jaccard):
        self.logger.info('IoU class {i:} [{class_str:}] = {jacc:.3f}'.format(
            i=i, class_str=class_func(i), jacc=jacc))

    return acc.avg, iou.avg, losses.avg
