#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
import imp
import yaml
import time
from PIL import Image
import __init__ as booger
import collections
import copy
import cv2
import os
import numpy as np

from .postproc.KNN import KNN
from .dataset.Parser import Parser
from .models import *
from ptflops import get_model_complexity_info

class User():
  def __init__(self, ARCH, DATA, datadir, logdir, checkpoint, split='valid'):
    # parameters
    self.ARCH = ARCH
    self.DATA = DATA
    self.datadir = datadir
    self.logdir = logdir
    self.checkpoint = checkpoint
    self.split = split

    # get the data
    ARCH['train']['batch_size'] = 1
    self.parser = Parser(root=self.datadir,
                     data_cfg = DATA,
                     arch_cfg = ARCH,
                     gt=True,
                     shuffle_train=False)
    self.input_size = (ARCH['model']['in_channels'], ARCH['dataset']['sensor']['img_prop']['height'], ARCH['dataset']['sensor']['img_prop']['width'])

    # concatenate the encoder and the head
    self.model = get_model(ARCH['model']['name'])(ARCH['model']['in_channels'], self.parser.get_n_classes(), dropout=0, is_train=False)

    # use knn post processing?
    self.post = None
    if self.ARCH["post"]["KNN"]["use"]:
      self.post = KNN(self.ARCH["post"]["KNN"]["params"],
                      self.parser.get_n_classes())

    # GPU?
    self.gpu = False
    self.model_single = self.model
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Infering in device: ", self.device)
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
      cudnn.benchmark = True
      cudnn.fastest = True
      self.gpu = True
      self.model.cuda()

    pretrained_dict = torch.load(self.checkpoint)['model']
    model_dict = self.model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict) 
    self.model.load_state_dict(pretrained_dict)
 

    print("Finish inilization ...")

  def infer(self):
     ## do train set

    if 'train' == self.split:
        self.infer_subset(loader=self.parser.get_train_set(),
                      to_orig_fn=self.parser.to_original)

     # do valid set
    if 'valid' == self.split:
        self.infer_subset(loader=self.parser.get_valid_set(),
                      to_orig_fn=self.parser.to_original)

    # do test set
    if 'test' == self.split:
        self.infer_subset(loader=self.parser.get_test_set(),
                      to_orig_fn=self.parser.to_original)


    print('Finished Infering ...')

    return

  def infer_subset(self, loader, to_orig_fn):
    # switch to evaluate mode
    self.model.eval()

    # empty the cache to infer in high res
    if self.gpu:
      torch.cuda.empty_cache()

    total_time = 0
    print("Start infering ...")

    with torch.no_grad():

      for i, (proj_in, proj_mask, _, _, path_seq, path_name, p_x, p_y, proj_range, unproj_range, _, _, _, _, npoints, _) in enumerate(loader):
        p_x = p_x[0, :npoints]
        p_y = p_y[0, :npoints]
        proj_range = proj_range[0, :npoints]
        unproj_range = unproj_range[0, :npoints]
        path_seq = path_seq[0]
        path_name = path_name[0]

        if self.gpu:
          proj_in = proj_in.cuda()
          proj_mask = proj_mask.cuda()
          p_x = p_x.cuda()
          p_y = p_y.cuda()
          if self.post:
            proj_range = proj_range.cuda()
            unproj_range = unproj_range.cuda()

        #proj_in = proj_in.unsqueeze(0).permute(2, 1, 0, 3, 4)
        end = time.time()

        # compute output
        proj_output, _ = self.model(proj_in)
        proj_argmax = proj_output[0].argmax(dim=0)

        if self.post:
          # knn postproc
          unproj_argmax = self.post(proj_range,
                                    unproj_range,
                                    proj_argmax,
                                    p_x,
                                    p_y)
        else:
          # put in original pointcloud using indexes
          unproj_argmax = proj_argmax[p_y, p_x]

        # measure elapsed time
        if torch.cuda.is_available():
          torch.cuda.synchronize()

        t = time.time() - end
        print("Infered seq", path_seq, "scan", path_name,
              "in", t, "sec")
        total_time += t
        end = time.time()

        # save scan
        # get the first scan in batch and project scan
        pred_np = unproj_argmax.cpu().numpy()
        pred_np = pred_np.reshape((-1)).astype(np.int32)

        # map to original label
        pred_np = to_orig_fn(pred_np)

        # save scan
        path = os.path.join(self.logdir, "sequences",
                            path_seq, "predictions", path_name)
        pred_np.tofile(path)

      with torch.cuda.device(0):
        macs, params = get_model_complexity_info(self.model.cuda(), self.input_size, as_strings=True,
                                               print_per_layer_stat=True, verbose=True)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))


      print('Avg run time: ', total_time/len(loader))
