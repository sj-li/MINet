import os
import numpy as np
import torch
from torch.utils.data import Dataset
from ..utils.laserscan import LaserScan, SemLaserScan

from .SemanticKitti import SemanticKitti

class Parser():
  # standard conv, BN, relu
  def __init__(self,
               root,              # directory for data
               data_cfg,
               arch_cfg,
               gt=True,           # get gt?
               shuffle_train=True):  # shuffle training set?
    super(Parser, self).__init__()

    # if I am training, get the dataset
    self.DATASET_TYPE = eval(data_cfg["name"])
    self.root = root
    self.train_sequences = data_cfg["split"]["train"]
    self.valid_sequences = data_cfg["split"]["valid"]
    self.test_sequences = data_cfg["split"]["test"]
    self.labels = data_cfg["labels"]
    self.color_map = data_cfg["color_map"]
    self.learning_map = data_cfg["learning_map"]
    self.learning_map_inv = data_cfg["learning_map_inv"]
    self.sensor = arch_cfg["dataset"]["sensor"]
    self.max_points = arch_cfg["dataset"]["max_points"]
    self.batch_size = arch_cfg["train"]["batch_size"]
    self.workers = arch_cfg["train"]["workers"]
    self.gt = gt
    self.shuffle_train = shuffle_train

    # number of classes that matters is the one for xentropy
    self.nclasses = len(self.learning_map_inv)

    # Data loading code
    self.train_dataset = self.DATASET_TYPE(root=self.root,
                                       sequences=self.train_sequences,
                                       labels=self.labels,
                                       color_map=self.color_map,
                                       learning_map=self.learning_map,
                                       learning_map_inv=self.learning_map_inv,
                                       sensor=self.sensor,
                                       max_points=self.max_points,
                                       gt=self.gt)

    self.trainloader = torch.utils.data.DataLoader(self.train_dataset,
                                                   batch_size=self.batch_size,
                                                   shuffle=self.shuffle_train,
                                                   num_workers=self.workers,
                                                   pin_memory=True,
                                                   drop_last=True)
    assert len(self.trainloader) > 0
    self.trainiter = iter(self.trainloader)

    self.valid_dataset = self.DATASET_TYPE(root=self.root,
                                       sequences=self.valid_sequences,
                                       labels=self.labels,
                                       color_map=self.color_map,
                                       learning_map=self.learning_map,
                                       learning_map_inv=self.learning_map_inv,
                                       sensor=self.sensor,
                                       max_points=self.max_points,
                                       gt=self.gt, skip=0)

    self.validloader = torch.utils.data.DataLoader(self.valid_dataset,
                                                   batch_size=self.batch_size,
                                                   shuffle=False,
                                                   num_workers=self.workers,
                                                   pin_memory=True,
                                                   drop_last=True)
    assert len(self.validloader) > 0
    self.validiter = iter(self.validloader)

    if self.test_sequences:
      self.test_dataset = self.DATASET_TYPE(root=self.root,
                                        sequences=self.test_sequences,
                                        labels=self.labels,
                                        color_map=self.color_map,
                                        learning_map=self.learning_map,
                                        learning_map_inv=self.learning_map_inv,
                                        sensor=self.sensor,
                                        max_points=self.max_points,
                                        gt=False)

      self.testloader = torch.utils.data.DataLoader(self.test_dataset,
                                                    batch_size=self.batch_size,
                                                    shuffle=False,
                                                    num_workers=self.workers,
                                                    pin_memory=True,
                                                    drop_last=True)
      assert len(self.testloader) > 0
      self.testiter = iter(self.testloader)

  def get_train_batch(self):
    scans = self.trainiter.next()
    return scans

  def get_train_set(self):
    return self.trainloader

  def get_valid_batch(self):
    scans = self.validiter.next()
    return scans

  def get_valid_set(self):
    return self.validloader

  def get_test_batch(self):
    scans = self.testiter.next()
    return scans

  def get_test_set(self):
    return self.testloader

  def get_train_size(self):
    return len(self.trainloader)

  def get_valid_size(self):
    return len(self.validloader)

  def get_test_size(self):
    return len(self.testloader)

  def get_n_classes(self):
    return self.nclasses

  def get_original_class_string(self, idx):
    return self.labels[idx]

  def get_xentropy_class_string(self, idx):
    return self.labels[self.learning_map_inv[idx]]

  def to_original(self, label):
    # put label in original values
    return self.DATASET_TYPE.map(label, self.learning_map_inv)

  def to_xentropy(self, label):
    # put label in xentropy values
    return self.DATASET_TYPE.map(label, self.learning_map)

  def to_color(self, label):
    # put label in original values
    label = self.DATASET_TYPE.map(label, self.learning_map_inv)
    # put label in color
    return self.DATASET_TYPE.map(label, self.color_map)
