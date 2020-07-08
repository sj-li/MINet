#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import argparse
import subprocess
import datetime
import yaml
from shutil import copyfile
import os
import shutil

from lib.user import *


if __name__ == '__main__':
  parser = argparse.ArgumentParser("./infer.py")
  parser.add_argument(
      '--dataset',
      type=str,
      required=True,
      help='Dataset to train with. No Default',
  )

  parser.add_argument(
      '--arch_cfg',
      type=str,
      required=True,
      default=None,
      help='arch config path'
  )

  parser.add_argument(
      '--data_cfg',
      type=str,
      required=True,
      default=None,
      help='data config path'
  )

  parser.add_argument(
      '--checkpoint',
      type=str,
      required=True,
      default=None,
      help='Directory to checkpoint.'
  )

  parser.add_argument(
      '--log',
      type=str,
      required=False,
      default='predictions',
      help='Directory to save predictions.'
  )



  FLAGS, unparsed = parser.parse_known_args()

  # print summary of what we will do
  print("----------")
  print("INTERFACE:")
  print("dataset", FLAGS.dataset)
  print("----------\n")

  # open arch config file
  try:
    print("Opening arch config file from %s" % FLAGS.arch_cfg)
    ARCH = yaml.safe_load(open(FLAGS.arch_cfg, 'r'))
  except Exception as e:
    print(e)
    print("Error opening arch yaml file.")
    quit()

  # open data config file
  try:
    print("Opening data config file from %s" % FLAGS.data_cfg)
    DATA = yaml.safe_load(open('config/labels/semantic-kitti.yaml', 'r'))
  except Exception as e:
    print(e)
    print("Error opening data yaml file.")
    quit()

  # create log folder
  try:
    if os.path.isdir(FLAGS.log):
      shutil.rmtree(FLAGS.log)
    os.makedirs(FLAGS.log)
    os.makedirs(os.path.join(FLAGS.log, "sequences"))
    for seq in DATA["split"]["train"]:
      seq = '{0:02d}'.format(int(seq))
      print("train", seq)
      os.makedirs(os.path.join(FLAGS.log, "sequences", seq))
      os.makedirs(os.path.join(FLAGS.log, "sequences", seq, "predictions"))
    for seq in DATA["split"]["valid"]:
      seq = '{0:02d}'.format(int(seq))
      print("valid", seq)
      os.makedirs(os.path.join(FLAGS.log, "sequences", seq))
      os.makedirs(os.path.join(FLAGS.log, "sequences", seq, "predictions"))
    for seq in DATA["split"]["test"]:
      seq = '{0:02d}'.format(int(seq))
      print("test", seq)
      os.makedirs(os.path.join(FLAGS.log, "sequences", seq))
      os.makedirs(os.path.join(FLAGS.log, "sequences", seq, "predictions"))
  except Exception as e:
    print(e)
    print("Error creating log directory. Check permissions!")
    raise

  except Exception as e:
    print(e)
    print("Error creating log directory. Check permissions!")
    quit()

  # create user and infer dataset
  user = User(ARCH, DATA, FLAGS.dataset, FLAGS.log, FLAGS.checkpoint)
  user.infer()
