#!/bin/bash

#block(name=mot, threads=12, memory=10000, gpus=1, hours=200)
python train.py -d data/ -ac config/arch/MINet-2048-knn.yaml -dc config/labels/semantic-kitti-all.yaml -l logs
