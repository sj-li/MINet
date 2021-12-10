# Multi-scale Interaction for Real-time LiDAR Data Segmentation on an Embedded Platform (RA-L)

## Dependence:
1. Accoding to LiDAR-Bonnetal (https://github.com/PRBonn/lidar-bonnetal/tree/master/train)
2. flops-counter.pytorch (https://github.com/sovrasov/flops-counter.pytorch)

## Infer:
1. put 'sequences' folder under 'data/'
2. 'python infer.py --dataset data  --arch_cfg config/arch/config_file  --data_cfg config/labels/semantic-kitti.yaml --checkpoint checkpoints/checkpoint_file --log predictions'

## Attention

1. Only infer validation set, refer 'lib/user.py' line 70-80.
2. pay attention to kNN setting. (In 'config/arch/*.yaml')

