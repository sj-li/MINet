# MINet

## Dependence:
1. Accoding to LiDAR-Bonnetal (https://github.com/PRBonn/lidar-bonnetal/tree/master/train)
2. flops-counter.pytorch (https://github.com/sovrasov/flops-counter.pytorch)

## Infer:

'''
sh infer.sh
'''

## Attention

1. Only infer validation set, refer 'lib/user.py' line 70-80.
2. pay attention to kNN setting. (In 'config/arch/*.yaml')

