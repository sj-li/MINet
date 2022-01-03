# Multi-scale Interaction for Real-time LiDAR Data Segmentation on an Embedded Platform (RA-L)

## Dependence:
1. Accoding to LiDAR-Bonnetal (https://github.com/PRBonn/lidar-bonnetal/tree/master/train)
2. flops-counter.pytorch (https://github.com/sovrasov/flops-counter.pytorch)
3. Edge files: [Edges](https://1drv.ms/u/s!AqmpjbHa-zD-ibFOZr8GlhLdrcpy2g?e=X43LXt)

## Infer:
1. put 'sequences' folder under 'data/'
2. 'python infer.py --dataset data  --arch_cfg config/arch/config_file  --data_cfg config/labels/semantic-kitti.yaml --checkpoint checkpoints/checkpoint_file --log predictions'

## Attention

1. Only infer validation set, refer 'lib/user.py' line 70-80.
2. pay attention to kNN setting. (In 'config/arch/*.yaml')

## Citation
Please cite the following paper if you use this repository in your reseach.
```
@ARTICLE{9633188,
  author={Li, Shijie and Chen, Xieyuanli and Liu, Yun and Dai, Dengxin and Stachniss, Cyrill and Gall, Juergen},
  journal={IEEE Robotics and Automation Letters}, 
  title={Multi-Scale Interaction for Real-Time LiDAR Data Segmentation on an Embedded Platform}, 
  year={2022},
  volume={7},
  number={2},
  pages={738-745},
  doi={10.1109/LRA.2021.3132059}}
