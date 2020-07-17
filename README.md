# MINet

This part of the framework deals with the training of segmentation networks for point cloud data using range images.

## Dependence:
First you need to install the nvidia driver and CUDA.

- CUDA Installation guide: [link](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)

- System dependencies:

  ```sh
  $ sudo apt-get update
  $ sudo apt-get install -yqq  build-essential ninja-build \
    python3-dev python3-pip apt-utils curl git cmake unzip autoconf autogen \
    libtool mlocate zlib1g-dev python3-numpy python3-wheel wget \
    software-properties-common openjdk-8-jdk libpng-dev  \
    libxft-dev ffmpeg python3-pyqt5.qtopengl
  $ sudo updatedb
  ```

- Python dependencies

  ```sh
  $ sudo pip3 install -r requirements.txt
  $ pip3 install --upgrade git+https://github.com/sovrasov/flops-counter.pytorch.git
  ```

## Configuration files

  Architecture configuration files are located at [config/arch](config/arch/)
  Dataset configuration files are located at [config/labels](config/labels/)

## Data Preparation
1. Download SemanticKITTI: [link](http://semantic-kitti.org/)
1. put 'sequences' folder under 'data/'
2. 'python infer.py --dataset data  --arch_cfg config/arch/config_file  --data_cfg config/labels/semantic-kitti.yaml --checkpoint checkpoints/checkpoint_file --log predictions'

## Apps

`ALL SCRIPTS CAN BE INVOKED WITH -h TO GET EXTRA HELP ON HOW TO RUN THEM`

### Visualization

To visualize the data (in this example sequence 00):

```sh
$ python lib/utils/visualize.py -d /path/to/dataset/ -s 00
```

To visualize the predictions (in this example sequence 00):

```sh
$ python lib/utils/visualize.py -d /path/to/dataset/ -p /path/to/predictions/ -s 00
```

### Training

To train a network (from scratch):

```sh
$ python train.py -d /path/to/dataset  --ac config/arch/CHOICE.yaml -l /path/to/log
```

To train a network (from pretrained model):

```
$ python train.py -d /path/to/dataset  --ac config/arch/CHOICE.yaml -l /path/to/log -p /path/to/pretrained
```

This will generate a tensorboard log, which can be visualized by running:

```sh
$ cd /path/to/log
$ tensorboard --logdir=. --port 5555
```

And acccessing [http://localhost:5555](http://localhost:5555) in your browser.

### Inference

To infer the predictions for the entire dataset:

```sh
$ ./infer.py -d /path/to/dataset/ -l /path/for/predictions -m /path/to/model
python infer.py -d /path/to/dataset/ --ac  config/arch/CHOICE.yaml --checkpoint CHECKPOINT --log predictions
````

### Evaluation

To evaluate the overall IoU of the point clouds (of a specific split, which in semantic kitti can only be train and valid, since test is only run in our evaluation server):

```sh
$ ./evaluate_iou.py -d /path/to/dataset -p /path/to/predictions/ --split valid
```

To evaluate the border IoU of the point clouds (introduced in RangeNet++ paper):

```sh
$ ./evaluate_biou.py -d /path/to/dataset -p /path/to/predictions/ --split valid --border 1 --conn 4
```
