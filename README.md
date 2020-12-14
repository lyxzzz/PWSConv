# PWSConv

## Introduction

Code repository for paper "Delving into variance transmission and normalization: Shift of average gradient makes the network collapse."
The implementation of this project is a lightweight classification framework based on [mmcv](https://github.com/open-mmlab/mmcv), and some details are similar to [mmclassification](https://github.com/open-mmlab/mmclassification).

### Points for attention

- This repository only supports single GPU training. 
- The implementation of PWS is in [modules/utils/pws_layer.py](modules/utils/pws_layer.py).
- When we set `use_pws=True` in configs, the **norm_cfg** will **no longer** add norm layer for bottleneck. For classification tasks, PWS only uses two norm layer in the whole model. The corresponding codes are [resnet.py](modules/backbone/resnet.py):325-329 and [relu_neck.py](modules/neck/relu_neck.py).
- The implementation of our ResNet is a little different from the general ResNet. The conventional implementation for bottleneck is: 
``` python
    def forward(self, x):
        def _inner_forward(x):
            identity = x

            bottleneck_forward(x)
            
            out += identity
            return out

        x = self.relu(x)
        out = _inner_forward(x)
        return out
```
Our implementation for bottleneck is:
``` python
    def forward(self, x):
        def _inner_forward(x):
            identity = x
            x = self.relu(x)
            
            bottleneck_forward(x)

            out += identity
            return out

        out = _inner_forward(x)
        return out
```
Our code will guarantee the identity to maintain an approximate zero mean, which may maintain a balance variance as discussed in the appendix.
- The results of our ImageNet experiments are based on this repository. The results of our CIFAR10 and VOC experiments are based on an old TensorFlow repository, which may produce inferior results. However, all comparisons in the paper are under the same code repository. As the old TensorFlow repository is abandoned, we will continuously provide corresponding experimental results in this repository. For old tensorflow code and detection code please refer to https://github.com/lyxzzz/PWSConv_backup

- This repository may produce some low eval results (We have encountered this situation in one of our GPU server) during the training. Please check the eval result by running `python linear_val.py --config ${your_config_file} --ckpt ${your_ckpt_file}`.

- Some class-wise results are presented in [results](results). Every row indicates {class id}, {top-1 acc}. The first row is the total top-1 acc.

| Methods | Top-1(%) | File |
| :----- | :----- | :----: |
| PWS | 76.0 | [imagenet.txt](results/imagenet.txt) |


## Installation

### Requirements

- Python 3.6+
- PyTorch 1.6+
- [mmcv](https://github.com/open-mmlab/mmcv) 0.6.0+

To config environment, one can run:
```
conda create -n torch1.6 python=3.6.7
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch
pip install mmcv==0.6.0
```
Notice that you should install a compatible version of PyTorch with your Cuda version (here we use **cudatoolkit=10.1**). Please refer to [pytorch](https://pytorch.org/get-started/locally/) to find a detailed installation for PyTorch.

## Getting Started
Before running a scipt, you would better run 
```
ln -s ${DATASET_ROOT} dataset
```
to configure your data path. If your folder structure is different, you may need to change the corresponding paths in config files.
```
PWSConv
├── configs
├── imagenet_label
├── modules
├── dataset
│   ├── imagenet
│   │   ├── train
│   │   ├── val
│   ├── cifar
│   │   ├── cifar-10-batches-py
```
We provide a [scipt](train.sh) and some [configs](configs) to train models. In [train.sh](train.sh), one can run:
``` bash
runModel ${configfile} ${logname}
````
For example, to run PWS model, one can replace the command in [train.sh](train.sh):21 with
``` bash
runModel resnet50_imagenet PWSmodel
```
The above command will use **configs/resnet50_imagenet.py** as the configuration and output training logs and checkpoints in **result/PWSmodel**.

## Experimental results

### ImageNet
| Backbone | Methods | Training speed(task/sec) | Inference speed(task/sec) | Params(M) | Top-1(%) |
| :----- | :----- | :----: | :----: | :----: | :----: |
| ResNet-50 | BN | 4.2 | 14.9 | 23.508 | 76.4 |
| ResNet-50 | PWS | 4.6 | 14.9 | 23.508 | 76.0 |
| ResNet-50 | GN | 3.9 | 13.4 | 23.508 | 75.7 |

Speed represents the number of iterations (or steps) that can be trained (inference) on an RTX2080ti per second.

### Cifar10
| Backbone | Methods | Top-1(%) |
| :----- | :----- | :----: |
| ResNet-34 | GN | 94.2 |
| ResNet-34 | BN | 95.3 |
| ResNet-34 | PWS | 95.1 |
