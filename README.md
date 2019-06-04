**F-E3D: FPGA-based Acceleration of An Efficient 3D Convolutional Neural Networkfor Human Action Recognition**

**If you have any questions, pls leave an issue or email me at h.fan17@imperial.ac.uk**

# 1. E3DNet
This is the evaluation code and model for paper "F-E3D: FPGA-based Acceleration of An Efficient 3D Convolutional Neural Networkfor Human Action Recognition". The whole implementation is based on [MXNet](https://mxnet.apache.org/).

*Please cite our paper if it is helpful for your research*

# 2. Content
<!-- TOC -->

- [1. E3DNet](#1-e3dnet)
- [2. Content](#2-content)
- [3. Preparation](#3-preparation)
    - [3.1. Installation of MXNet](#31-installation-of-mxnet)
    - [3.2. Download UCF101 Dataset](#32-download-ucf101-dataset)
- [Evaluation](#evaluation)
    - [Finetuning](#finetuning)
    - [Validation](#validation)

<!-- /TOC -->

# 3. Preparation
## 3.1. Installation of MXNet

1. Set up your environment with python and pip. You can use your native environment or [Anaconda](https://www.anaconda.com/).
1. Install [CUDA 9.2](https://developer.nvidia.com/cuda-92-download-archive) and [cudnn](https://developer.nvidia.com/cudnn).
1. Install MXNet using `pip install mxnet-cu92`.

## 3.2. Download UCF101 Dataset
1. Download the UCF101 dataset from this [link](https://www.crcv.ucf.edu/data/UCF101.php).
1. Decompress the downloaded dataset, remember the absolute path to this dataset.

# Evaluation
## Finetuning
1. Since Training on Kinetics dataset takes takes nearly one month even on six-GPU cluster, the checkpoint of E3DNet model pretrained on Kinetcs is available on `ckpt/E3DNet_ckpt_kinetics`.
1. In `script/finetuning_E3DNet_ucf101.sh`, changing the path of UCF101 dataset to your local location.
1. Run the script:
```
sh script/finetuning_E3DNet_ucf101.sh
```
Optional: If you like, you can use [screen](https://www.rackaid.com/blog/linux-screen-tutorial-and-how-to/)  to separate training from the terminal shell.

## Validation
1. In `script/validation_ucf101.sh`, changing the path of UCF101 dataset to your local location, and pointing the model path to the finetuned model.
1. The accuracy of other models is also shown in table below. 

**Please note, the accuracy we are testing is Clip@1 accuracy (prediction using only 1 clip). Many papers report Video@1 accuracy (prediction using only 10 or 20 clips), which is impossible for real-life applications**


|   Model  |  ResNeX-101 [Link1](https://github.com/kenshohara/3D-ResNets-PyTorch/blob/master/README.md) [Link2]( https://github.com/kenshohara/3D-ResNets-PyTorch/issues/121) | P3D [Link](https://arxiv.org/pdf/1711.10305.pdf) | C3D [Link](https://github.com/hx173149/C3D-tensorflow) | E3DNet |
|---|---|---|---|---|
|Clip@1 Accuracy|  87.7% |  84.2% | 79.87% | 85.17% |


*For some models which do not report Clip@1 accuracy in their paper, we use the Clip@1 accuracy reported in their github repository. Link followed after the accuracy is their github implementation*
