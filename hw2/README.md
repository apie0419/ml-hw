# CCUCS ML Course HW2

This repo is an sample code for ML HW2.

It's for cat&dog classification.

You can modify some pieces of code on your demand.

## Data Preparation

Download Data From: [link](https://www.kaggle.com/pocahontas1010/dogs-vs-cats-for-pytorch/download)

## Environment Installation

```
conda create -f environment.yml -n <env_name>
```

## Config

You can modify parameters in config/config.yml.

such as 
  - batch_size
  - epoch
  - learning rate

and so on

Desriptions for every parameters were writen in config/defaults.py.

## Model

The model structure shows in following picture.

![](https://github.com/apie0419/ml-hw/tree/master/hw2/figures/cnn.png)