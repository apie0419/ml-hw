# CCUCS ML Course HW2

This repo is an sample code for ML HW2.

It's for cat&dog classification.

You can modify some pieces of code on your demand.

## Data Preparation

Download Data From: [Link](https://www.kaggle.com/pocahontas1010/dogs-vs-cats-for-pytorch/download)

Train Data Size: 25000 pictures\
Test Data Size: 25000 pictures

(12500 pictures for both cats and dogs)

## Data Processing

Dataloader was defined in datasets/dataloader.py. It helps you load data in Pytorch datatype.
In this case, dataset were splitted in 80%/20% for train data and validation data seperately.

We wrote pieces of code to solve this problem.

```python
trainset = datasets.ImageFolder(train_path, transform=transforms)
    
num_train = len(trainset)
indices = list(range(num_train))
split = int(np.floor(valid_size * num_train))

np.random.shuffle(indices)

train_idx, valid_idx = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)
```

If your validation and train data are splitted already, just do

```python
trainset = datasets.ImageFolder(train_path, transform=transforms)
validset = datasets.ImageFolder(valid_path, transform=transforms)
```

Data preprocessing like Normalization and Resize were written in datasets/transforms.py. Wrote your own code for data preprocessing on your demand.

## Environment Installation

```
conda env create -f environment.yml -n <env_name>
```

**<env_name> must be a new env name**

### Activate Environment

```
conda activate <env_name>
```

## Config

You can modify parameters in config/config.yml.

such as 
  - batch_size
  - epoch
  - learning rate

and so on

Desriptions for every parameters were written in config/defaults.py.

## Model

The model structure shows in following picture.

You can edit model/cnn.py to modify the model structure.

![](https://github.com/apie0419/ml-hw/blob/master/hw2/figures/cnn.png)