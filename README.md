# ASPLOS2018-workshop
This repository demos parallelization of fully connected layer on state-of-art 
deep learning neural network.

## Installation
You need to make sure you have <b>Python 2.7</b> running on your device. We have
two versions of model inference. One is using GPU and running model inference on
single machine. Another is using CPU and using RPC to off-shore the computation
to other devices. We will have different installation guide for those two versions
model inference. 

#### Single device (GPU).

Dependencies:
* tensorflow-gpu >= 1.5.0
* Keras >= 2.1.3

[We suggest to follow official installation guideline from Keras.](https://github.com/keras-team/keras)

#### Multiple devices (CPU and RPC).

Dependencies:
* tensorflow >= 1.5.0
* Keras >= 2.1.3
* avro >= 1.8.2

## Quick Start

#### Single device (GPU)
Execute predict file to run model inference. 
```
python predict.py
```

#### Multiple devices (CPU and RPC)