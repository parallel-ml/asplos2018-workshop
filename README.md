# Real-Time Image Recognition Using Collaborative IoT Devices 
__ACM ReQuEST workshop co-located with ASPLOS 2018__


This repository contains demo files for demonstration of Musical Chair[1] applied on two state-of-art 
deep learning neural networks, AlexNet[2] and VGG16[3].


## Installation
Please make sure that you have <b>Python 2.7</b> running on your device. We have
two versions of model inference. One is using GPU and running model inference on
single machine. Another is using CPU and using RPC to off-shore the computation
to other devices. We will have different installation guide for those two versions
model inference. 

### Single device (GPU and CPU).
_(This is NVidia Jetson TX2 version in our paper)_

Dependencies:
* tensorflow-gpu >= 1.5.0
* Keras >= 2.1.3

```angular2html
pip install keras
```
[Please refer to official installation guideline from Keras for more information](https://github.com/keras-team/keras)

### Multiple devices (CPU and RPC).
_(This is Raspberry PI 3 versions in our paper)_

Dependencies:
* tensorflow >= 1.5.0
* Keras >= 2.1.3
* avro >= 1.8.2

We have provided dependency file here. You can execute this file to install packages.
```angular2html
pip install -r requirements.txt
```

## Quick Start

### Single device (GPU and CPU)
_(This is NVidia Jetson TX2 version in our paper)_

#### GPU Version
Execute predict file to run model inference. 
```
python predict.py
```
#### CPU Version
```
CUDA_VISIBLE_DEVICES= python predict.py
```

### Multiple devices (CPU and RPC)
_(This is Raspberry PI 3 versions in our paper)_

We make a checklist for you before running our program.
- [ ] Have all correct packages installed on Raspberry Pi. 
- [ ] The Raspberry PI has port 12345, 9999 open. 
- [ ] Put correct IP address in IP table file `mutiple-devices/alexnet/resource/ip`. 
The IP table file is in `json` format. 

#### AlexNet

For AlexNet, we have same model partition, so we will use the same node file for 
different system setup. The IP table is default to 4 devices setup. You need to 
add 1 more IP address to `block1` if you want to test 6 devices setup.

![alexnet](https://github.com/parallel-ml/asplos2018-workshop/blob/master/figs/alexnet-nodes.png)

* On all of your device except the initial sender, run the node.
```angular2html
python node.py
```

* Start the data sender. You should be able to see console log.
```angular2html
python initial.py
```

* If you modify our code, you can use flag to debug.
```angular2html
python node.py -d
```

#### VGG16

For VGG16, we have different model separation for different system setup, so we put
two directories under `mutiple-devices/vgg16`. For `8devices`, you should have 3 devices for
<b>block234</b> and 2 devices for <b>fc1</b>, which means you need 2 IP addresses for those
2 blocks in IP table. For `11devices`, you should have 7 devices for <b>block12345</b>,
so put 7 IP addresses at IP table. 

![vgg16](https://github.com/parallel-ml/asplos2018-workshop/blob/master/figs/vgg-8nodes.png)

* On all of your device except the initial sender, run the node.
```angular2html
python node.py
```

* Start the data sender. You should be able to see console log.
```angular2html
python initial.py
```


### Refereces
[1]: R. Hadidi, J. Cao, M. Woodward, M. Ryoo, and H. Kim, "Musical Chair: Efficient Real-Time Recognition Using Collaborative IoT Devices," ArXiv e-prints:1802.02138.

[2]: A. Krizhevsky, I. Sutskever, and G. E. Hinton, "Imagenet Classification With Deep Convolutional Neural Networks}," in Advances in Neural InformationProcessing Systems (NIPS), pp. 1097--1105, 2012.

[3]: K. Simonyan and A. Zisserman, "Very Deep Convolutional Networks for Large-Scale Image Recognition," in International Conference onLearning Representations (ICLR), 2015.
