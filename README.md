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

We have provided dependency file here. You can execute this file to install packages.
```angular2html
pip install -r requirements.txt
```

## Quick Start

#### Single device (GPU)
Execute predict file to run model inference. 
```
python predict.py
```

#### Multiple devices (CPU and RPC)

We make a checklist for you before running our program.
- [ ] Have all correct packages installed on Raspberry Pi. 
- [ ] The Raspberry PI has port 12345, 9999 open. 
- [ ] Put correct IP address in IP table file `CPU/alexnet/resource/ip`. 
The IP table file is in `json` format. 

##### Start alex net system

For Alex Net, we have same model partition, so we will use the same node file for 
different system setup. The IP table is default to 4 devices setup. You need to 
add 1 more IP address to `block1` if you want to test 6 devices setup.

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

##### Start vgg16 net system

For VGG16, we have different model separation for different system setup, so we put
two directories under `CPU/vgg16`. For `8devices`, you should have 2 devices for
<b>block234</b> and <b>block6</b>, which means you need 2 IP addresses for those
2 blocks in IP table. For `11devices`, you should have 7 devices for <b>block12345</b>,
so put 7 IP addresses at IP table. 

* On all of your device except the initial sender, run the node.
```angular2html
python node.py
```

* Start the data sender. You should be able to see console log.
```angular2html
python initial.py
```