# README


## Configuring the code

Configuration is stored in two locations:

1. config.py

```
IMAGE_PATH = 'path to your images'
ANNOTATION_PATH = 'path to the folder containing the image annotation xmls'
PIPELINE_CONFIG_PATH = './faster_rcnn_resnet101.config'
TRAIN_PATH = 'path to where your trained model should be stored'
OUTPUT_INFERENCE_GRAPH_PATH = 'path to the output inference graph folder'
```

This file is not stored in the repository, but a default can be found at config_default.py. You can copy this file to config.py

2. ./faster_rcnn_resnet101.config

This is a rather complex file. The relevant parts are only two points:

num_steps in train_config, which determines how many examples the network will iterate over until it is done.

label_map_path: "/mnt/nfs/projects/trash_recognition/data/examples/annotations/label_map.pbtxt" in train_input_config, which is used to determine the labels of the neural network.


## Running the code

Retrain the neural net from scratch with this command:
```
./run_retrain.sh
```

Start the api with this command
```
./run_api_server.sh

```
