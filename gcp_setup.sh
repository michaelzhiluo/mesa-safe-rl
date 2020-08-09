#!/bin/bash

sudo apt update
sudo apt install -y python3-pip ffmpeg zip unzip libsm6 libxext6 libgl1-mesa-dev libosmesa6-dev libgl1-mesa-glx patchelf

pip3 install numpy scipy gym dotmap matplotlib tqdm opencv-python tensorboardX moviepy plotly
pip3 install tensorflow-gpu==1.14.0
pip3 install torch==1.4.0
pip3 install torchvision==0.5.0
pip3 install mujoco_py==1.50.1.68
