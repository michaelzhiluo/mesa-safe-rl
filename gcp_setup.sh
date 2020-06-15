#!/bin/bash

sudo apt update
sudo apt install -y python3-pip ffmpeg zip unzip libsm6 libxext6 libgl1-mesa-dev libosmesa6-dev libgl1-mesa-glx patchelf

pip3 install numpy scipy gym dotmap matplotlib tqdm opencv-python tensorboardX moviepy
pip3 install tensorflow-gpu==1.14.0
pip3 install torch==1.4.0
pip3 install torchvision==0.5.0

curl https://www.roboti.us/download/mjpro150_linux.zip --output mjpro150_linux.zip
unzip mjpro150_linux.zip
mkdir ~/.mujoco
mv mjpro150 ~/.mujoco/
rm mjpro150_linux.zip
cp mjkey.txt ~/.mujoco/
pip3 install mujoco_py==1.50.1.68

echo 'alias python=python3' >> ~/.bashrc
echo 'alias pip=pip3' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/brijenthananjeyan/.mujoco/mjpro150/bin' >> ~/.bashrc
. ~/.bashrc

# upload demo data (can't automate this)
