#!/bin/bash

cat << EOF

# Install Tensorflow 2.4.1 for iVIT-I

## Workflow
    1. Download tensorflow-2.4.1-cp37-cp37m-linux_aarch64.whl from Github Release Page.
    2. Install .wheel directly.

## Note
    14:12 - 
    1. It takes about 20 mins to install on K260.

EOF

# Pre-requirement
sudo python3 -m pip install --upgrade pip
sudo python3 -m pip install scikit-build cmake mock cython

# Download wheel
wget \
https://github.com/p513817/tf-2.4.1-no-h5py/releases/download/2.4.1/tensorflow-2.4.1-cp37-cp37m-linux_aarch64-no-h5py.whl

# Install
mv *.whl tensorflow-2.4.1-cp37-cp37m-linux_aarch64.whl
sudo pip3 install tensorflow-2.4.1-cp37-cp37m-linux_aarch64.whl

# Delete wheel
rm tensorflow-2.4.1-cp37-cp37m-linux_aarch64.whl