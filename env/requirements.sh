#!/bin/bash

function install_tf(){
    # Install Tensorflow
    sudo python3 -m pip install --upgrade pip
    sudo python3 -m pip install scikit-build cmake mock cython
    wget https://github.com/p513817/tf-2.4.1-no-h5py/releases/download/2.4.1/tensorflow-2.4.1-cp37-cp37m-linux_aarch64-no-h5py.whl
    mv *.whl tensorflow-2.4.1-cp37-cp37m-linux_aarch64.whl
    sudo pip3 install tensorflow-2.4.1-cp37-cp37m-linux_aarch64.whl
    rm tensorflow-2.4.1-cp37-cp37m-linux_aarch64.whl
}

# Check Tensorflow
RET=$(pip3 list | grep tensorflow)
if [[ -z ${RET} ]]; then
    install_tf;
else

    TF_VER=$(python3 -c "import tensorflow as tf; print(tf.__version__)")
    

fi

# Python Module
pip3 install --no-cache-dir \
    wget \
    colorlog \
    tqdm==4.64.0 \
    cython==0.29.32 \
    setuptools==52.0.0 \
    packaging==21.3 \
    gdown==4.5.4 \
    flasgger==0.9.5 \
    Flask==2.0.3 \
    flask-sock==0.5.2 \
    python-dateutil==2.8.2 \
    Flask-Cors==3.0.10 \
    flask-socketio==5.1.2 \
    gunicorn==20.1.0 \
    eventlet==0.30.2 \
    python-dateutil==2.8.2 \
    python-engineio==4.3.2 \
    python-socketio==5.6.0 \
    flask_mqtt==1.1.1 \
    paho-mqtt==1.6.1