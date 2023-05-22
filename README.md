![LOGO](assets/images/iVIT-I-Logo-B.png)

# iVIT-I-Xilinx
iNIT-I is an AI inference tool which could support multiple AI framework and this repository is just for xilin platform.
* [Hardware & BSP Information](#hardware-and-bsp-information)
* [Requirements](#requirements)
* [Prepare Environment](#prepare-environment)
* [Run Sample](#run-sample)
* [Fast Testing](#fast-testing)
* [Web API](#web-api)
* [Supported Model](#supported-model)

# Hardware and BSP Information
|   Name        | Desc   
|   ---         | --- 
|   Hardware    | Xilinx K26
|   System      | Petalinux 
|   Mics        | Vitis-AI 2.5.0 

# Requirements
* [Docker 20.10 + ](https://docs.docker.com/engine/install/ubuntu/)
* [Docker-Compose v2.15.1 ](https://docs.docker.com/compose/install/linux/#install-using-the-repository)
    * you can check via `docker compose version`


# Getting Start
1. Clone Repository
    
    * Donwload Target Version
        ```bash
        git clone -b r1.1 https://github.com/InnoIPA/ivit-i-xilinx.git && cd ivit-i-xilinx
        ```

2. Run iVIT-I Docker Container

    * Run CLI container
        ```bash
        sudo ./docker/run.sh

        "USAGE: ./docker/run.sh -h" << EOF
        Run the iVIT-I environment.

        Syntax: scriptTemplate [-bqh]
        options:
        b               Run in background
        q               Qucik launch iVIT-I
        h               help.
        >>
        ```

3. Run Samples

    * [Source Sample](samples/classification_sample/README.md)
    * [Displayer Sample](samples/ivit_displayer_sample/README.md)
    * [Classification Sample](samples/classification_sample/README.md)
    * [Object Detection Sample](samples/object_detection_sample/README.md)
    * [iDevice Sample](samples/ivit_device_sample/README.md)

# Python Library Documentation
[iVIT-I-Intel API Docs](https://innoipa.github.io/ivit-i-intel/)

