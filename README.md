![LOGO](docs/images/iVIT-I-Logo-B.png)

# iVIT-I-Xilinx
iNIT-I is an AI inference tool which could support multiple AI framework and this repository is just for xilin platform.
* [Pre Requirements](#pre-requirements)
* [Prepare Environment](#prepare-environment)
* [Run Sample](#run-sample)
* [Fast Testing](#fast-testing)
* [Web API](#web-api)
* [Supported Model](#supported-model)

# Pre-requirements
|   Require     | Desc   
|   ---         | --- 
|   Hardware    | Xilinx K26
|   System      | Petalinux 
|   Mics        | Vitis-AI Available 
|               | Docker

# Prepare Environment
1. Download iVIT-I with Web API
    ```bash
    git clone https://github.com/InnoIPA/ivit-i-xilinx.git
    cd ivit-i-xilinx
    ```
2. Install requirement
    ```bash
    sudo -s
    chmod +x ./requirements.sh && ./requirements.sh
    ```
3. Prepare AI Module
    ```bash
    # Load AI App for Vitis-AI, load-xilinx-app.sh < App Name >
    ./env/load-xilinx-app.sh VCU200EDDPU200B4096
    ```
4. Ignore IVIT verify
    ```bash
    export IVIT_DEBUG=True
    ```

# Run Sample
We use `task.json` to configure each AI tasks and using `<model>.json` to configure each AI models, check [ task configuration ](./docs/task_configuration.md) and [model configuration](./docs/model_configuration.md) to get more detail.

1. Prepare model and meta data.
    ```bash
    # Model
    ./task/classification_sample/download_model.sh
    # Meta data
    ./task/classification_sample/download_data.sh
    ```
2. Run demo script
    * GUI Mode
        ```bash
        python3 demo.py -c task/classification_sample/task.json
        ```
    * CLI Mode
        ```bash
        python3 demo.py -c task/classification_sample/task.json -s
        ```
    * RTSP Mode
        ```bash
        # Launch RTSP Server Fist
        ./env/rtsp-server.sh start
        python3 demo.py -c task/classification_sample/task.json -r
        ```
    * Usage
        ``` bash
            python3 demo.py -h

            usage: demo.py [-h] [-c CONFIG] [-s] [-r] [-m MODE] [-i IP] [-p PORT] [-n NAME]

            optional arguments:
            -h, --help            show this help message and exit
            -c CONFIG, --config CONFIG The path of application config
            -s, --server          Server mode, not to display the opencv windows
            -r, --rtsp            RTSP mode, not to display the opencv windows
            -m MODE, --mode MODE  Select sync mode or async mode{ 0: sync, 1: async }
            -i IP, --ip IP        The ip address of RTSP uri
            -p PORT, --port PORT  The port number of RTSP uri
            -n NAME, --name NAME  The name of RTSP uri
        ```

# Run Web Service
* Run Web API
    ```bash
    sudo ./exec_web_api.sh
    ```

# Stop micro-service
```bash
./env/stream-server.sh stop
```

# Fast Testing
We provide the fast-test for each sample, please check the [document](./test/README.md).

# Web API
<details>
    <summary>
        We recommand <a href="https://www.postman.com/">Postman</a> to test your web api , you could see more detail in <code>{IP Address}:{PORT}/apidocs</code>.
    </summary>
    <img src="docs/images/apidocs.png" width=80%>
</details>
<br>

# Supported Model
* iVIT-I support the pre-trained model which download from [AI Model Zoo](https://github.com/Xilinx/Vitis-AI/tree/v1.4.1/models/AI-Model-Zoo/model-list).
* Tag: `1.4.X`

|   Name        |   Type    |   Description 
|   ---         |   ---     |   ---
|   Mobile Net  |   Classification  |   https://github.com/Xilinx/Vitis-AI/blob/1.4.1/models/AI-Model-Zoo/model-list/tf2_mobilenetv1_imagenet_224_224_1.15G_1.4/model.yaml
|   YOLOv3      |   Object Detection    |   https://github.com/Xilinx/Vitis-AI/blob/v1.4/models/AI-Model-Zoo/model-list/tf_yolov3_voc_416_416_65.63G_1.4/