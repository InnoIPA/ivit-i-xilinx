# Fast Testing
Fast Testing for `Xilinx` platform:
1. Download data and model from Google Drive and Xilinx AI Model Zoo.
2. Run demo script if you give the argument `-r`, you could use `-s` to run non-interactive mode.

* Run the testing script
    ```bash
    ./docker/run.sh -c
    <script.sh> < -r > < -s > < -h >
    ```
    |   name    |   descr                   
    |   ----    |   -----
    |   `-r`    |   run the demo script and display the result
    |   `-s`    |   server mode, only print the result not dispay
    |   `-h`    |   show help information

* Run script outside the docker container

    ```bash
    docker start ivit-i-xilinx
    docker exec -it ivit-i-xilinx <script.sh> < -r | s | h >
    ```

* Examples
    * classification.sh
        ```bash
        docker exec -it ivit-i-xilinx ./test/classification.sh -r
        ```
    * object_detection.sh
        ```bash
        docker exec -it ivit-i-xilinx ./test/object_detection.sh -r
        ```