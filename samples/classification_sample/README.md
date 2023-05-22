# Classification Sample
iVIT Classification Sample, this sample demonstrates how to do inference of image classification models using iVIT [Source](../ivit_source_sample/README.md) and [Displayer](../ivit_displayer_sample/README.md).

## Getting Start
* Clone Repository    
    ```bash
    git clone  https://github.com/InnoIPA/ivit-i-intel.git && cd ivit-i-intel
    ```
* Run iVIT-I Docker Container
    ```bash
    sudo ./docker/run.sh
    ```
* Download Data
    ```bash
    # Move to target folder
    cd /workspace/samples/classification_sample
    
    # Download File
    chmod u+x ./*.sh
    ./download_data.sh        
    ./download_model.sh       
    ```
* Setting Varaible
    ```bash
    
    EXEC_PY="python3 ./classification_demo.py"

    ROOT=/workspace
    MODEL=${ROOT}/model/mobilenet_1_1_224_tf2/mobilenet_1_0_224_tf2.xmodel
    LABEL=${ROOT}/model/mobilenet_1_1_224_tf2/imagenet.txt
    INPUT=${ROOT}/data/cat.jpg
    ```

* Run Sample
    ```bash
    ${EXEC_PY} -m ${MODEL} -l ${LABEL} -i ${INPUT} --no_show
    ```

## Usage
* Help
    ```bash
    ${EXEC_PY} -h
    ```

* Top 5 Labels
    ```bash
    ${EXEC_PY} -m ${MODEL} -l ${LABEL} -i ${INPUT} \
    -topk 5
    ```
    
## Format of output 
*  The format of result after model predict like below.

| Type | Description |
| --- | --- |
|tuple|( id, label, score )|
* Example:
    ```bash
        id      # (type int)           value : 0   
        label   # (type str)           value : cat   
        score   # (type numpy.float32) value : 0.5921569      
    ```