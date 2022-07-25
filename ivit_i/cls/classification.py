import sys, os, copy, logging, time
import numpy as np
sys.path.append("..")
from ivit_i.common.common import XMODEL, IVIT_MODEL, runDPU_
from ivit_i.common.pre_post_proc import preprocess_fn
from ivit_i.utils.parser import load_txt
from ivit_i.utils.err_handler import handle_exception

# Define Key
TAG         = "tag"
FRAMEWORK   = "framework"
MODEL_PATH  = "model_path"
LABEL_PATH  = "label_path"
INPUT_SIZE  = "input_size"
TYPE        = "type"

# Define Result Dictionary Key
DETECTION   = "detections"
ID          = "id"
LABEL       = "label"
SCORE       = "score"
XMIN        = "xmin"
XMAX        = "xmax"
YMIN        = "ymin"
YMAX        = "ymax"


class Classification(IVIT_MODEL):
    
    def __init__(self, cfg):
        """
        Classification Object for IVIT-I-Xilinx
        """
        self.cfg = cfg
        self.init_status = False

    def init_model(self) -> bool:
        """
        Initialize Model.
        - Return: 
            - a bool value, which means is initialize successfuly or not.
        """
        # Parse Information From Config File
        self.model_path = self.cfg[MODEL_PATH]
        self.classes = load_txt(self.cfg[LABEL_PATH])
        self.input_size = tuple(map(int, self.cfg[INPUT_SIZE].split(",")))

        # Initialize XModel
        time_init_start = time.time()
        try:
            self.x = XMODEL(self.model_path, 'cnn')
            self.x.init()
            self.init_status = True
            logging.debug("Init cnn times = {:.4f} seconds".format(time.time() - time_init_start))
        except Exception as e:
            handle_exception(e)
            
        return self.init_status

    def inference(self, frame) -> dict:
        """
        Do Inference.
        - Argument:
            - frame : frame is an image data
        - Return:
            - describe : a python dictionary include the result of detection
            - example : 
                {[{
                    "id"        : 0,
                    "label"     : cat,
                    "score"     : None,
                    "xmin"      : None
                    "ymin"      : None
                    "xmax"      : None
                    "ymax"      : None
                }]}
        """
        # Process Input Data
        time_start = time.time()
        img = []
        img.append(preprocess_fn(frame, self.input_size[1:]))

        # Do Inference
        self.x.outputs = runDPU_(self.x, img[0:])

        # Parse Result
        info = { DETECTION: list() }
        temp_pattern        = self.get_dets_pattern()
        temp_pattern[ID]    = np.argmax(self.x.outputs[0][0])
        temp_pattern[LABEL] = self.classes[temp_pattern[ID]]
        
        # Update info
        info[DETECTION].append(temp_pattern)
        logging.info(temp_pattern)
        
        # Calculate During Time and Throughput
        time_end = time.time()
        time_total = time_end - time_start
        fps = 1 / time_total
        logging.info("Throughput = {:.2f} fps, time = {:.4f} seconds".format(fps, time_total))

        return info
