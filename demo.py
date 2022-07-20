import os
import argparse
import time
import numpy as np
import queue
import logging
import datetime
import sys
import copy

# ivit
from ivit_i.utils.logger import config_logger
from ivit_i.utils.parser import load_json
from ivit_i.common import api
from ivit_i.common.pipeline import Source
from ivit_i.utils.drawing_tools import Draw
from ivit_i.utils.err_handler import handle_exception

# legacy
# from mod.predictor import PREDICTOR
# from mod.util import open_json

def main(args):
    
    config_logger(log_name='ivit-i-xilinx.log', write_mode='w', level='debug')

    app_conf = load_json(args.config)                       # load the configuration of the application
    model_conf = load_json(app_conf['prim']['model_json'])             # load the configuration of the AI model
    
    total_conf = model_conf.copy()
    total_conf.update(app_conf)

    logging.debug(total_conf)

    try:
        logging.debug("Loading ivit_i API")
        trg = api.get(total_conf)
        draw = Draw()
    except Exception as e:
        handle_exception(error=e, title="Could not get ivit-i API", exit=True)

    try:
        ret = trg.init_model()
        
    except Exception as e:
        handle_exception(error=e, title="Could not load AI model", exit=True)

    src = Source(total_conf['source'], total_conf['source_type'])

    try:
        while True:
            
            ret_frame, frame = src.get_frame()

            org_frame = frame.copy()

            frame = trg.inference(org_frame)
    except KeyError as e:
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='the path of configuration.')
    parser.add_argument('-d', '--debug', action="store_true", help='the debug mode.')
    parser.add_argument('-s', '--server', action="store_true", help='the server mode.')
    args = parser.parse_args()
    main(args)    
        