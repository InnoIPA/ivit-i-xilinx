import os
import cv2
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
from ivit_i.utils.drawing_tools import Draw, get_palette
from ivit_i.utils.err_handler import handle_exception

from ivit_i.app.handler import get_application

CV_WIN  =   "Test"

# legacy
# from mod.predictor import PREDICTOR
# from mod.util import open_json

def main(args):
    
    app_conf = load_json(args.config)                       # load the configuration of the application
    model_conf = load_json(app_conf['prim']['model_json'])             # load the configuration of the AI model
    
    total_conf = model_conf.copy()
    total_conf.update(app_conf)

    logging.debug(total_conf)

    # Load iVIT-I API
    try:
        logging.debug("Loading ivit_i API")
        trg = api.get(total_conf)
        draw = Draw()
        palette = get_palette(total_conf)
    except Exception as e:
        handle_exception(error=e, title="Could not get ivit-i API", exit=True)

    # Load Model
    try:
        ret = trg.init_model()
    except Exception as e:
        handle_exception(error=e, title="Could not load AI model", exit=True)

    # Get source object
    src = Source(total_conf['source'], total_conf['source_type'])

    # Get Application
    if not args.server:
        has_app=False
        try:
            application = get_application(total_conf)
            has_app = False if application == None else True

            # Setup parameter if needed
            app_info = total_conf["application"]

            # Area detection: point_points
            if "area" in app_info["name"]:
                key = "area_points"
                if not key in app_info: 
                    application.set_area(pnts=None, frame=src.get_first_frame())
                else: 
                    application.set_area(pnts=app_info[key])   

        except Exception as e:
            handle_exception(error=e, title="Could not load application ... set app to None", exit=False)
            has_app=False

    # Main Loop
    try:
        while True:
            
            ret_frame, frame = src.get_frame()

            draw_frame = frame.copy()

            info = trg.inference(frame)

            if args.server:
                logging.info(info)
                continue
            
            if not has_app:
                info["frame"]=draw_frame
                draw_frame = draw.draw_detections(info, palette, total_conf)
            else:
                draw_frame = application(draw_frame, info)

            # show the results            
            cv2.imshow(CV_WIN, draw_frame)
            key = cv2.waitKey(1)
            if key in [ ord('q'), ord('Q'), 27 ]:
                break 

        src.release()

    except KeyError as e:
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='the path of configuration.')
    parser.add_argument('-d', '--debug', action="store_true", help='the debug mode.')
    parser.add_argument('-s', '--server', action="store_true", help='the server mode.')
    args = parser.parse_args()
    main(args)    
        