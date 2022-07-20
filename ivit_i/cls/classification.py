import sys, os, copy, logging, time
import numpy as np
sys.path.append("..")
from ivit_i.common.common import XMODEL, runDPU_
from ivit_i.common.config_key_list import *
from ivit_i.common.pre_post_proc import preprocess_fn


class Classification():
    
    def __init__(self, cfg):
        self.cfg = cfg

    def init_model(self):
        
        self.model = self.cfg[MODLES][XMODELS_CLASS][MODEL]
        self.classes = self.cfg[MODLES][XMODELS_CLASS][CLASS]
        self.input_size = self.cfg[MODLES][XMODELS_CLASS][INPUT_SIZE]

        time_init_start = time.time()
        ret = True
        try:
            self.x = XMODEL(self.model, 'cnn')
            self.x.init()
            logging.debug("Init cnn times = {:.4f} seconds".format(time.time() - time_init_start))
        except Exception as e:
            ret = False
        
        return ret

    def inference(self, frame):
        time1 = time.time()
        img = []
        img.append(preprocess_fn(frame, self.input_size))


        self.x.outputs = runDPU_(self.x, img[0:])

        prediction = self.classes[np.argmax(self.x.outputs[0][0])]

        time2 = time.time()
        time_total = time2 - time1
        fps = 1 / time_total

        logging.info("Throughput={:.2f} fps, total frames = {:.0f}, time={:.4f} seconds".format(fps, 1, time_total))

        logging.info("Detection: {}".format(prediction))

        ''' Put fps and pridict class on image '''
        # frame = cv2.putText(frame, '{} fps: {:.4f}'.format(prediction, fps), (10, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2)

        return frame
