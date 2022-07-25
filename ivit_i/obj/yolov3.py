# Copyright (c) 2022 Innodisk Crop.
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import sys, os, copy, logging, time, random, colorsys
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Lambda

sys.path.append("..")
from ivit_i.common.common import XMODEL, runDPU_, IVIT_MODEL
from ivit_i.common.pre_post_proc import preprocess_fn
from ivit_i.utils.parser import load_txt
from ivit_i.utils.err_handler import handle_exception

# Define Config Key
TAG         = "tag"
FRAMEWORK   = "framework"
TRG         = "vitis-ai"
MODEL_PATH  = "model_path"
LABEL_PATH  = "label_path"
INPUT_SIZE  = "input_size"
TYPE        = "type"
ANCHORS     = "anchors"
IOU         = "iou"
NMS         = "nms"
THRES       = "thres"
BOX_MAX_NUM = "box_max_num"

# Define Result Dictionary Key
DETECTION   = "detections"
ID          = "id"
LABEL       = "label"
SCORE       = "score"
XMIN        = "xmin"
XMAX        = "xmax"
YMIN        = "ymin"
YMAX        = "ymax"


def _meshgrid(n_a, n_b):
    """
    As tensorflow lite doesn't support tf.size used in tf.meshgrid, 
    we reimplemented a simple meshgrid function that use basic tf function.
    """
    return [
        tf.reshape(tf.tile(tf.range(n_a), [n_b]), (n_b, n_a)),
        tf.reshape(tf.repeat(tf.range(n_b), n_a), (n_b, n_a))
    ]

def yolo_boxes(pred, anchors, classes):
    """
    pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...classes))
    """
    grid_size = tf.shape(pred)[1:3]
    box_xy, box_wh, objectness, class_probs = tf.split(
        pred, (2, 2, 1, classes), axis=-1)

    box_xy = tf.sigmoid(box_xy)
    objectness = tf.sigmoid(objectness)
    class_probs = tf.sigmoid(class_probs)
    pred_box = tf.concat((box_xy, box_wh), axis=-1)  # original xywh for loss

    # !!! grid[x][y] == (y, x)
    grid = _meshgrid(grid_size[1],grid_size[0])
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # [gx, gy, 1, 2]

    box_xy = (box_xy + tf.cast(grid, tf.float32)) / \
        tf.cast(grid_size, tf.float32)
    box_wh = tf.exp(box_wh) * anchors

    box_x1y1 = box_xy - box_wh / 2
    box_x2y2 = box_xy + box_wh / 2
    bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)

    return bbox, objectness, class_probs, pred_box

def yolo_nms(outputs, classes, max_output, iot_th, score_th):
    """
    boxes, conf, type
    """
    b, c, t = [], [], []

    for o in outputs:
        b.append(tf.reshape(o[0], (tf.shape(o[0])[0], -1, tf.shape(o[0])[-1])))
        c.append(tf.reshape(o[1], (tf.shape(o[1])[0], -1, tf.shape(o[1])[-1])))
        t.append(tf.reshape(o[2], (tf.shape(o[2])[0], -1, tf.shape(o[2])[-1])))

    bbox = tf.concat(b, axis=1)
    confidence = tf.concat(c, axis=1)
    class_probs = tf.concat(t, axis=1)

    scores = confidence * class_probs

    dscores = tf.squeeze(scores, axis=0)
    scores = tf.reduce_max(dscores,[1])
    bbox = tf.reshape(bbox,(-1,4))
    classes = tf.argmax(dscores,1)

    selected_indices, selected_scores = tf.image.non_max_suppression_with_scores(
        boxes=bbox,
        scores=scores,
        max_output_size=max_output,
        iou_threshold=iot_th,
        score_threshold=score_th,
        soft_nms_sigma=0.6
    )
    
    num_valid_nms_boxes = tf.shape(selected_indices)[0]

    selected_indices = tf.concat([selected_indices,tf.zeros(max_output-num_valid_nms_boxes, tf.int32)], 0)
    selected_scores = tf.concat([selected_scores,tf.zeros(max_output-num_valid_nms_boxes,tf.float32)], -1)

    boxes = tf.gather(bbox, selected_indices)
    boxes = tf.expand_dims(boxes, axis=0)

    scores = selected_scores
    scores = tf.expand_dims(scores, axis=0)

    classes = tf.gather(classes,selected_indices)
    classes = tf.expand_dims(classes, axis=0)
    
    valid_detections = num_valid_nms_boxes
    valid_detections = tf.expand_dims(valid_detections, axis=0)

    return boxes, scores, classes, valid_detections


class YOLOV3_PATTERN(XMODEL):

    def __init__(self, model, name, input_size, box_max_num, anchors, classed_num, iou, nms, thres):
        """
        YOLOV3 PATTERN is made by Innodisk IPA Team 4 which inherit XMODEL
        """
        super().__init__(model, name)
        
        self.prediction = []
        self.anchors = anchors
        self.classes = classed_num
        self.iou_th = iou
        self.nms_th = nms
        self.conf_th = thres
        self.box_num = box_max_num
        self.input_size = input_size

    def sort_boxes(self):
        """
        Sort bounding boxes.
        
        YOLO        : YOLO Layer is 3
        YOLO-TINY   : YOLO Layer is 2
        """
        if len(self.outputs) == 3:
            boxes = np.array([[0, 1, 2], [0, 0, 0]], dtype=int)
            boxes[1][0] = self.outputs[0].shape[1]
            boxes[1][1] = self.outputs[1].shape[1]
            boxes[1][2] = self.outputs[2].shape[1]

        elif len(self.outputs) == 2:
            boxes = np.array([[0, 1], [0, 0]], dtype=int)
            boxes[1][0] = self.outputs[0].shape[1]
            boxes[1][1] = self.outputs[1].shape[1]

        sorted = list(np.argsort(boxes[1:]))
        sorted = sorted[0]

        return sorted

    def pred_boxes(self):
        """
        Parse information from bounding box.

        YOLO        : YOLO Layer is 3
        YOLO-TINY   : YOLO Layer is 2
        """
        if len(self.outputs) == 3:
            a64 = np.reshape(self.outputs[self.sorted[2]], (1, int(self.input_size/8), int(self.input_size/8), 3, 5+self.classes))
            a32 = np.reshape(self.outputs[self.sorted[1]], (1, int(self.input_size/16), int(self.input_size/16), 3, 5+self.classes))
            a16 = np.reshape(self.outputs[self.sorted[0]], (1, int(self.input_size/32), int(self.input_size/32), 3, 5+self.classes))

            b0 = Lambda(lambda x: yolo_boxes(x, self.anchors[:3], self.classes), name='yb0')(a64)
            b1 = Lambda(lambda x: yolo_boxes(x, self.anchors[3:6], self.classes), name='yb1')(a32)
            b2 = Lambda(lambda x: yolo_boxes(x, self.anchors[6:], self.classes), name='yb2')(a16)

            self.prediction = Lambda(
                lambda x: yolo_nms(x, self.classes, self.box_num, self.iou_th, self.nms_th),
                name='yolo_nms')((b0[:3], b1[:3], b2[:3]))

            boxes, scores, classes, nums = self.prediction
            return np.array(boxes), np.array(scores), np.array(classes), np.array(nums)

        elif len(self.outputs) == 2:
            o_13 = self.outputs[0]
            o_26 = self.outputs[1]

            a32 = np.reshape(o_13, (1, int(self.input_size/32), int(self.input_size/32), 3, 5+self.classes))
            a16 = np.reshape(o_26, (1, int(self.input_size/16), int(self.input_size/16), 3, 5+self.classes))

            b1 = Lambda(lambda x: yolo_boxes(x, self.anchors[3:6], self.classes), name='yb0')(a32)
            b2 = Lambda(lambda x: yolo_boxes(x, self.anchors[0:3], self.classes), name='yb1')(a16)

            self.prediction =  Lambda(
                lambda x: yolo_nms(x, self.classes, self.box_num, self.iou_th, self.nms_th),
                name='yolo_nms')((b1[:3], b2[:3]))

            boxes, scores, classes, nums = self.prediction

            return np.array(boxes), np.array(scores), np.array(classes), np.array(nums)

class YOLOv3(IVIT_MODEL):

    def __init__(self, cfg):
        """
        YOLOv3 Object for IVIT-I-Xilinx
        """
        self.cfg    = cfg
        self.init_status = False

    def init_model(self):
        """
        Initialize Model.
        - Return: 
            - a bool value, which means is initialize successfuly or not.
        """
        
        # Get information from config
        self.model_path = self.cfg[TRG][MODEL_PATH]
        self.classes = load_txt(self.cfg[TRG][LABEL_PATH])
        self.input_size = tuple(map(int, self.cfg[TRG][INPUT_SIZE].split(",")))[1:]        
        self.classed_num = len(self.classes)

        self.iou = float(self.cfg[TRG][IOU])
        self.nms = float(self.cfg[TRG][NMS])
        self.thres = float(self.cfg[TRG][THRES])
        self.box_max_num = int(self.cfg[TRG][BOX_MAX_NUM])
        raw_anchors = self.cfg[TRG][ANCHORS]

        # Set anchors
        rows, cols = int(len(raw_anchors) / 2), 2
        self.anchors = np.zeros(shape=(rows, cols), dtype=float)
        for i in range(rows):
            for j in range(cols):
                self.anchors[i][j] = int(raw_anchors[cols * i + j]) / self.input_size[0]

        # Initialize YOLO Model        
        try:
            time_init = time.time()
            self.x = YOLOV3_PATTERN(    model=self.model_path, 
                                        name='yolov3', 
                                        input_size=self.input_size[0], 
                                        box_max_num=self.box_max_num, 
                                        anchors=self.anchors, 
                                        classed_num=self.classed_num, 
                                        iou=self.iou, 
                                        nms=self.nms, 
                                        thres=self.thres  )
            self.x.init()
            self.init_status = True
            logging.debug("Init {} times = {:.4f} seconds".format('yolov3', time.time() - time_init))
        
        except Exception as e:
            handle_exception(e)
            
        return self.init_status

    def inference(self, frame):
        """
        Do Inference.
        - Argument:
            - frame : frame is an image data
        - Return:
            - describe : a python dictionary include the result of detection
            - example : 
                {[{
                    "id"        : 0,
                    "score"     : 0.9123,
                    "label"     : cat,
                    "xmin"      : 32
                    "ymin"      : 435
                    "xmax"      : 42
                    "ymax"      : 654
                }]}
        """

        # Process Input Data
        time_start = time.time()
        img = []
        image = preprocess_fn(frame, self.input_size)
        img.append(image)

        # Do Inference
        self.x.outputs = runDPU_(self.x, img[0:])

        # Parse information from Bounding Box
        time_pred_box = time.time()
        self.x.sorted = self.x.sort_boxes()
        self.p_boxes, self.p_scores, self.p_classes, self.p_nums = self.x.pred_boxes()
        logging.debug("bb output times = {:.4f} seconds".format(time.time() - time_pred_box))

        # Parse the result
        logging.info("Detections: {}".format( len(self.p_nums) ))
        info = { DETECTION: list() }
        for i in range(self.p_nums[0]):
            
            # Add Info and Show Result
            temp_pattern = self.get_dets_pattern()
            temp_pattern[ID]      = int(self.p_classes[0][i])
            temp_pattern[SCORE]   = float(self.p_scores[0][i])
            temp_pattern[LABEL]   = self.classes[int(self.p_classes[0][i])]

            # Rescale BBOX to image size
            bbox = np.array(self.p_boxes[0][i])     # x1, y1, x2, y2
            wh = np.flip(frame.shape[0:2])          # hwc to wh
            temp_pattern[XMIN], temp_pattern[YMIN] = np.array((bbox[0:2] * wh)).astype(np.int32)
            temp_pattern[XMAX], temp_pattern[YMAX] = np.array((bbox[2:4] * wh)).astype(np.int32)

            # Update info
            info[DETECTION].append( temp_pattern )
            logging.info("\t[{}] {}".format(i, temp_pattern))
        
        # Calculate During Time and Throughput
        time_end = time.time()
        time_total = time_end - time_start
        fps = 1 / time_total
        
        logging.info("Throughput = {:.2f} fps, time = {:.4f} seconds".format(fps, time_total))

        return info
