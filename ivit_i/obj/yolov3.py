# Copyright (c) 2022 Innodisk Crop.
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import sys, os, copy, logging, time, random, colorsys
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
	Lambda,
)

try:
    import tensorflow as tf
except:
    pass

sys.path.append("..")
from ivit_i.common.common import XMODEL, runDPU_
from ivit_i.common.config_key_list import *
from ivit_i.common.pre_post_proc import preprocess_fn

# As tensorflow lite doesn't support tf.size used in tf.meshgrid, 
# we reimplemented a simple meshgrid function that use basic tf function.
def _meshgrid(n_a, n_b):
    return [
        tf.reshape(tf.tile(tf.range(n_a), [n_b]), (n_b, n_a)),
        tf.reshape(tf.repeat(tf.range(n_b), n_a), (n_b, n_a))
    ]

def yolo_boxes(pred, anchors, classes):
    # pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...classes))
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
    # boxes, conf, type
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

	def __init__(self, model, name, input_size, box_max_num, anchors, classed_num, iou, nms, conf):
		super().__init__(model, name)
		self.prediction = []
		self.anchors = anchors
		self.classes = classed_num
		self.iou_th = iou
		self.nms_th = nms
		self.conf_th = conf
		self.box_num = box_max_num
		self.input_size = input_size

	def sort_boxes(self):
        """
        
        """
		if len(self.outputs) == 3:
			''' yolo '''
			boxes = np.array([[0, 1, 2], [0, 0, 0]], dtype=int)
			boxes[1][0] = self.outputs[0].shape[1]
			boxes[1][1] = self.outputs[1].shape[1]
			boxes[1][2] = self.outputs[2].shape[1]

		elif len(self.outputs) == 2:
			''' tiny yolo '''
			boxes = np.array([[0, 1], [0, 0]], dtype=int)
			boxes[1][0] = self.outputs[0].shape[1]
			boxes[1][1] = self.outputs[1].shape[1]

		sorted = list(np.argsort(boxes[1:]))
		sorted = sorted[0]

		return sorted

	def pred_boxes(self):
		if len(self.outputs) == 3:
			''' yolo '''
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
			''' tiny yolo '''
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

			# a32 = np.reshape(self.outputs[self.sorted[1]], (1, int(self.input_size/16), int(self.input_size/16), 3, 5+self.classes))
			# a16 = np.reshape(self.outputs[self.sorted[0]], (1, int(self.input_size/32), int(self.input_size/32), 3, 5+self.classes))

			# b0 = Lambda(lambda x: yolo_boxes(x, self.anchors[0:3], self.classes), name='yb1')(a32)
			# b1 = Lambda(lambda x: yolo_boxes(x, self.anchors[3:], self.classes), name='yb2')(a16)

			# self.prediction = Lambda(
			# 		lambda x: yolo_nms(x, self.classes, self.box_num, self.iou_th, self.nms_th),
			# 		name='yolo_nms')((b0[:3], b1[:3]))

class YOLOv3():

    def __init__(self, cfg):
        self.cfg    = cfg

        self.run_model  = None
        self.output     = None
        self.get_frame  = None

        self.f_first_run_model    = True
        self.f_first_output       = True
        self.f_first_get_frame    = True

    def init_model(self):
        ''' Get config from cfg '''
        self.iou = float(self.cfg[MODLES][XMODELS_OBJ][IOU])
        self.nms = float(self.cfg[MODLES][XMODELS_OBJ][NMS])
        self.conf = float(self.cfg[MODLES][XMODELS_OBJ][CONF])
        self.box_max_num = int(self.cfg[MODLES][XMODELS_OBJ][BOX_MAX_NUM])
        raw_anchors = self.cfg[MODLES][XMODELS_OBJ][ANCHORS]

        self.model = self.cfg[MODLES][XMODELS_OBJ][MODEL]
        self.classes = self.cfg[MODLES][XMODELS_OBJ][CLASS]
        self.input_size = self.cfg[MODLES][XMODELS_OBJ][INPUT_SIZE]

        self.label_classes = len(self.classes)

        ''' Set anchors '''
        rows = int(len(raw_anchors) / 2)
        cols = 2

        self.anchors = np.zeros(shape=(rows, cols), dtype=float)
        for i in range(rows):
            for j in range(cols):
                self.anchors[i][j] = int(raw_anchors[cols * i + j]) / self.input_size[0]

        # random color for predict box
        self.color_list = []
        for i in range(len(self.classes)):
            h, s, l = random.random(), 0.5 + random.random() / 2.0, 0.4 + random.random() / 5.0
            r, g, b = [int(256 * i) for i in colorsys.hls_to_rgb(h, l, s)]
            self.color_list.append([r, g, b])

        time_init_start = time.time()
        ret = True
        
        try:
            self.x = YOLOV3_PATTERN(self.model, 'yolov3', self.input_size[0], self.box_max_num, self.anchors, self.label_classes, self.iou, self.nms, self.conf)
            self.x.init()
            logging.debug("Init yolov3 times = {:.4f} seconds".format(time.time() - time_init_start))
        
        except Exception as e:
            ret = False
            logging.error(e)
            
        return ret

    def inference(self, frame):

        time1 = time.time()
        img = []
        image = preprocess_fn(frame, self.input_size)
        img.append(image)

        # Do Inference
        self.x.outputs = runDPU_(self.x, img[0:])

        time_pred_box = time.time()
        
        # Sort Bounding Box
        self.x.sorted = self.x.sort_boxes()
        # self.x.prediction = self.x.pred_boxes()
        logging.debug("bb output times = {:.4f} seconds".format(time.time() - time_pred_box))

        # Parse Bounding Box
        self.p_boxes, self.p_scores, self.p_classes, self.p_nums = self.x.pred_boxes()

        time2 = time.time()
        time_total = time2 - time1

        fps = 1 / time_total

        logging.info(" Throughput={:.2f} fps, total frames = {:.0f}, time = {:.4f} seconds".format(fps, 1, time_total))

        logging.info(' Detections: {}'.format( len(self.p_nums) ))

        for i in range(self.p_nums[0]):
            logging.info('\t{}, {}, {}'.format(self.classes[int(self.p_classes[0][i])],
												np.array(self.p_scores[0][i]),
												np.array(self.p_boxes[0][i])))
            
            # frame = draw_outputs(frame, (self.p_boxes, self.p_scores, self.p_classes, self.p_nums), self.classes, i, self.color_list[int(self.p_classes[0][i])], fps)

        return frame
