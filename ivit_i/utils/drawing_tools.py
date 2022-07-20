import logging, random, os, sys, cv2, colorsys, time
import numpy as np
sys.path.append(os.getcwd())                                # import local module
from ivit_i.utils.parser import load_json, load_txt
from ivit_i.utils.logger import config_logger

# ================================================================================================================
# Global Variable
FONT = cv2.FONT_HERSHEY_COMPLEX_SMALL
THICK = 1
FONT_SIZE = 1.0
PADDING = 10

# ================================================================================================================
# Main Function
# ================================================================================================================
# Entry

# class 
class Draw:
    def __init__(self) -> None:
        pass

    def draw_detections(self, info, palette, conf):

        frame = info['frame']
        output_resolution = info['output_resolution']
        if 'cls' in conf['tag']:
            frame_ret = self.draw_cls(frame, info, palette)
        elif 'obj' in conf['tag']:
            frame_ret = self.draw_obj(frame, info, palette)
        elif 'darknet' in conf['tag']:
            frame_ret = self.draw_obj(frame, info, palette)
        elif 'semseg' in conf['tag']:
            frame_ret = self.draw_semseg(frame, info, palette)
        elif 'seg' in conf['tag']:
            frame_ret = self.draw_seg(frame, info, palette) 
        elif 'pose' in conf['tag']:
            frame_ret = self.draw_pose(frame, info, palette) 
        return frame_ret
    # ----------------------------------------------------------------------------------------------------------------
    # For Classification
    def draw_cls(self, frame, info, palette):
        
        h,w,c = frame.shape
        for idx, det in enumerate(info['detections']):        
            if det:
                class_id = det['id']
                class_name = det['label']
                class_score = det['score']
                content = '{} {:.1%}'.format(class_name, class_score)
                # --------------------------------------------------------
                (t_width, t_height) = get_text_size(content)
                xmin = max(int(det['xmin']), 0) if det['xmin'] else 0+PADDING
                ymin = max(int(det['ymin']), 0) if det['ymin'] else 0+t_height+PADDING
                frame = put_text(frame, content, (xmin, ymin), palette[class_id][::-1])

        return frame
    # ----------------------------------------------------------------------------------------------------------------
    # For Objected Detection
    def draw_obj(self, frame, info, palette):
        
        h,w,c = frame.shape
        # ret_frame = frame.copy()
        for idx, det in enumerate(info['detections']):  
            
            class_id = det['id']
            class_name = det['label']
            class_score = det['score']
            # x1, x2, y1, y2 = det['xmin'], det['xmax'], det['ymin'], det['ymax']
            x1, x2, y1, y2 = map(int, [ det['xmin'], det['xmax'], det['ymin'], det['ymax'] ] )
            # --------------------------------------------------------
            content = '{} {:.1%}'.format(class_name, class_score)
            # --------------------------------------------------------
            (t_width, t_height) = get_text_size(content)
            
            frame = cv2.rectangle(frame, (x1,y1), (x2,y2), palette[class_id][::-1], 2)
            frame = put_text(frame, content, (x1, y1-10), palette[class_id][::-1])

        return frame
    # ----------------------------------------------------------------------------------------------------------------
    # For Segmantic Segmentation
    def draw_semseg(self, frame, info, palette):
        
        src_h, src_w, src_c = frame.shape
        trg_c, trg_h, trg_w = info["input_size"]
        
        _frame = frame.copy()
        
        for det in info['detections']:  
            # parse mask
            # ======================================================================================================
            classes= len(palette)       # palette
            
            _frame = frame.copy()
            mask = det['mask']

            for i in range(classes):        
                trg_mask = mask[...,i]
                bool_mask = np.where(trg_mask>0.5, 1, 0).astype(np.uint8)
                print(bool_mask.shape, type(bool_mask))
                new_mask = cv2.resize(bool_mask, (src_w, src_h), interpolation=cv2.INTER_NEAREST )
                new_mask = self.apply_color_map(new_mask, palette)
                _frame = np.floor_divide(_frame, 2) + np.floor_divide(new_mask, 2)  # 向下取整 7/2=3

        return _frame

    def create_color_map(self, color_palette):
        classes = np.array(color_palette, dtype=np.uint8)[:, ::-1] # RGB to BGR
        classes_num = len(classes)
        color_map = np.zeros((256, 1, 3), dtype=np.uint8)
        color_map[:classes_num, 0, :] = classes
        color_map[classes_num:, 0, :] = np.random.uniform(0, 255, size=(256-classes_num, 3))
        return color_map

    def apply_color_map(self, _input, color_map):
        color_map = self.create_color_map(color_map).astype(np.uint8)
        input_3d = cv2.merge([_input, _input, _input]).astype(np.uint8)
        return cv2.LUT(input_3d, color_map)
    # ----------------------------------------------------------------------------------------------------------------
    # For Instance Segmentation
    def draw_seg(self, frame, info, palette):
        
        src_h, src_w, src_c = frame.shape
        trg_c, trg_h, trg_w = info["input_size"]
        scale_h, scale_w = src_h/trg_h, src_w/trg_w
        
        _frame = frame.copy()
        _mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        t0 = time.time()    
        for det in info['detections']:  
            class_id, class_name, class_score = det['id'], det['label'], det['score']
            x1, x2, y1, y2 = det['xmin']*scale_w, det['xmax']*scale_w, det['ymin']*scale_h, det['ymax']*scale_h
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2] )
            
            content = '{} {:.1%}'.format(class_name, class_score)
            _frame = cv2.rectangle(_frame, (x1,y1), (x2,y2), palette[class_id][::-1], 2)   # draw bounding box
            _frame = put_text(_frame, content, (x1, y1-10), palette[class_id][::-1])
            _mask = self.parse_mask(_mask, det['mask'], (x1, y1, x2, y2), thres=0.5)

        new_mask = self.apply_color_map(_mask, palette)
        _frame = np.floor_divide(_frame, 2) + np.floor_divide(new_mask, 2)  # 向下取整 7/2=3

        return _frame

    # parse the mask
    def parse_mask(self, full_mask, mask, bbox, thres=0.5):
        
        # normalize
        mask = mask[1]
        mask += np.abs(np.min(mask))
        mask /= np.max(mask)                    
        
        # resize mask
        x1, y1, x2, y2 = bbox                   # bounding box with correct size.
        target_size = (x2-x1, y2-y1) 
        mask = cv2.resize( mask, target_size)       

        # set true and false depend on confidence
        mask = np.where(mask>=thres, 1, 0).astype(np.uint8)

        # Put the mask in the right location.
        full_mask[y1:y2, x1:x2] = mask
        return full_mask

    # draw mask with weights
    def draw_mask(self, image, mask, color, alpha=0.5):
        """ Apply the given mask to the image. """
        for c in range(3):
            image[:, :, c] = np.where(  mask == 1, 
                                        image[:, :, c] * (1 - alpha) + alpha * color[c],
                                        image[:, :, c])
        return image
    # ----------------------------------------------------------------------------------------------------------------
    # For Human Pose Estimation ( drawing tool will provide by itself - det['drawer] )
    def draw_pose(self, frame, info, palette):
        
        for det in info['detections']:  
            drawer = det['drawer']      
            _frame = drawer(frame, det['counts'], det['objects'], det['peaks'], palette)
            _frame = draw_sth(_frame, 'COUNTS', int(det['counts']))
        return _frame

# ================================================================================================================
# Common utilities
# ----------------------------------------------------------------------------------------------------------------
# Get random color for palette
def get_random_color( format='bgr'):
    r = random.randint(0,255)
    g = random.randint(0,255)
    b = random.randint(0,255)
    return [r, g, b] if format=='rgb' else [b, g, r]
# ----------------------------------------------------------------------------------------------------------------
# Return palette with random color
def get_palette( conf:dict) -> list:
    # init
    palette, content = list(), list()
    
    # get labels
    if not ('label_path' in conf.keys()):
        msg = "Error configuration file, can't find `label_path`"
        logging.error(msg)
        raise Exception(msg)

    label_path = conf['label_path']
    color_map_ext ='.txt'

    if conf['tag'] == 'pose':
        
        # parse the path
        labels = load_json(label_path)
        output_palette_path = "{}_colormap{}".format(os.path.splitext(label_path)[0], color_map_ext)

        # get max index
        n_keypoint = len(labels['keypoints'])
        n_skeleton = len(labels['skeleton'])
        max_len = n_keypoint if n_keypoint > n_skeleton else n_skeleton

        # update palette and the content of colormap
        logging.info("Get colormap ...")
        for label in range(max_len+1):
            color = get_random_color()                                          # get random color
            palette.append(color)                                               # get palette's color list
            content.append('{label}: {color}'.format(label=label, color=tuple(color)))  # setup content
        
        # write map table into colormap
        logging.info("Write colormap into `{}`".format(output_palette_path))
        with open(output_palette_path, 'w') as f:
            f.write('\n'.join(content))

    else:
        
        # parse the path
        labels = load_txt(label_path)
        output_palette_path = "{}_colormap{}".format(os.path.splitext(label_path)[0], color_map_ext)

        # update palette and the content of colormap
        logging.info("Get colormap ...")
        for label in labels:
            color = get_random_color()                                          # get random color
            palette.append(color)                                               # get palette's color list
            content.append('{label}: {color}'.format(label=label, color=tuple(color)))  # setup content
        
        # write map table into colormap
        logging.info("Write colormap into `{}`".format(output_palette_path))
        with open(output_palette_path, 'w') as f:
            f.write('\n'.join(content))
    
    return palette
# ----------------------------------------------------------------------------------------------------------------
# Get text size for the right
def get_text_size(text):
    return cv2.getTextSize(text, FONT, FONT_SIZE, THICK)[0]
# ----------------------------------------------------------------------------------------------------------------
# Give text color and return the white or black background
def get_bg_color(color, thres=30):
    color = np.array(color)/255.0
    (h, l, s) = colorsys.rgb_to_hls(color[2], color[1], color[0])
    return (0,0,0) if (l*100)>thres else (255, 255, 255)    # 
# ----------------------------------------------------------------------------------------------------------------
# Put text with background on frame
def put_text(frame, text, position, fg_color=None, bg_color=None):
    fg_color = fg_color if fg_color else (0,0,0)
    bg_color = bg_color if bg_color else (255,255,255)
    frame_bg = cv2.putText(frame, text, position, FONT, FONT_SIZE, get_bg_color(fg_color) , THICK + 3,  cv2.LINE_AA)
    frame_fg = cv2.putText(frame_bg, text, position, FONT, FONT_SIZE, fg_color , THICK, cv2.LINE_AA)
    return frame_fg
# ----------------------------------------------------------------------------------------------------------------
# Draw FPS, default position is top-right corner
def draw_fps(frame, fps, position=None):
    h, w, c = frame.shape
    content = "FPS:{:>3}".format(fps)
    (t_width, t_height) = get_text_size(content)
    (xmin, ymin) = (w-t_width-PADDING, 0+t_height+PADDING)
    return put_text(frame, content, (xmin, ymin))
# ----------------------------------------------------------------------------------------------------------------
# Draw `key`: `value` , default position is top-left corner
def draw_sth(frame, key, value, position=None):
    h, w, c = frame.shape
    content = "{}:{}".format(key, value)
    (t_width, t_height) = get_text_size(content)
    (xmin, ymin) = (0+PADDING, 0+t_height+PADDING)
    return put_text(frame, content, (xmin, ymin))
