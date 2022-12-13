import logging, cv2, math, json, os

FONT            = cv2.LINE_AA
FONT_SCALE      = 1
FONT_THICKNESS  = 2

def get_text_size(label) -> tuple:
    """ return width, height in tuple """
    return cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)[0]

def get_distance(pt, pt2):
    pt = ( float(pt[0]), float(pt[1]) )
    pt2 = ( float(pt2[0]), float(pt2[1]) )
    return math.hypot( pt2[0]-pt[0], pt2[1]-pt[1])

def get_framework():
    try:
        import vart
        return "vitis-ai"
    except Exception as e:
        pass
    try:
        import openvino
        return "openvino"
    except Exception as e:
        pass
    try:
        import tensorrt
        return "tensorrt"
    except Exception as e:
        pass

def get_coord_distance(p1 , p2):
    coordinate_distance = math.sqrt( ((int(p1[0])-int(p2[0]))**2)+((int(p1[1])-int(p2[1]))**2) )
    return coordinate_distance

def read_json(path):
    with open(path) as f:
        return json.load(f)

def get_angle_for_cv(pos1, pos2):
    """
    Calculate Vector's Angle for OpenCV Pixel

    (0, 0) . . (X, 0)
    .
    .
    (Y, 0) . . (X, Y)

    Because the pixel position is reversed, Angle will be reversed, too
    """

    dx = pos2[0] - pos1[0]
    dy = pos2[1] - pos1[1]
    theta = int(math.atan2(dy, dx)*180/math.pi) 
    return theta*(-1)
