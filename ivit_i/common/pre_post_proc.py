import cv2
import numpy as np
import logging
import json

def preprocess_fn(image, rs):
	'''
	Image pre-processing.
	Rearranges from BGR to RGB then normalizes to range 0:1
	input arg: path of image file
	return: numpy array
	'''
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = cv2.resize(image, rs, cv2.INTER_LINEAR)
	image = image/255.0
	return image