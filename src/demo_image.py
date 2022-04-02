import cv2, os
import sys
sys.path.insert(0, 'FaceBoxesV2')
sys.path.insert(0, '..')
import numpy as np
import pickle
import importlib
from math import floor
from opensource_module.PIPNet.FaceBoxesV2.faceboxes_detector import *
import time
from pipline import FacePipeline


image_file = './images/2022-04-01_213743.jpg'

def demo_image(image_file):

    face_pipeline = FacePipeline()

    image = cv2.imread(image_file)
    image_height, image_width, _ = image.shape

    results = face_pipeline.pipeline(image, draw_immediate=True)

    cv2.imshow('1', image)
    cv2.waitKey(0)
        

demo_image(image_file)
