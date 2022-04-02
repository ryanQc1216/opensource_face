import copy
import sys
sys.path.insert(0, 'FaceBoxesV2')
sys.path.insert(0, '..')
import cv2, os
import sys
import numpy as np
import pickle
import importlib
from math import floor
import time
from pipline import FacePipeline

video_file = './videos/007.avi'

def demo_video(video_file):
    if video_file == 'camera':
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(video_file)
    if cap.isOpened()== False:
        print("Error opening video stream or file")
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    count = 0
    face_pipeline = FacePipeline()

    while(cap.isOpened()):
        ret, frame = cap.read()

        image = copy.deepcopy(frame)
        image_height, image_width, _ = image.shape
        results = face_pipeline.pipeline(image, draw_immediate=True)

        if ret == True:
            cv2.imshow('res', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

demo_video(video_file)
