# Imports
import cv2
import urllib.request
import tarfile
import shutil
import os
import tensorflow as tf
from helper_functions.model import load_model
from helper_functions.object_detection import object_detection
from helper_functions.counting import counting
from helper_functions import drawing

load_model()
total_vehicles_detected = 0

# input video
source_video = 'input_video.mp4'
cap = cv2.VideoCapture(source_video)

# Variables
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fps = int(cap.get(cv2.CAP_PROP_FPS))
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
is_vehicle_detected = False
ROI_line = "vertical"  # vertical or horizontal

# output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output = cv2.VideoWriter(source_video.split(".")[0] + '_output.mp4', fourcc, fps, (width, height))

while cap.isOpened():
    ret, current_frame = cap.read()
    if ret is True:
        # object detection helper function
        (num_detect, classes, boxes, box_centers) = object_detection(current_frame, ROI_line)

        # counting helper function
        (total_vehicles_detected, is_vehicle_detected) = counting(box_centers, width)

        # drawing helper functions
        drawing.draw_roi(current_frame, width, height, is_vehicle_detected, ROI_line)
        drawing.draw_detection_boxes(boxes, current_frame)
        drawing.draw_counter(current_frame, total_vehicles_detected)

        output.write(current_frame)
        cv2.imshow('framwdite', current_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        else:
            break
