# Imports
import cv2
from helper_functions.model import load_model
from helper_functions.object_detection import object_detection
from helper_functions.counting import counting
from helper_functions import drawing

model, labels = load_model()

# input video
source_video = 'human-test.mp4'
cap = cv2.VideoCapture(source_video)

# Variables
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fps = int(cap.get(cv2.CAP_PROP_FPS))
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
ticker = 0
counter = 0
ROI_line = "vertical"  # vertical or horizontal

# output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# output = cv2.VideoWriter(source_video.split(".")[0] + '_output.mp4', fourcc, fps, (width, height))
output = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

while cap.isOpened():
    ret, current_frame = cap.read()
    if ret is True:
        ticker += 1
        # progress ticker
        print(f"Processing frame {ticker} of {num_frames}")
        # object detection helper function
        (num_detect, classes, boxes, box_centers) = object_detection(current_frame, model, ROI_line, height, width)

        # counting helper function
        (counter, is_vehicle_detected) = counting(box_centers, classes, width, counter)

        # drawing helper functions
        drawing.draw_roi(current_frame, width, height, is_vehicle_detected, ROI_line)
        drawing.draw_detection_boxes(current_frame, num_detect, boxes, labels, classes)
        drawing.draw_counter(current_frame, counter)

        output.write(current_frame)
        # cv2.imshow('framwdite', current_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        cap.release()
        output.release()
        break
