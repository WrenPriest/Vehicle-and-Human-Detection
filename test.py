# Imports

from numpy.lib.utils import source
from helper_functions.object_detection import detect
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
import numpy as np
import csv
import time
from packaging import version
from tqdm import tqdm


from collections import defaultdict
from io import StringIO
from PIL import Image

# Object detection imports
from utils import label_map_util
from utils import visualization_utils as vis_util


# input video
source_video = 'input_video.mp4'
cap = cv2.VideoCapture(source_video)

# Variables
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fps = int(cap.get(cv2.CAP_PROP_FPS))
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))



#Getting and Downloading Model
MODEL_NAME = 'ssd_mobilenet_v1_coco_2018_01_28'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = \
    'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')


NUM_CLASSES = 90


with tf.io.gfile.GFile(PATH_TO_CKPT, "rb") as f:
  graph_def = tf.compat.v1.GraphDef()
  loaded = graph_def.ParseFromString(f.read())

outputs = (
  'num_detections:0',
  'detection_classes:0',
  'detection_scores:0',
  'detection_boxes:0',
)




def wrap_graph(graph_def, inputs, outputs, print_graph=False):
  wrapped = tf.compat.v1.wrap_function(
    lambda: tf.compat.v1.import_graph_def(graph_def, name=""), [])

  return wrapped.prune(
    tf.nest.map_structure(wrapped.graph.as_graph_element, inputs),
    tf.nest.map_structure(wrapped.graph.as_graph_element, outputs))
    
model = wrap_graph(graph_def=graph_def,
                   inputs=["image_tensor:0"],
                   outputs=outputs)

# #output video
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# output_movie = cv2.VideoWriter(source_video.split(".")[0]+'_output.avi', fourcc, fps, (width, height))







# ret = True
# while ret is True:
#     ret, frame = source_video.read()
#     source_video.release()
#     if not ret:
#         raise Exception(f"Problem reading frame {i} from video")
#     # input_image will be edited while frame is used as a static variable
#     input_image = frame
#     # convert to tensor for model
#     tensor = tf.convert_to_tensor([frame], dtype=tf.uint8)
#     # run tensor frame through model
#     detections = model(tensor)

# detection_graph = tf.Graph()
# with detection_graph.as_default():
#     od_graph_def = tf.GraphDef()
#     with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
#     #od_graph_def = tf.compat.v1.GraphDef() # use this line to run it with TensorFlow version 2.x
#     #with tf.compat.v2.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid: # use this line to run it with TensorFlow version 2.x
#         serialized_graph = fid.read()
#         od_graph_def.ParseFromString(serialized_graph)
#         tf.import_graph_def(od_graph_def, name='')


# # Loading label map
# # Label maps map indices to category names, so that when our convolution network predicts 5, we know that this corresponds to airplane. Here I use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
# label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
# categories = label_map_util.convert_label_map_to_categories(label_map,
#         max_num_classes=NUM_CLASSES, use_display_name=True)
# category_index = label_map_util.create_category_index(categories)

# # Helper code
# def load_image_into_numpy_array(image):
#     (im_width, im_height) = image.size
#     return np.array(image.getdata()).reshape((im_height, im_width,
#             3)).astype(np.uint8)


# #Main
# total_passed_vehicle = 0
# speed = 'waiting...'
# direction = 'waiting...'
# size = 'waiting...'
# color = 'waiting...'


# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# output_movie = cv2.VideoWriter(source_video.split(".")[0]+'_output.avi', fourcc, fps, (width, height))













# with detection_graph.as_default():
#     with tf.Session(graph=detection_graph) as sess:
#     #with tf.compat.v1.Session(graph=detection_graph) as sess: # use this line to run it with TensorFlow version 2.x

#         # Definite input and output Tensors for detection_graph
#         image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

#         # Each box represents a part of the image where a particular object was detected.
#         detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

#         # Each score rerpesent how level of confidence for each of the objects.
#         # Score is shown on the result image, together with the class label.
#         detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
#         detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
#         num_detections = detection_graph.get_tensor_by_name('num_detections:0')

#         # for all the frames that are extracted from input video

# for i in tqdm(range(num_frames)):
#     (ret, frame) = cap.read()

#     if not ret:
#         print ('end of the video file...')
#         break

#     input_frame = frame

#     # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
#     image_np_expanded = np.expand_dims(input_frame, axis=0)

#     # Actual detection.
#     (boxes, scores, classes, num) = \
#         sess.run([detection_boxes, detection_scores,
#                     detection_classes, num_detections],
#                     feed_dict={image_tensor: image_np_expanded})


# roi_info = {
#     'position': 600,
#     'direction': 'left_right'
# }
# assert roi_info['direction'] in ['top_down', 'bottom_up', 'left_right', 'right_left'], 'Invalid ROI direction !'

# # Visualization of the results of a detection.
# (counter, csv_line) = \
#     detect(
#     cap.get(1),
#     input_frame,
#     np.squeeze(boxes),
#     np.squeeze(classes).astype(np.int32),
#     np.squeeze(scores),
#     category_index,
#     roi_info,
#     use_normalized_coordinates=True,
#     line_thickness=4,
    #)