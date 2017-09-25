import os
import cv2
import time
from time import sleep
import numpy as np
import tensorflow as tf
from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
import sys
from io import StringIO
import imutils
from utils import visualization_utils as vis_util

#Load pretrained tensorflow model into memory (detail of model training in training_model.txt)
detection_graph = tf.Graph()
with detection_graph.as_default():
	graph_def = tf.GraphDef()
	with tf.gfile.GFile('usbmem/tensorflow/tf_files/retrained_graph.pb','rb') as f:
   		g_graph = f.read()
		graph_def.ParseFromString(g_graph)
		tf.import_graph_def(graph_def, name='')

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 16
resolution = [640, 480]
rawCapture = PiRGBArray(camera,size=tuple(resolution))

# allow the camera to warmup
time.sleep(0.1)
print "Starting Background Model"

# Start tensoflow session
with detection_graph.as_default():
	with tf.Session(graph=detection_graph) as sess:
		while True:
			ret, image_np = camera.read()
			# capture frames from the camera
			for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
       			 # grab the raw NumPy array representing the image
				#image = frame.array
        			image = imutils.resize(image, width=500)
				rawCapture.truncate(0)

				image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
					# Each box represents a part of the image where a particular object was detected.
      				boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
  			        # Each score represent how level of confidence for each of the objects.
      				# Score is shown on the result image, together with the class label.
      				scores = detection_graph.get_tensor_by_name('detection_scores:0')
      				classes = detection_graph.get_tensor_by_name('detection_classes:0')
      				num_detections = detection_graph.get_tensor_by_name('num_detections:0')

					# Actual detection.
      				(boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections],
					feed_dict={image_tensor: image_np_expanded})

				   # Visualization of the results of a detection.
      			vis_util.visualize_boxes_and_labels_on_image_array(
					image_np,
					np.squeeze(boxes),
					np.squeeze(classes).astype(np.int32),
					np.squeeze(scores), category_index,
					use_normalized_coordinates=True,
					line_thickness=8)

      			cv2.imshow('object detection', cv2.resize(image_np, (800,600)))
      			if cv2.waitKey(25) & 0xFF == ord('q'):
        			cv2.destroyAllWindows()
				sess.close()
        			break
