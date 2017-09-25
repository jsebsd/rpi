import argparse
import os.path
import re
import sys
import cv2
from time import sleep
import numpy as np
import tensorflow as tf
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import imutils

Model_dir = 'usbmem/tensorflow/tf_files/retrained_graph.pb'

detection_graph = tf.Graph()
with detection_graph.as_default():
	graph_def = tf.GraphDef()
	with tf.gfile.GFile(Model_dir,'rb') as f:
   		g_graph = f.read()
		graph_def.ParseFromString(g_graph)
		tf.import_graph_def(graph_def, name='')

# Variables declarations
frame_count=0
score=0
start = time.time()
pred=0
last=0

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
		# capture frames from the camera
			for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
       			 # grab the raw NumPy array representing the image
        			image = frame.array
        			image = imutils.resize(image, width=500)
				rawCapture.truncate(0)
                
			frame_count+=1
                # Only run every 5 frames
                	if frame_count%5==0:

                        # Save the image as the first layer of inception is a DecodeJpeg
                        	cv2.imwrite("current_frame.jpg",frame)

                        	image_data = tf.gfile.FastGFile("./current_frame.jpg", 'rb').read()
				image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
				# Each box represents a part of the image where a particular object was detected.
      				boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
  			        # Each score represent how level of confidence for each of the objects.
      				# Score is shown on the result image, together with the class label.
      				scores = detection_graph.get_tensor_by_name('detection_scores:0')
      				classes = detection_graph.get_tensor_by_name('detection_classes:0')
      				num_detections = detection_graph.get_tensor_by_name('num_detections:0')
      				# Actual detection.
      				(boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections], feed_dict={image_tensor: image_np_expanded})
  			        # Visualization of the results of a detection.
      				vis_util.visualize_boxes_and_labels_on_image_array(image_np,np.squeeze(boxes),np.squeeze(classes).astype(np.int32), np.squeeze(scores), category_index, use_normalized_coordinates=True, line_thickness=8)
                                

                # Show info during some time
                	if last<40 and frame_count>10:
                        	cv2.putText(frame,human_string, (20,400), cv2.FONT_HERSHEY_TRIPLEX, 1, (255,255,255))
                        	cv2.putText(frame,str(np.round(score,2))+"%", (20,440), cv2.FONT_HERSHEY_TRIPLEX, 1, (255,255,255))

				if frame_count>20:
                        		cv2.putText(frame,"fps: "+str(np.round(fps,2)), (460,460), cv2.FONT_HERSHEY_TRIPLEX, 1, (255,255,255))

                	cv2.imshow("Frame", frame)
                	last+=1


                	# if the 'q' key is pressed, stop the loop
                	if cv2.waitKey(1) & 0xFF == ord("q"):
				break

# cleanup everything
cv2.destroyAllWindows()
sess.close()
print("Done")

