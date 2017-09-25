''' This code was created with reference to several sources: 
Google Inc. David Dao <daviddao@broad.mit.edu>
	https://github.com/tensorflow/models.git
Dat Tran
	https://github.com/datitran/object_detector_app.git
PyImageSearch. Adrian Rosebrock
	http://www.pyimagesearch.com/2015/03/30/accessing-the-raspberry-pi-camera-with-opencv-and-python/ 
'''

# to run this script 
#> python trial1.py --conf conf.json

from ocvtensor.tempimage import TempImage
import argparse
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
import datetime
import dropbos
import json


#filter warnings, load the configuration and initialize the Dropbox client
warnings.filterwarnings("ignore")
conf = json.load(open(args["conf"]))
client = None

# check to see if the Dropbox should be used
if conf["use_dropbox"]:
# connect to dropbox and start the session authorization process
	client = dropbox.Dropbox(conf["dropbox_access_token"])
	print "[SUCCESS] connected to Dropbox"

def create_graph():
	#Load pretrained tensorflow model into memory (detail of model training in training_model.txt)
	with tf.gfile.GFile('usbmem/tensorflow/tf_files/retrained_graph.pb','rb') as f:
   		graph_def = tf.GraphDef()
		g_graph = f.read()
		graph_def.ParseFromString(g_graph)
		tf.import_graph_def(graph_def, name='')

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 16
resolution = [640, 480]
rawCapture = PiRGBArray(camera,size=tuple(resolution))

# allow the camera to warmup, then initialize the average frame, last
print "[INFO] warming up"
time.sleep(0.1)

# Start tensoflow session
create_graph()
with tf.Session() as sess:
	softmax_tensor = sess.graph.get_tensor_by_name('Final:0')
		
	while True:
		print"[INFO] starting background model"
			# capture frames from the camera
		for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
       			 # grab the raw NumPy array representing the image
			image = frame.array
        		timestamp = datetime.datetime.now()
			image = imutils.resize(image, width=500)

			rawCapture.truncate(0)
			rawCapture.seek(0)
			t = TempImage()
			cv2.imwrite(t.path,image)

                       	image_data = tf.gfile.FastGFile(t.path,'rb').read()
                        predictions =(sess.run(softmax_tensor,{'Final:0' : image_data}))
			predictions = np.squeeze(predictions)
 			
			f = tf.gfile.GFile('usbmem/tensorflow/tf_files/retrained_labels.txt','rb') 
        		lines = f.readlines()
           	        labels = [str(w).replace("\n", "") for w in lines]
       			top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
			for label in top_k:
           			LabelClass = labels[label]
            			score = predictions[label]
            			print('%s (score = %.5f)' % (LabelClass, score))

			cv2.putText(frame,str(np.round(score,2))+"%", (20,440), cv2.FONT_HERSHEY_TRIPLEX, 1, (255,255,255))

			# draw the text 
				#ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
			cv2.putText(frame,Class, (20,400), cv2.FONT_HERSHEY_TRIPLEX, 1, (255,255,255))
			cv2.putText(frame, str(np.round(score,2))+"%", (20,400), cv2.FONT_HERSHEY_TRIPLEX, 1, (255,255,255))
			cv2.imshow("Frame", frame)
			
			if conf["use_dropbox"]:
				# write the image to temporary file
				
				t = TempImage()
				cv2.imwrite(t.path, frame)

				# upload the image to Dropbox and cleanup the tempory image
				print("[UPLOAD] {}".format(ts))
				path = "/{base_path}/{timestamp}.jpg".format(base_path=conf["dropbox_base_path"], timestamp=ts)
				client.files_upload(open(t.path, "rb").read(), path)

				path = "/{base_path}/{timestamp}.jpg".format(base_path=conf["dropbox_base_path"], timestamp=ts)
                    dbx.files_upload(open(t.path, "rb").read(), path)
				t.cleanup()

			if cv2.waitKey(25) & 0xFF == ord('q'):
        			cv2.destroyAllWindows()
				sess.close()
       				break
