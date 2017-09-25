import tensorflow as tf
import os

model_dir='home/pi/usbmem/tensorflow/tf_files'
with tf.Graph().as_default():
	with tf.gfile.FastGFile(os.path.join(model_dir, 'retrained_graph.pb'), 'rb') as f:
		graph_def = tf.GraphDef()
	graph_def.ParseFromString(f.read())
	_ = tf.import_graph_def(graph_def, name='')
	
	with tf.Session() as sess:
		create_graph()
		for op in  sess.graph.get_operations():
			print(op.name)
