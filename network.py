import tensorflow as tf
import numpy as np
import os
import random
import scipy as sp

DATA_DIR = "Data/"
TRAIN_DIR = DATA_DIR + "Train/"
EVAL_DIR = DATA_DIR + "Eval/"
TRAIN_KEY_DIR = TRAIN_DIR + "Key/"
TRAIN_NONKEY_DIR = TRAIN_DIR + "Non-Key/"
EVAL_KEY_DIR = EVAL_DIR + "Key/"
EVAL_NONKEY_DIR = EVAL_DIR + "Non-Key/"
MODEL_DIR = "Model/"
EXPORT_DIR = "Export/"

IMAGE_HEIGHT = 90
IMAGE_WIDTH = 160
IMAGE_SIZE = IMAGE_HEIGHT*IMAGE_WIDTH

FILTER_HEIGHT = 3
FILTER_WIDTH = 3

IN_CHANNELS = 3
OUT_CHANNELS = 1

MANIPULATIONS = 10

STRIDE = [1,1,1,1]

def input_fn(train=True, batch_size=6, num_epochs=1, shuffle=False, seed=42):
	positive_example_filenames = os.listdir(TRAIN_KEY_DIR) if train else os.listdir(EVAL_KEY_DIR)
	print("Positive files: ")
	print(positive_example_filenames)
	negative_example_filenames = os.listdir(TRAIN_NONKEY_DIR) if train else os.listdir(EVAL_NONKEY_DIR)
	print("Negative files: ")
	print(negative_example_filenames)

	positive_example_filenames = [x for x in positive_example_filenames if x != '.DS_Store']
	negative_example_filenames = [x for x in negative_example_filenames if x != '.DS_Store']

	all_labels = tf.concat([tf.cast(tf.ones([(1+MANIPULATIONS)*len(positive_example_filenames)]), tf.int32),
		tf.cast(tf.zeros((1+MANIPULATIONS)*len(negative_example_filenames)), tf.int32)], 0)

	all_images = []

	for filename in positive_example_filenames + negative_example_filenames:
		if filename in positive_example_filenames:
			curr_dir = TRAIN_KEY_DIR if train else EVAL_KEY_DIR
		else:
			curr_dir = TRAIN_NONKEY_DIR if train else EVAL_NONKEY_DIR
		image = tf.convert_to_tensor(sp.ndimage.imread(curr_dir+filename))
		image = tf.cast(image, tf.int32)
		image = tf.cast(image, tf.float32)
		resized_image = tf.image.resize_images(image, [IMAGE_HEIGHT, IMAGE_WIDTH])
		manipulated_images = [ tf.constant(0) for i in range(MANIPULATIONS) ]
		for i in range(MANIPULATIONS):
			r_num = random.randint(1,3)
			if r_num == 1:
				manipulated_images[i] = tf.image.random_flip_up_down(resized_image)
			elif r_num == 2:
				manipulated_images[i] = tf.image.random_flip_left_right(resized_image)
			# Transpose screws up dimensions
			# elif r_num == 3:
			# 	manipulated_images[i] = tf.image.transpose_image(resized_image)
			else:
				num_rots = random.randint(0,1)
				manipulated_images[i] = tf.image.rot90(resized_image, k=num_rots*2)
		all_images += [resized_image] + manipulated_images

	images, labels = tf.train.slice_input_producer( [all_images, all_labels], 
                                                num_epochs=num_epochs,
                                                shuffle=shuffle, seed=seed,
                                                capacity=32,
                                              )

	feature_cols = dict(images=images, labels=labels)
	batched_cols = tf.train.batch(feature_cols, batch_size)

	batched_labels = batched_cols.pop('labels')

	return batched_cols, batched_labels



def model_fn(features, labels, mode, params):

	# Input layer comes from features, which come from input_fn
	# input_layer = tf.cast(features["x"], tf.float32)
	image = tf.reshape(features['images'], [-1,IMAGE_HEIGHT,IMAGE_WIDTH,IN_CHANNELS])

	name = "Name"
	kernel = tf.get_variable(
			name+"_filter",
			[FILTER_HEIGHT,FILTER_WIDTH,IN_CHANNELS,OUT_CHANNELS],
			initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32)
			)

	conv = tf.nn.conv2d(image, kernel, STRIDE, padding='SAME')
	bias = tf.get_variable(name+"_biases", [OUT_CHANNELS], initializer=tf.constant_initializer(0.0))
	pre_activation = tf.nn.bias_add(conv, bias)
	norm = tf.layers.batch_normalization(inputs=pre_activation, training=True)
	norm = tf.reshape(norm, [-1, IMAGE_SIZE*OUT_CHANNELS])
	output_layer = tf.layers.dense(inputs=norm, units=1, activation=tf.nn.sigmoid)

	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(
			mode=mode,
			predictions=output_layer,
			export_outputs={
				tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(
					{"prediction": output_layer}
					)
				}
			)

	# TODO: add l2 regularization
	loss = tf.reduce_mean(
		tf.nn.sigmoid_cross_entropy_with_logits(
			logits=output_layer, labels=tf.cast(tf.reshape(labels, [-1, 1]), tf.float32))
	)

	eval_metric_ops = {
		'accuracy': tf.metrics.accuracy(labels=labels, predictions=tf.round(output_layer))
	}

	# global_step = tf.Variable(0, trainable=False)
	# TODO: Match the specs in the paper about learning rate decay
	learning_rate = tf.train.exponential_decay(0.01, tf.train.get_global_step(),
									   10000, 0.96, staircase=True)

	optimizer = tf.train.MomentumOptimizer(
		learning_rate=learning_rate,
		momentum=0.9)

	train_op = optimizer.minimize(
		loss=loss, global_step=tf.train.get_global_step())

	return tf.estimator.EstimatorSpec(mode, output_layer, loss, train_op, eval_metric_ops,
		export_outputs={
				tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(
					{"prediction": output_layer}
					)
				}
			)

def export(estimator):
	serving_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(
			features={'images': tf.placeholder(tf.float32, shape=[1, IMAGE_SIZE*IN_CHANNELS])})
	estimator.export_savedmodel(EXPORT_DIR, serving_fn)

def build_estimator():
	return tf.estimator.Estimator(model_dir=MODEL_DIR,model_fn=model_fn, params={})

def main():
	estimator = build_estimator()
	estimator.train(input_fn=lambda: input_fn(), steps=10)
	print(estimator.evaluate(input_fn=lambda: input_fn(train=False), steps=1))
	export(estimator)

if __name__=='__main__':
	main()

