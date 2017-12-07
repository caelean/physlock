import tensorflow as tf
import pandas as pd
import numpy as np

DATA_FILE = "Data/data.csv"
EVAL_FILE = DATA_FILE
MODEL_DIR = "Model/"
EXPORT_DIR = "Export/"

#TODO: Fill in correct image size
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 90
IMAGE_SIZE = IMAGE_HEIGHT*IMAGE_WIDTH

FILTER_HEIGHT = 3
FILTER_WIDTH = 3

IN_CHANNELS = 3
OUT_CHANNELS = 1

STRIDE = [1,1,1,1]

def feature_col_names():
	return [str(x) for x in range(IMAGE_SIZE*IN_CHANNELS)]

def input_fn(data_file, num_epochs, batch_size=10, shuffle=False, num_threads=1):
	feature_cols = feature_col_names()
	target_cols = ['label']
	dataset = pd.read_csv(
		tf.gfile.Open(data_file),
		header=0,
		usecols=feature_cols + target_cols,
		skipinitialspace=True,
		engine="python")
	# Drop NaN entries
	dataset.dropna(how="any", axis=0)

	labels = dataset.label
	dataset.pop('label')

	return tf.estimator.inputs.numpy_input_fn(
		x={"x": np.array(dataset)},
		y=np.array(labels),
		batch_size=batch_size,
		num_epochs=num_epochs,
		shuffle=shuffle,
		num_threads=num_threads)

def model_fn(features, labels, mode, params):

	# Input layer comes from features, which come from input_fn
	input_layer = tf.cast(features["x"], tf.float32)
	image = tf.reshape(input_layer, [-1,IMAGE_HEIGHT,IMAGE_WIDTH,IN_CHANNELS])

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
		tf.nn.softmax_cross_entropy_with_logits(logits=tf.transpose(output_layer), labels=labels)
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
			features={'x': tf.placeholder(tf.float32, shape=[1, IMAGE_SIZE*IN_CHANNELS])})
	estimator.export_savedmodel(EXPORT_DIR, serving_fn)

def build_estimator():
	return tf.estimator.Estimator(model_dir=MODEL_DIR,model_fn=model_fn, params={})

def main():
	estimator = build_estimator()
	estimator.train(input_fn=input_fn(DATA_FILE, num_epochs=None, shuffle=True), steps=2)
	print(estimator.evaluate(input_fn=input_fn(EVAL_FILE, num_epochs=None, shuffle=False), steps=1))
	export(estimator)

if __name__=='__main__':
	main()

