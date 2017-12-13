import tensorflow as tf
import numpy as np
import pandas as pd
import os
import random

DATA_DIR = "Data/"
TRAIN_FILE = DATA_DIR + "train.csv"
EVAL_FILE = DATA_DIR + "eval.csv"
MODEL_DIR = "Model/"
EXPORT_DIR = "Export/"

def feature_col_names():
	return ['form_factor', 'area', 'perimeter']

def decode_label(label):
	if str(label) == '1':
		return np.array([0,0,1])
	elif str(label) == '2':
		return np.array([0,1,0])
	else:
		return np.array([1,0,0])

def input_fn(data_file, num_epochs, batch_size=32, shuffle=False, num_threads=1):
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

		labels = dataset.label.apply(lambda x: decode_label(x))
		labels=labels.apply(pd.Series)

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
	hidden_layer_1 = tf.layers.dense(inputs=input_layer, units=20, activation=tf.nn.sigmoid)
	hidden_layer_2 = tf.layers.dense(inputs=hidden_layer_1, units=10, activation=tf.nn.sigmoid)
	hidden_layer_3 = tf.layers.dense(inputs=hidden_layer_2, units=5, activation=tf.nn.sigmoid)
	output_layer = tf.layers.dense(inputs=hidden_layer_3, units=3, activation=tf.nn.sigmoid)

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

	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=labels))

	eval_metric_ops = {
		'accuracy': tf.metrics.accuracy(tf.argmax(labels, 1), tf.argmax(output_layer, 1))
	}

	optimizer = tf.train.GradientDescentOptimizer(
		learning_rate=0.1)

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
			features={'x': tf.placeholder(tf.float32, shape=[1, 3])})
	estimator.export_savedmodel(EXPORT_DIR, serving_fn)

def build_estimator():
	return tf.estimator.Estimator(model_dir=MODEL_DIR,model_fn=model_fn, params={})

def main():
	estimator = build_estimator()
	estimator.train(input_fn=input_fn(data_file=TRAIN_FILE, num_epochs=None), steps=20)
	print(estimator.evaluate(input_fn=input_fn(data_file=EVAL_FILE, num_epochs=None), steps=1))
	export(estimator)

if __name__=='__main__':
	main()

