import tensorflow as tf
import numpy as np
import pandas as pd
import os
import random

DATA_DIR = "Data/"
TRAIN_FILE = DATA_DIR + "train.csv"
EVAL_FILE = DATA_DIR + "eval.csv"
PREDICT_FILE = DATA_DIR + "predict.csv"
MODEL_DIR = "Model/"
EXPORT_DIR = "Export/"

# GET COLUMN NAMES IN CSV FILE
def feature_col_names():
	return ['form_factor', 'area', 'perimeter']

# CONVERT FROM FLOAT LABELS TO ONE-HOT VECTORS
def decode_label(label):
	if str(label) == '1.0':
		return np.array([0,0,1])
	elif str(label) == '2.0':
		return np.array([0,1,0])
	else:
		return np.array([1,0,0])

# INPUT FUNCTION FOR PREDICTION
# THIS VERSION ONLY REQUIRES FEATURES, NO LABELS
def predict_input_fn(data_file, num_epochs=None, batch_size=6, shuffle=False, num_threads=1):
	feature_cols = feature_col_names()
	dataset = pd.read_csv(
		tf.gfile.Open(data_file),
		header=0,
		usecols=feature_cols,
		skipinitialspace=True,
		engine="python")
	# Drop NaN entries
	dataset.dropna(how="any", axis=0)
	dataset=(dataset-dataset.mean())/dataset.std()

	return tf.estimator.inputs.numpy_input_fn(
			x={"x": np.array(dataset)},
			batch_size=batch_size,
			num_epochs=num_epochs,
			shuffle=shuffle,
			num_threads=num_threads)

# STANDARD INPUT FUNCTION
# GETS FEATURES AND LABELS FROM CSV
def input_fn(data_file, num_epochs=None, batch_size=6, shuffle=False, num_threads=1):
	feature_cols = feature_col_names()
	target_cols = ['label']
	dataset = pd.read_csv(
		tf.gfile.Open(data_file),
		header=0,
		usecols=feature_cols + target_cols,
		skipinitialspace=True,
		engine="python")

	# DROP NaN ENTRIES
	dataset.dropna(how="any", axis=0)

	# DECODE AND EXTRACT LABELS FROM DATASET
	labels = dataset.label.apply(lambda x: decode_label(x))
	labels = labels.apply(pd.Series)
	dataset.pop('label')

	# MEAN NORMALIZATION
	dataset=(dataset-dataset.mean())/dataset.std()

	return tf.estimator.inputs.numpy_input_fn(
		x={"x": np.array(dataset)},
		y=np.array(labels),
		batch_size=batch_size,
		num_epochs=num_epochs,
		shuffle=shuffle,
		num_threads=num_threads)


# DEFINE THE STRUCTURE OF OUR NEURAL NETWORK
def model_fn(features, labels, mode, params):

	# CONVERT FEATURES TO FLOAT
	input_layer = tf.cast(features["x"], tf.float32)
	# HIDDEN LAYER OF 10 NODES WITH SIGMOID ACTIVATION
	hidden_layer = tf.layers.dense(inputs=input_layer, units=10, activation=tf.nn.sigmoid)
	# RAW OUTPUT LOGIT LAYER WITH 3 NODES (FOR 3 CLASSES)
	output_layer = tf.layers.dense(inputs=hidden_layer_1, units=3)

	# IF PREDICTING, DONT NEED TO EVALUATE LOSS AND OPTIMIZE
	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(
			mode=mode,
			predictions=tf.argmax(output_layer, axis=1),
			export_outputs={
				tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(
					{"prediction": tf.argmax(output_layer, axis=1)}
					)
				}
			)

	# USE CROSS ENTROPY LOSS BETWEEN OUTPUT LOGITS AND ONE-HOT LABELS
	loss = tf.losses.softmax_cross_entropy(logits=output_layer, onehot_labels=labels)

	# ACCURACY DEFINED BY WHETHER HIGHEST LOGIT IS IN SAME INDEX AS ONE IN ONE-HOT LABEL
	eval_metric_ops = {
		'accuracy': tf.metrics.accuracy(tf.argmax(labels, axis=1), tf.argmax(output_layer, axis=1))
	}

	# USE GRADIENT DESCENT TO UPDATE WEIGHTS TO MINIMIZE LOSS FUNCTION
	optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
	train_op = optimizer.minimize(
		loss=loss, global_step=tf.train.get_global_step())

	# RETURN AN ESTIMATOR MATCHING THESE SPECIFICATIONS
	return tf.estimator.EstimatorSpec(mode, output_layer, loss, train_op, eval_metric_ops,
		export_outputs={
				tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(
					{"prediction": tf.argmax(output_layer, axis=1)}
					)
				}
			)

# FUNCTION WHICH EXPORTS THE CURRENT MODEL VERSION
def export(estimator):
	# DEFINE HOW TO REQUEST PREDICTIONS FROM THE EXPORTED MODEL
	serving_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(
			features={'x': tf.placeholder(tf.float32, shape=[1, 3])})
	# EXPORT THE MODEL TO THE EXPORT_DIR
	estimator.export_savedmodel(EXPORT_DIR, serving_fn)

# BUILD AN ESTIMATOR WITH THE GIVEN MODEL_FN
def build_estimator():
	return tf.estimator.Estimator(model_dir=MODEL_DIR,model_fn=model_fn, params={})

def main():
	estimator = build_estimator()
	# THREE STAGE TRAINING CYCLE
	# 1. TRAIN THE MODEL ON OUR TRAIN_FILE DATA FOR 10,000 STEPS
	estimator.train(input_fn=input_fn(data_file=TRAIN_FILE, num_epochs=None), steps=10000)
	# 2. EVALUATE THE MODEL ACCURACY ON A DIFFERENT SET OF LABELLED DATA AND PRINT ACCURACY
	print(estimator.evaluate(input_fn=input_fn(data_file=EVAL_FILE, num_epochs=None), steps=1))
	# 3. MAKE PREDICTIONS ON DATA NOT PRESENT IN TRAIN_FILE OR EVAL_FILE FOR MANUAL VERIFICATION
	print(list(estimator.predict(input_fn=predict_input_fn(data_file=PREDICT_FILE, num_epochs=1))))

	# EXPORT NEW MODEL VERSION SO IT CAN BE SERVED ON LOCALHOST
	export(estimator)

if __name__=='__main__':
	main()
