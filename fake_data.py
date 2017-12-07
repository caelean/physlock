import random

IMAGE_HEIGHT = 160
IMAGE_WIDTH = 90
IMAGE_SIZE = IMAGE_HEIGHT*IMAGE_WIDTH
IN_CHANNELS = 3
NUM_IMAGES=10

def feature_col_names():
	return [str(x) for x in range(IMAGE_SIZE*IN_CHANNELS)] + ['label']

with open('/Users/evanmdoyle/Programming/PervasiveComputing/LabProject/Data/images.csv', 'w') as f:
	f.write(",".join(feature_col_names()) + "\n")
	for i in range(NUM_IMAGES):
		f.write(",".join([str(random.randint(0,255)) for x in range(IMAGE_SIZE*IN_CHANNELS)]) + "," + str(random.randint(0,1)) + "\n")