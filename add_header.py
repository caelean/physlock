IMAGE_HEIGHT = 160
IMAGE_WIDTH = 90
IMAGE_SIZE = IMAGE_HEIGHT*IMAGE_WIDTH

IN_CHANNELS = 3

lines = [str(x) for x in range(IMAGE_SIZE*IN_CHANNELS)] + ['label']
lines = [",".join(lines) + "\n"]

with open('Data/data.csv', 'r') as f:
	lines += f.readlines()

with open('Data/data.csv', 'w') as f:
	f.writelines(lines)