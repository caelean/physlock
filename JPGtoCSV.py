from PIL import Image
import numpy
import os

file = open('Data/data.csv', 'w')
indir = './images/'
for _, __, filenames in os.walk(indir):
	for f in filenames:
		if f[-3:] == 'jpg':
			ans = ''
			img = Image.open(indir + f)
			img_array = numpy.array(img)
			for row in img_array:
				for column in row:
					for value in column:
						if value < 10:
							ans += '00'
						elif value < 100:
							ans += '0'
						ans += str(value) + ','
			ans += '1\n'
			file.write(ans)
file.close()
