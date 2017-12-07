from PIL import Image
import numpy
import os
import sys

'''
	'Python JPGtoCSV.py -t' : process training data in /images
	'Python JPGtoCSV.py -j' : process junk data in /images
	'Python JPGtoCSV.py -v' : process image to be verified in /test
'''

indir = 'images/'
fname = 'csv/data.csv'
# process junk data
label = '0'
# process training data
if sys.argv[1] == '-t':
	label = '1'
# verify image
elif sys.argv[1] == '-v':
	indir = 'test/'
	fname = 'csv/verify.csv'

file = open(fname, 'w')
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
			ans += label + '\n'
			file.write(ans)
file.close()
