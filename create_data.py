with open('Data/train.csv', 'w') as f:
	f.write('form_factor, area, perimeter, label\n')
	for i in range(10):
		f.write('10,10,10,1\n')
	for i in range(10):
		f.write('10,10,10,2\n')
	for i in range(10):
		f.write('10,10,10,3\n')

with open('Data/eval.csv', 'w') as f:
	f.write('form_factor, area, perimeter, label\n')
	for i in range(10):
		f.write('10,10,10,1\n')
	for i in range(10):
		f.write('10,10,10,2\n')
	for i in range(10):
		f.write('10,10,10,3\n')