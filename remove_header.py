lines = []
with open('Data/data.csv', 'r') as f:
	lines += f.readlines()

with open('Data/data.csv', 'w') as f:
	f.writelines(lines[1:])