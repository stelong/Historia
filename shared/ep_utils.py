def divide_et_impera(name_in):
	with open(name_in + '.txt', 'r') as f:
		lines = f.readlines()
		for i in range(4):
			with open(name_in + '_' + str(i+1) + '.txt', 'w') as fi:
				for c, line in enumerate(lines):
					if c // 200 == i:
						fi.write(line)
	return

def init_header(rat, n_traces, filename):
	with open('Historia/shared/headers/' + rat + '/MeshFlatBase.in', 'r') as f:
		with open(filename + '.in', 'w') as fh:
			for line in f:
				fh.write(line)
			fh.write('\n')
			fh.write('stims: {}'.format(n_traces))
			fh.write('\n')
	return