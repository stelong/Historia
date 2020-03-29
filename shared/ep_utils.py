import numpy as np

def init_header(rat, n_stims, filename):
	with open('Historia/shared/headers/' + rat + '/MeshFlatBase.in', 'r') as f:
		with open(filename + '.in', 'w') as fh:
			for line in f:
				fh.write(line)
			fh.write('\n')
			fh.write('stims: {}'.format(n_stims))
			fh.write('\n')
	return

def append_to_header(f, stim, ca):
	f.write('ca{} 166 1 '.format(stim))
	np.savetxt(f, ca.reshape(1, -1), fmt='%.6f')
	return