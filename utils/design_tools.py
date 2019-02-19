import numpy as np
from scipy.stats import uniform

def lhd(p0, E, n_samp):
	pmin = pmax = p0
	n_par = len(p0)
	pmin = [pmin[i] - 0.01*E[i, 0]*pmin[i] for i in range(n_par)]
	pmax = [pmax[i] + 0.01*E[i, 1]*pmax[i] for i in range(n_par)]

	D = np.zeros(shape=(n_samp, n_par))
	H = np.zeros(shape=(n_samp, n_par))
	dp = 1./n_samp
	for j in range(n_par):
		for i in range(n_samp):
			a = i*dp
			D[i, j] = uniform.rvs(loc=a, scale=dp, size=1)

		D[:, j] = np.random.permutation(D[:, j])
		H[:, j] = pmin[j] + (pmax[j] - pmin[j])*D[:, j]

	return H

def split_dataset(X, name_out):
	T = [M for M in np.vsplit(X, 4)]
	for i in range(4):
		with open(name_out + '_' + str(i+1) + '.txt', 'w') as f:
			np.savetxt(f, T[i], fmt='%f')
	return

def putlab(X, name_out):
	n_samp = X.shape[0]
	n_par = X.shape[1]
	labels = ['-p', '-ap', '-z', '-c1', '-ca50', '-kxb', '-koff', '-Tref']
	with open(name_out + '.txt', 'w') as f:
		for i in range(n_samp):
			for j in range(n_par):
				f.write('{} {:g} '.format(labels[j], X[i, j]))
			f.write('\n')
	return

def divide_et_impera(name_in):
	with open(name_in + '.txt', 'r') as f:
		lines = f.readlines()
		for i in range(4):
			with open(name_in + '_' + str(i+1) + '.txt', 'w') as fi:
				for c, line in enumerate(lines):
					if c // 200 == i:
						fi.write(line)

def append_to_header(rat, ca_traces, number):
	with open('utils/headers/' + rat + '/MeshFlatBase.in', 'r') as f:
		with open('MeshFlatBase_' + str(number) + '.in', 'w') as fh:
			for line in f:
				fh.write(line)
			with open(ca_traces + '_' + str(number) + '.txt', 'r') as fc:
				for line in fc:
					fh.write(line)