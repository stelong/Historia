import numpy as np
from scipy.stats import uniform

def write_txt(X, fmtstr, name_out):
	with open(name_out + '.txt', 'w') as f:
		np.savetxt(f, X, fmt=fmtstr)

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

def lhd_int(I, n_samp):
	n_par = I.shape[0]
	D = np.zeros(shape=(n_samp, n_par))
	H = np.zeros(shape=(n_samp, n_par))
	dp = 1./n_samp
	for j in range(n_par):
		for i in range(n_samp):
			a = i*dp
			D[i, j] = uniform.rvs(loc=a, scale=dp, size=1)

		D[:, j] = np.random.permutation(D[:, j])
		H[:, j] = I[j, 0] + (I[j, 1] - I[j, 0])*D[:, j]

	return H

def split_dataset(X, name_out):
	T = [M for M in np.vsplit(X, 4)]
	for i in range(4):
		write_txt(T[i], name_out + '_' + str(i+1))
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
