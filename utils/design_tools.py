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