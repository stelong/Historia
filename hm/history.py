from Historia.emulator import gp
from Historia.shared import indices_utils as desu
import numpy as np

class Wave:
	def __init__(self, emulator):
		self.emulator = emulator

	def compute_impl(self, X, maxno, mean, var):
		samp_dim = X.shape[0]
		I = np.zeros((samp_dim,), dtype=float)
		for i in range(samp_dim):
			In = []
			for j, emul in enumerate(self.emulator):
				gp_mean, gp_std = emul.predict(X[i, :].reshape(1, -1))
				gp_var = np.power(gp_std, 2.0)
				In.append( np.sqrt((np.power(gp_mean-mean[j], 2.0))/(gp_var+var[j])) )
			In.sort()
			I[i] = In[-maxno]
		return I

	def find_regions(self, X_test, maxno, cutoff, mean, var):
		I = self.compute_impl(X_test, maxno, mean, var)
		l = np.where(I < cutoff)[0]
		nl = desu.diff(range(X_test.shape[0]), l)
		X_nimp = X_test[l]
		X_imp = X_test[nl]
		return I, X_nimp, X_imp