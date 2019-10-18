import diversipy as dp
from Historia.shared import design_utils as desu
from Historia.shared import indices_utils as indu
import numpy as np
import pickle

def add_points(NROY, n_points, prev_wave):
	W = prev_wave
	while NROY.shape[0] < n_points:
		I = desu.get_minmax(NROY)
		SCALE = 0.1*np.array([I[i, 1]-I[i, 0] for i in range(NROY.shape[1])])
		minmax = desu.get_minmax(W.emulator[0].X)
		minlist = minmax[:, 0]
		maxlist = minmax[:, 1]
		temp = np.random.normal(loc=NROY, scale=SCALE)
		while True:
			l = []
			for i in range(temp.shape[0]):
				diff1 = temp[i, :] - minlist
				diff2 = maxlist - temp[i, :]
				if np.sum(np.sign(diff1)) != temp.shape[1] or np.sum(np.sign(diff2)) != temp.shape[1]:
					l.append(i)
			if l:
				temp[l, :] = np.random.normal(loc=NROY[l, :], scale=SCALE)
			else:
				break
		W.find_regions(temp)
		NROY = np.vstack((NROY, W.NIMP))
	SNROY = dp.subset.psa_select(NROY, n_points)
	return SNROY

class Wave:
	def __init__(self, emulator=None, cutoff=None, maxno=None, mean=None, var=None):
		self.emulator = emulator
		self.cutoff = cutoff 
		self.maxno = maxno
		self.mean = mean
		self.var = var
		self.I = None
		self.NIMP = None
		self.IMP = None

	def compute_impl(self, X):
		samp_dim = X.shape[0]
		features_dim = len(self.emulator)
		M = np.zeros((samp_dim, features_dim), dtype=float)
		V = np.zeros((samp_dim, features_dim), dtype=float)
		for j, emul in enumerate(self.emulator):
			gp_mean, gp_std = emul.predict(X, with_std=True)
			gp_var = np.power(gp_std, 2.0)
			M[:, j] = gp_mean
			V[:, j] = gp_var
		I = np.zeros((samp_dim,), dtype=float)
		for i in range(samp_dim):
			In = np.sqrt((np.power(M[i, :] - self.mean, 2.0))/(V[i, :] + self.var))
			I[i] = np.sort(In)[-self.maxno]
		return I

	def find_regions(self, X):
		I = self.compute_impl(X)
		l = np.where(I < self.cutoff)[0]
		nl = indu.diff(range(X.shape[0]), l)
		X_nimp = X[l]
		X_imp = X[nl]
		self.I = I
		self.NIMP = X_nimp
		self.IMP = X_imp

	def save(self, filename):
		with open(filename + '.pickle', 'wb') as f:
			pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

	@classmethod
	def load(cls, filename):
		with open(filename + '.pickle', 'rb') as f:
			wave_obj = pickle.load(f)
		return wave_obj