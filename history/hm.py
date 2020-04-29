import diversipy as dp
from Historia.share import concurrent_utils as cucu
from Historia.shared import design_utils as desu
from Historia.shared import indices_utils as indu
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

CHUNK_SIZE = 10000

def fastpredict(emulator, X):
	M = np.zeros((CHUNK_SIZE, len(emulator)), dtype=float)
	V = np.zeros((CHUNK_SIZE, len(emulator)), dtype=float)
	for j, emul in enumerate(emulator):
		mean, std = emul.posterior(X)
		var = np.power(std, 2.0)
		M[:, j] = mean
		V[:, j] = var
	return M, V

def add_points(NROY, n_points, prev_wave): # FIX ME: make this a class method using an auxiliary Wave object to enrich NROY
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
		self.NIMP_l = None
		self.IMP = None
		self.IMP_l = None

	def compute_impl(self, X):
		samp_dim = X.shape[0]
		features_dim = len(self.emulator)
		n_chunks = int(samp_dim/CHUNK_SIZE)

		inputs = {i: (self.emulator, x) for i, x in enumerate(np.vsplit(X, n_chunks))}
		outputs = cucu.execute_task_in_parallel(fastpredict, inputs)

		M = np.zeros((0, features_dim), dtype=float)
		V = np.zeros((0, features_dim), dtype=float)
		for i in range(n_chunks):
			M = np.vstack((M, outputs[i][0]))
			V = np.vstack((V, outputs[i][1]))

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
		self.NIMP_l = l
		self.IMP = X_imp
		self.IMP_l = nl

	def print_stats(self):
		imp = self.IMP.shape[0]
		nimp = self.NIMP.shape[0]
		tests = imp+nimp
		perc = 100*nimp/imp
		stats = pd.DataFrame(index=['TESTS', 'IMP', 'NIMP', 'PERC'], columns=['#POINTS'], data=[tests, imp, nimp, '{:.4f} %'.format(perc)])
		print(stats)
		return

	def reconstruct_tests(self):
		X_test = np.zeros((self.NIMP.shape[0]+self.IMP.shape[0], len(self.emulator)), dtype=float)
		X_test[self.NIMP_l] = self.NIMP
		X_test[self.IMP_l] = self.IMP
		return X_test

	def save(self, filename):
		with open(filename + '.pickle', 'wb') as f:
			pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

	@classmethod
	def load(cls, filename):
		with open(filename + '.pickle', 'rb') as f:
			wave_obj = pickle.load(f)
		return wave_obj

	def plot_Impl(self, xlabels):
		X = self.reconstruct_tests()
		n_features = X.shape[1]
		fig, axes = plt.subplots(nrows=n_features, ncols=n_features, sharex='col', sharey='row', figsize=(1.8*5.809078888889, 1.8*11.69/3))
		for i, axis in enumerate(axes.flatten()):
			mm = [np.min(X[:, i % n_features]), np.max(X[:, i % n_features]), np.min(X[:, i // n_features]), np.max(X[:, i // n_features])]
			im = axis.hexbin(X[:, i % n_features], X[:, i // n_features], C=self.I, reduce_C_function=np.min, gridsize=12, extent=mm, cmap='jet', vmin=1.0, vmax=self.cutoff)
			if i // n_features == n_features - 1:
				axis.set_xlabel(xlabels[i % n_features], fontsize=12)
			if i % n_features == 0:
				axis.set_ylabel(xlabels[i // n_features], fontsize=12)
		for i in range(n_features):
			for j in range(n_features):
				if i <= j:
					axes[i, j].set_visible(False)

		fig.subplots_adjust(right=0.9)
		cbar_ax = fig.add_axes([0.83, 0.108, 0.01, 0.68])
		cbar = fig.colorbar(im, cax=cbar_ax)
		cbar.set_label('Implausibility measure', size=12)
		plt.show()
		return