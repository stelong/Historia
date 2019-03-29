import numpy as np
from scipy.stats import zscore

def diff(l1, l2):
	return list(set(l1) - set(l2))

def inters(l1, l2):
	return list(set(l1) & set(l2))

def inters_many(L):
	return list(set.intersection(*map(lambda x: set(x), L)))

def union_many(L):
	return list(set.union(*map(lambda x: set(x), L)))

def restrict_kth_comp(data, k, ib, ub):
	l = []
	for i in range(data.shape[0]):
		if np.where(data[i, k] > ib)[0].shape[0] and np.where(data[i, k] < ub)[0].shape[0]:
			l.append(i)
	return l

def find_start_seq(index, feat_dim):
	i = 0
	while i < len(index):
		if index[i:8+i] == list(range(feat_dim)):
			return i
		else:
			i += 1
	return

def whereq_whernot(X, SX):
	feat_dim = X.shape[1]	
	l = []
	for i in range(SX.shape[0]):
		index = np.where(X == SX[i, :])
		if len(list(index[1])) > 8:
			l.append(index[0][find_start_seq(list(index[1]), feat_dim)])
		else:
			l.append(index[0][0])
	nl = diff(range(X.shape[0]), l)
	nl.sort()
	return l, nl

def filter_zscore(X):
	samp_dim = X.shape[0]
	feat_dim = X.shape[1]
	L = []
	thre = 3.0
	for j in range(feat_dim):
		z = np.abs(zscore(X[:, j]))
		L.append(list(np.where(z > 3)[0]))
	l = union_many(L)
	l.sort()
	return l
