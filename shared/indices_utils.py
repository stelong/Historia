import numpy as np

def inters_pair(l1, l2):
	return list(set(l1) & set(l2))

def inters_many(L):
	n = len(L)
	if n == 2:
		return inters_pair(L[0], L[1])
	if n > 2:
		if n % 2:
			l_end = L.pop()
			return inters_pair(inters_many(L), l_end)
		else:
			nc = int(n/2)
			for i in range(nc):
				aux = inters_pair(L[-2], L[-1])
				L.insert(0, aux)
				L.pop(-1)
				L.pop(-1)
			return inters_many(L)

def restrict_kth_comp(data, k, ib, ub):
	l = []
	for i in range(data.shape[0]):
		if np.where(data[i, k] > ib)[0].shape[0] and np.where(data[i, k] < ub)[0].shape[0]:
			l.append(i)
	return l