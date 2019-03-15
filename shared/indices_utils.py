import numpy as np

def diff(l1, l2):
	return list(set(l1) - set(l2))

def inters(l1, l2):
	return list(set(l1) & set(l2))

def inters_many(L):
	return list(set.intersection(*map(lambda x: set(x), L)))

def restrict_kth_comp(data, k, ib, ub):
	l = []
	for i in range(data.shape[0]):
		if np.where(data[i, k] > ib)[0].shape[0] and np.where(data[i, k] < ub)[0].shape[0]:
			l.append(i)
	return l

def whereq_whernot(M, SM):
	l = []
	i = 0
	while i < SM.shape[0]:
		for j in range(M.shape[0]):
			if not np.sum(SM[i, :] - M[j, :]):
				l.append(j)
				i += 1
				break
	return l, diff(range(M.shape[0]), l)


# def whereq_whernot(M, SM): # PROBABLY A BUG IN NPWHERE
# 	l = []
# 	for i in range(SM.shape[0]):
# 		l.append(np.where(M == SM[i, :])[0][0])
# 	return l, diff(range(M.shape[0]), l)