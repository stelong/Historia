import itertools
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def l(i, x):
	r = np.linspace(0, 1, 4)
	ind = list(set(range(4)) - set([i]))

	num = (x - r[ind[0]]) * (x - r[ind[1]]) * (x - r[ind[2]])
	den = (r[i] - r[ind[0]]) * (r[i] - r[ind[1]]) * (r[i] - r[ind[2]])

	return num/den

def psi(i, x):
	# ind = [0, 1, 2, 3]
	# L = [p for p in itertools.product(ind, repeat=3)][i]

	L = [[0, 0, 0],
		[1, 0, 0],
		[2, 0, 0],
		[3, 0, 0],
		[0, 1, 0],
		[1, 1, 0],
		[2, 1, 0],
		[3, 1, 0],
		[0, 2, 0],
		[1, 2, 0],
		[2, 2, 0],
		[3, 2, 0],
		[0, 3, 0],
		[1, 3, 0],
		[2, 3, 0],
		[3, 3, 0],
		[0, 0, 1],
		[1, 0, 1],
		[2, 0, 1],
		[3, 0, 1],
		[0, 1, 1],
		[1, 1, 1],
		[2, 1, 1],
		[3, 1, 1],
		[0, 2, 1],
		[1, 2, 1],
		[2, 2, 1],
		[3, 2, 1],
		[0, 3, 1],
		[1, 3, 1],
		[2, 3, 1],
		[3, 3, 1],
		[0, 0, 2],
		[1, 0, 2],
		[2, 0, 2],
		[3, 0, 2],
		[0, 1, 2],
		[1, 1, 2],
		[2, 1, 2],
		[3, 1, 2],
		[0, 2, 2],
		[1, 2, 2],
		[2, 2, 2],
		[3, 2, 2],
		[0, 3, 2],
		[1, 3, 2],
		[2, 3, 2],
		[3, 3, 2],
		[0, 0, 3],
		[1, 0, 3],
		[2, 0, 3],
		[3, 0, 3],
		[0, 1, 3],
		[1, 1, 3],
		[2, 1, 3],
		[3, 1, 3],
		[0, 2, 3],
		[1, 2, 3],
		[2, 2, 3],
		[3, 2, 3],
		[0, 3, 3],
		[1, 3, 3],
		[2, 3, 3],
		[3, 3, 3]]

	return l(L[i][0], x[0])*l(L[i][1], x[1])*l(L[i][2], x[2])

def PSI(c, x):
	val = 0
	for i in range(64):
		val += c[i, :]*psi(i, x)
	return val

def T(c, xhat):
	return PSI(c, xhat)

#----------------------------------------

def exnode(file):
	A = []
	with open(file, 'r') as f:
		for _ in range(7):
			next(f)
		for i, line in enumerate(f):
			if i % 2:
				A.append( [float(line.split()[i]) for i in range(3)] )
	A = np.array(A)

	return A

def exelem(file):
	B = []
	with open(file, 'r') as f:
		for _ in range(588):
			next(f)
		for i, line in enumerate(f):
			if i % 2 == 1:
				B.append( [int(line.split()[1:][j]) for j in range(64)] )
	B = np.array(B)

	return B

def exdata_xi(file):
	Ct = []
	Ft = []
	JXt = []
	with open(file, 'r') as f:
		for _ in range(17):
			next(f)
		for i, line in enumerate(f):
			if (i-5) % 4 == 0:
				Ct.append( [float(line.split()[j]) for j in range(3, 6)] )
			elif (i-6) % 4 == 0:
				Ft.append( [float(line.split()[j]) for j in range(3)] )
			elif (i-7) % 4 == 0:
				JXt.append( float(line) )
	Ct = np.array(Ct)
	Ft = np.array(Ft)
	JXt = np.array(JXt).reshape(-1, 1)

	C = [c for c in np.vsplit(Ct, 108)]
	F = [f for f in np.vsplit(Ft, 108)]
	JX = [j for j in np.vsplit(JXt, 108)]

	return C, F, JX

def exdata(file):
	ept = []
	Tat = []
	lambdt = []
	dlambdt = []
	with open(file, 'r') as f:
		for _ in range(22):
			next(f)
		for i, line in enumerate(f):
			if (i-4) % 3 == 0:
				v = [float(line.split()[j]) for j in range(1, 5)]
				ept.append( v[0] )
				Tat.append( v[1] )
				lambdt.append( v[2] )
				dlambdt.append( v[3] )

	ept = np.array(ept).reshape(-1, 1)
	Tat = np.array(Tat).reshape(-1, 1)
	lambdt = np.array(lambdt).reshape(-1, 1)
	dlambdt = np.array(dlambdt).reshape(-1, 1)

	ep = [e for e in np.vsplit(ept, 108)]
	Ta = [t for t in np.vsplit(Tat, 108)]
	lambd = [l for l in np.vsplit(lambdt, 108)]
	dlambd = [dl for dl in np.vsplit(dlambdt, 108)]

	return ep, Ta, lambd, dlambd

def connect(B, n_nodes, n_elements):
	L = []
	for i in range(n_nodes):
		l = []
		for j in range(n_elements):
			if any(B[j, :] == i+1):
				l.append(j+1)
		L.append(l)
	return L

# def Tmap(A, B, C, M, n_elements):
# 	r = np.linspace(0, 1, 4)
# 	X, Y, Z = np.meshgrid(r, r, r)
# 	R = np.hstack(( np.hstack(( X.reshape(64, 1), Y.reshape(64, 1) )), Z.reshape(64, 1) ))

# 	coeff = [70, -140, 90, -20, 1]

# 	rg = np.roots(coeff)
# 	Xg, Yg, Zg = np.meshgrid(rg, rg, rg)
# 	Rg = np.hstack(( np.hstack(( Xg.reshape(64, 1), Yg.reshape(64, 1) )), Zg.reshape(64, 1) ))

# 	elem = 80

# 	W = []
# 	Wg = []
# 	Z = []
# 	CC = C[:64]
# 	for j in range(elem, elem+1):
# 		w = np.zeros((64, 3), dtype=float)
# 		wg = np.zeros((64, 3), dtype=float)
# 		z = np.zeros((64, 3), dtype=float)
# 		for i in range(64):
# 			w[i, :] = T(A[list(B[j, :]-1), :], R[i, :]).reshape(1, -1)
# 			wg[i, :] = T(A[list(B[j, :]-1), :], CC[i, :]).reshape(1, -1)
# 			z[i, :] = T(A[list(B[j, :]-1), :], CC[i, :] + M[j][i, :3]).reshape(1, -1)
# 		W.append(w)
# 		Wg.append(wg)
# 		Z.append(z)

# 	fig = plt.figure()
# 	ax = fig.add_subplot(111, projection='3d')
# 	ax.scatter(W[0][:, 0], W[0][:, 1], W[0][:, 2], c='b')
# 	ax.scatter(Wg[0][:, 0], Wg[0][:, 1], Wg[0][:, 2], c='r')
# 	for i in range(64):
# 		ax.plot([Wg[0][i, 0],Z[0][i, 0]], [Wg[0][i, 1],Z[0][i, 1]], [Wg[0][i, 2],Z[0][i, 2]], c='r')
# 	plt.show()



	# L = [[0, 0, 0],
	# 	 [1, 0, 0],
	# 	 [2, 0, 0],
	# 	 [3, 0, 0],
	# 	 [0, 1, 0],
	# 	 [1, 1, 0],
	# 	 [2, 1, 0],
	# 	 [3, 1, 0],
	# 	 [0, 2, 0],
	# 	 [1, 2, 0],
	# 	 [2, 2, 0],
	# 	 [3, 2, 0],
	# 	 [0, 3, 0],
	# 	 [1, 3, 0],
	# 	 [2, 3, 0],
	# 	 [3, 3, 0],
	# 	 [0, 0, 1],
	# 	 [1, 0, 1],
	# 	 [2, 0, 1],
	# 	 [3, 0, 1],
	# 	 [0, 1, 1],
	# 	 [1, 1, 1],
	# 	 [2, 1, 1],
	# 	 [3, 1, 1],
	# 	 [0, 2, 1],
	# 	 [1, 2, 1],
	# 	 [2, 2, 1],
	# 	 [3, 2, 1],
	# 	 [0, 3, 1],
	# 	 [1, 3, 1],
	# 	 [2, 3, 1],
	# 	 [3, 3, 1],
	# 	 [0, 0, 2],
	# 	 [1, 0, 2],
	# 	 [2, 0, 2],
	# 	 [3, 0, 2],
	# 	 [0, 1, 2],
	# 	 [1, 1, 2],
	# 	 [2, 1, 2],
	# 	 [3, 1, 2],
	# 	 [0, 2, 2],
	# 	 [1, 2, 2],
	# 	 [2, 2, 2],
	# 	 [3, 2, 2],
	# 	 [0, 3, 2],
	# 	 [1, 3, 2],
	# 	 [2, 3, 2],
	# 	 [3, 3, 2],
	# 	 [0, 0, 3],
	# 	 [1, 0, 3],
	# 	 [2, 0, 3],
	# 	 [3, 0, 3],
	# 	 [0, 1, 3],
	# 	 [1, 1, 3],
	# 	 [2, 1, 3],
	# 	 [3, 1, 3],
	# 	 [0, 2, 3],
	# 	 [1, 2, 3],
	# 	 [2, 2, 3],
	# 	 [3, 2, 3],
	# 	 [0, 3, 3],
	# 	 [1, 3, 3],
	# 	 [2, 3, 3],
	# 	 [3, 3, 3]]