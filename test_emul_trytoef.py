import numpy as np
import pickle
from emulator import gp
from sklearn.model_selection import train_test_split
from utils.design_tools import lhd
np.set_printoptions(formatter={"float_kind": lambda x: "%g" % x})

def main():

	# X = np.loadtxt('nuovi_inputs.txt', dtype=float)

	# a = np.where(X[:, -1] < 212)[0]
	# # b = np.where(X[a, -1] > 80)[0]
	# # c = np.where(X[a[b], -3] > 1e-2)[0]
	# # d = np.where(X[a[b[c]], 2] > 3)[0]
	# # e = np.where(X[a[b[c[d]]], 1] > 3)[0]
	# print(len(a))
	
	# E = np.zeros((8, 3), dtype=float)
	# for i in range(8):
	# 	E[i, 0] = np.min(X[a, i])
	# 	E[i, 1] = np.mean(X[a, i])
	# 	E[i, 2] = np.max(X[a, i])

	# print(X[a,:])

	# with open('nuoovoo.txt', 'w') as f:
	# 	np.savetxt(f, X[a,:], fmt='%f')
	# f.close()

	inFile1 = 'data/mech/ab/inputs.txt'
	inFile2 = 'data/mech/ab/outputs.txt'
	
	X = np.loadtxt(inFile1, dtype=float)
	Y = np.loadtxt(inFile2, dtype=float)

	# emul = gp.GPEmul()
	# emul.fit(X, Y)
	# emul.save('model_MECH_ab')

	# out_dim = Y.shape[1]
	#
	# # Test class is working properly 
	# # ------------------------------
	# for i in range(out_dim):
	# 	emul = gp.GPEmul()
	# 	emul.fit(X, Y[:, i])
	# 	emul.save('model_EP_sham_' + str(i+1))

	
	# Test pickle.load module
	# -----------------------
	with open('model_MECH_ab.pkl', 'rb') as f:
		emul = pickle.load(f)

	n_points = 800

	p0 = []
	for i in range(8):
		p0.append(np.mean(X[:, i]))
	E = np.array([[60, 60], [60, 60], [60, 60], [60, 60], [60, 60], [60, 60], [60, 60], [60, 60]])

	H = np.zeros((1, 8), dtype=float)
	while H.shape[0] - 1 < n_points:
		h = lhd(p0, E, 1)
		y_pred = emul.predict(h)
		if np.sum(np.sign(y_pred)) == 11:
			if y_pred[0, 0] > 400 and y_pred[0, 1] < 200 and y_pred[0, 1] > 100 and y_pred[0, 2] > 50:
				H = np.vstack((H, h))
				print(H.shape[0]-1)

	with open('hi_disegno_ab.txt', 'w') as f:
		np.savetxt(f, H[1:, :], fmt='%f')
	f.close()


	# prfx = np.array([466.5, 125.5, 73.0])

	# p0 = []
	# for i in range(3, 11):
	# 	p0.append(np.mean(X[:, i]))
	# E = np.array([[50, 50], [50, 50], [50, 50], [50, 50], [50, 50], [50, 50], [50, 50], [50, 50]])

	# H = np.zeros((1, 8), dtype=float)
	# while H.shape[0] - 1 < n_points:
	# 	h = lhd(p0, E, 1)
	# 	x_new = np.hstack((prfx, h.ravel()))
	# 	y_pred = emul.predict(x_new.reshape(1, -1))
		
	# 	if np.sum(np.sign(y_pred)) == 8:
	# 		if y_pred[0, 0] > 0.2:
	# 			if y_pred[0, 1] > 4:
	# 				if y_pred[0, 2] > 4:
	# 					if y_pred[0, 3] > 0.5:
	# 						if y_pred[0, 4] > 0.5:
	# 							if y_pred[0, 5] > 1e-3:
	# 								if y_pred[0, 6] > 1e-2:
	# 									if y_pred[0, 7] > 70:
	# 										H = np.vstack((H, y_pred))
	# 										print(H.shape[0]-1)

	# with open('nuovi_inputs.txt', 'w') as f:
	# 	np.savetxt(f, H[1:, :], fmt='%f')
	# f.close()
	

	## Test pickle.load module
	## -----------------------
	# with open('model.pkl', 'rb') as f:
	# 	emul = pickle.load(f)

	# print(emul.X.shape)
	# print(emul.Y.shape)
	# print(emul.mean.steps[0][1])
	# print(emul.mean.steps[1][1])
	# print(emul.gp)
	# print(emul.gp.kernel_)

	# Y_pred = emul.predict(X_test)
	# print(emul.accuracy(Y_test, Y_pred))

#-------------------------

if __name__ == "__main__":
    main()