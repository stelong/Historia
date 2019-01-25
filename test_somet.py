import numpy as np
from classifier import svm as c
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
np.set_printoptions(formatter={"float_kind": lambda x: "%g" % x})

def main():
	X = np.loadtxt('data/EP_3/ep_3_in.txt', dtype=float)
	Y = np.loadtxt('data/EP_3/ep_3_out.txt', dtype=float)

	# X2 = np.loadtxt('data/EP_2/ep_2_in.txt', dtype=float)
	# Y2 = np.loadtxt('data/EP_2/ep_2_conv.txt', dtype=float)

	# X = np.vstack((X1, X2))
	# y = np.hstack((Y1, Y2))

	sample_dim = X.shape[0]
	in_dim = X.shape[1]
	out_dim = Y.shape[1]
	y = []
	for i in range(sample_dim):
		if np.sum(Y[i, :]) == 0:
			y.append(0)
		else:
			y.append(1)

	with open('ep_3_conv.txt', 'w') as f:
		np.savetxt(f, y, fmt='%d')
	f.close()

	# clf = c.SVMCla()

	# clf.fit(X, y)

	# D = clf.hlc_sample(1000)

	# with open('hi_disegno.txt', 'w') as f:
	# 	np.savetxt(f, D, fmt='%f')
	# f.close()

	# print(D.shape)
	# print(D)
	# print(clf.predict(D))

	# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)
	# clf.plot_accuracy_demo(X_test, y_test)

# ------------------------

if __name__ == '__main__':
	main()