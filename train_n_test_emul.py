import numpy as np
import pickle
from emulator import gp
from sklearn.model_selection import train_test_split

def main():

	emul = gp.GPEmul()

	inFile1 = 'data/sham/w1_in.txt'
	inFile2 = 'data/sham/w1_out.txt'
	
	X = np.loadtxt(inFile1, dtype=float)
	Y = np.loadtxt(inFile2, dtype=float)

	# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=None)

	## Single features train and test
	## ------------------------------
	out_dim = Y.shape[1]
	 
	for i in range(2):
		emul = gp.GPEmul()
		emul.fit(X, Y[:, i])
		emul.save('w1_emul' + str(i+1))

		# with open('model_' + str(i+1) + '.pkl', 'rb') as f:
		# 	emul = pickle.load(f)
		# f.close()

		# y_pred = emul.predict(X_test)
		# print(emul.accuracy(Y_test[:, i].reshape(-1, 1), y_pred.reshape(-1, 1)))

	## All features train and test
	## ---------------------------
	# emul = gp.GPEmul()
	# emul.fit(X_train, Y_train)
	# emul.save('model_all')

	# with open('model_all.pkl', 'rb') as f:
	# 	emul = pickle.load(f)
	# f.close()

	# Y_pred = emul.predict(X_test)
	# print(emul.accuracy(Y_test, Y_pred))

#-------------------------

if __name__ == "__main__":
	main()