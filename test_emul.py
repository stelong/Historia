import numpy as np
import pickle
from emulator import gp
from sklearn.model_selection import train_test_split

def main():

	emul = gp.GPEmul()

	inFile1 = 'data/inputs.txt'
	inFile2 = 'data/outputs.txt'
	
	X = np.loadtxt(inFile1, dtype=float)
	Y = np.loadtxt(inFile2, dtype=float)

	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=0)
	
	# Test class is working properly 
	# ------------------------------
	emul.fit(X_train, Y_train)
	Y_pred = emul.predict(X_test)
	print(emul.accuracy(Y_test, Y_pred))
	emul.save('model')

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
