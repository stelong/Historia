import numpy as np
import pickle
from classifier import svm
from sklearn.model_selection import train_test_split

def main():

	cla = svm.SVMCla()

	inFile1 = 'data/mech/inputs_all.txt'
	inFile2 = 'data/mech/outputs_all.txt'
	
	X = np.loadtxt(inFile1, dtype=float)
	y = np.loadtxt(inFile2, dtype=float)

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)

	cla.fit(X_train, y_train)
	cla.save('cla')

	# with open('cla.pkl', 'rb') as f:
	# 	cla = pickle.load(f)
	# f.close()

	# cla.plot_accuracy_demo(X_test, y_test)

#-------------------------

if __name__ == "__main__":
    main()