from ep import solver as s
from ep.models import Gattoni_SHAM6Hz as sham
from ep.models import Gattoni_AB6Hz as ab
from utils import ep_out as e
from classifier import svm as c
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def main():

	clf = c.SVMCla()

	inFile1 = 'AB_10_inputs.txt'
	inFile2 = 'AB_11_inputs.txt'
	outFile1 = 'AB_10_outputs.txt' 
	outFile2 = 'AB_11_outputs.txt'

	X1 = np.loadtxt(inFile1, dtype=float)
	X2 = np.loadtxt(inFile2, dtype=float)
	y1 = np.loadtxt(outFile1, dtype=int)
	y2 = np.loadtxt(outFile2, dtype=int)

	X = np.vstack((X1, X2))
	y = np.vstack((y1, y2))

	clf.fit(X, y)

	D = clf.hlc_sample(800)

	with open('hi_disegno.txt', 'w') as f:
		np.savetxt(f, D, fmt='%f')
	f.close()

	# print(D.shape)
	# print(D)
	# print(clf.predict(D))

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)
	clf.plot_accuracy_demo(X_test, y_test)

#-------------------------

if __name__ == "__main__":
    main()
