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

	inFile1 = 'data/in.txt'
	inFile2 = 'data/out.txt'
	
	X = np.loadtxt(inFile1, dtype=float)
	y = np.loadtxt(inFile2, dtype=float)

	clf.fit(X, y)

	D = clf.hlc_sample(800)

	with open('hi_disegno.txt', 'w') as f:
		np.savetxt(f, D, fmt='%f')
	f.close()

	print(D.shape)
	print(D)
	print(clf.predict(D))

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)
	clf.plot_accuracy_demo(X_test, y_test)

#-------------------------

if __name__ == "__main__":
    main()
