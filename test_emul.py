from ep import solver as s
from ep.models import Gattoni_SHAM6Hz as sham
from ep.models import Gattoni_AB6Hz as ab
from utils import ep_out as e
from classifier import svm as c
from emulator import gp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def my_custom_loss_func(Y_true, Y_pred):
	n_samp = Y_true.shape[0]

	e = np.zeros(shape=(n_samp,), dtype=float)
	for i in range(n_samp):
		e[i] = np.linalg.norm((Y_pred[i, :] - Y_true[i, :])/Y_true[i, :], ord=1)

	metric = np.sum(e)/n_samp

	return metric

def main():

	emul = gp.GPEmul()

	inFile1 = 'data/inputs.txt'
	inFile2 = 'data/outputs.txt'
	
	X = np.loadtxt(inFile1, dtype=float)
	Y = np.loadtxt(inFile2, dtype=float)[:, 2]

	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
	
	emul.fit(X_train, Y_train)
	print(emul.gp)
	print(emul.gp.kernel_)
	

	# print(emul.accuracy(X_test, Y_test))
	Y_pred = emul.predict(X_test)
	print(Y_pred)
	print(my_custom_loss_func(Y_test, Y_pred))


#-------------------------

if __name__ == "__main__":
    main()
