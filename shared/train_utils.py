from Historia.classifier import svm
from Historia.emulator import gp
import numpy as np

def train_classifier(X, y, name_out):
	cla = svm.SVMCla()
	cla.fit(X, y)
	cla.save(name_out)
	return

def train_emulator(X, Y, sublist, name_out, feats='all'):
	if feats == 'single':
		for i in sublist:
			emul = gp.GPEmul()
			emul.fit(X, Y[:, i])
			emul.save(name_out + '_' + str(i))
	else:
		emul = gp.GPEmul()
		emul.fit(X, Y)
		emul.save(name_out)
	return

def predict_vec(emulator, v):
	l = len(emulator)
	mean = np.zeros((l,), dtype=float)
	std = np.zeros((l,), dtype=float)
	for i in range(l):
		mean[i], std[i] = emulator[i].predict(v.ravel().reshape(1, -1))
	return mean, std