from Historia.classifier import svm
from Historia.emulator import gp

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
