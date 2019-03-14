from history.classifier import svm
from history.emulator import gp

def train_classifier(X, y, name_out):
	cla = svm.SVMCla()
	cla.fit(X, y)
	cla.save(name_out)
	return

def train_emulator(X, Y, name_out, feats='all'):
	if feats == 'single':
		out_dim = Y.shape[1]
		for i in range(out_dim):
			emul = gp.GPEmul()
			emul.fit(X, Y[:, i])
			emul.save(name_out + '_' + str(i+1))
	else:
		emul = gp.GPEmul()
		emul.fit(X, Y)
		emul.save(name_out)
	return