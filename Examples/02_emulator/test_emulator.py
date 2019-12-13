from Historia.emulator import gp
from Historia.shared import design_utils as desu
from Historia.shared import plot_utils as plut
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import sys

def switch(case):
	return {
		'train': train,
		'test': test
	}.get(case, default)

def default():
	print('Could not find case! Choose another case.')
	return

def train(active_out_feats):
	# read the dataset
	data_path = 'data/'
	X = desu.read_txt(data_path + 'X', 'float64')
	Y = desu.read_txt(data_path + 'Y', 'float64')

	seed = 8
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=seed)

	# path where to save trained emulators
	binaries_path = 'tmp/'
	# emulators shared prefix name
	name = 'emul'

	# train one GP emulator for each specified feature
	gp.GPEmulUtils.train(X_train, Y_train, binaries_path+name, active_out_feats)
	return

def test(active_out_feats):
	# read the dataset
	data_path = 'data/'
	X = desu.read_txt(data_path + 'X', 'float64')
	Y = desu.read_txt(data_path + 'Y', 'float64')

	seed = 8
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=seed)

	# path where trained emulators were saved
	binaries_path = 'tmp/'
	# emulators shared prefix name
	name = 'emul'

	# load list of trained emulators
	emulist = gp.GPEmulUtils.load_emulators(binaries_path+name, active_out_feats)

	# loop over the emulators
	score = []
	for i, emul in enumerate(emulist):
		# get the appropriate component of the testing set
		y_test = Y_test[:, active_out_feats[i]]
		# compute GP posterior mean and std when predicting the testing set
		y_pred, std = emul.predict(X_test)
		# compute the R^2 score (coefficient of determination) using the emulator mean as a point-wise prediction of the model output		
		score.append(r2_score(y_test, y_pred))

	# print the R^2 scores vector
	print('\nObtained R2 scores: {}\n'.format(score))

	# import labels
	labels = ['p{}'.format(i) for i in range(Y.shape[1])]

	# get some nice colours for plotting
	blue = np.array(plut.get_col('blue')[1]).reshape(1, -1)
	orange = np.array(plut.get_col('orange')[1]).reshape(1, -1)

	# example x-vector just for plotting the y-coordinates
	x = np.arange(X_test[:, 0].shape[0]) 

	# plot observed values VS predicted values
	fig, axes = plt.subplots(1, len(active_out_feats))
	for i, ax in enumerate(axes.flatten()):
		# test vector
		y_test = Y_test[:, active_out_feats[i]]
		ax.scatter(x, y_test, c=blue, zorder=2)
		# predicted vector
		y_pred, std = emulist[i].predict(X_test)
		ax.scatter(x, y_pred, c=orange, zorder=1)
		ax.errorbar(x, y_pred, yerr=1.96*std, c=orange, ls='none', zorder=1)
		ax.set_xticks([])
		ax.set_ylabel(labels[active_out_feats[i]], size=12)
	plt.figlegend(['observations', 'predictions', '~95% confidence interval'], loc='upper center')
	plt.show()
	return

def main():
	## First execute 'train', then execute 'test'
	case = sys.argv[1]

	# list of output features to be emulated
	active_out_feats = [0, 1]

	switch(case)(active_out_feats)

if __name__ == '__main__':
	main()