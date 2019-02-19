from emulator import gp
import matplotlib.pyplot as plt
import numpy as np

def plot_dataset(Xdata, Ydata, xlabels, ylabels):
	in_dim = Xdata.shape[1]
	out_dim = Ydata.shape[1]
	sample_dim = Xdata.shape[0]

	fig, axes = plt.subplots(nrows=out_dim, ncols=in_dim, sharex='col', sharey='row', figsize=(16, 16))                   
	for i, a in enumerate(axes.flatten()):
		a.scatter(Xdata[:, i % in_dim], Ydata[:, i // in_dim])
		if i // in_dim == out_dim - 1:
			a.set_xlabel(xlabels[i % in_dim])
		if i % in_dim == 0:
			a.set_ylabel(ylabels[i // in_dim])
	plt.suptitle('Sample dimension = {} points'.format(sample_dim))
	plt.show()

	return

def main():
	rat = 'sham'

	inFile1 = 'data/mech/' + rat + '/inputs.txt'
	inFile2 = 'data/mech/' + rat + '/outputs.txt'
	
	X = np.loadtxt(inFile1, dtype=float)
	Y = np.loadtxt(inFile2, dtype=float)

	# xlabels = ['p', 'ap', 'z', 'c1', 'ca50', 'kxb', 'koff', 'Tref']
	# ylabels = ['EDV', 'ESV', 'EF', 'ICT', 'ET', 'IRT', 'Tdiast', 'PeakP', 'Tpeak', 'maxdP', 'mindP']
	# plot_dataset(X, Y, xlabels, ylabels)

	active_out_feat = Y.shape[1]
	for i in range(active_out_feat):
		emul = gp.GPEmul()
		emul.fit(X, Y[:, i])
		emul.save('history/trained_emulators/' + rat + '/w1_emul' + str(i+1))

#-------------------------

if __name__ == "__main__":
	main()