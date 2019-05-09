import matplotlib.gridspec as grsp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def get_col(color_name=None):
	"""Material Design color palettes (only '100' and '900' variants).
	Help: call with no arguments to see the list of available colors.
	Kwarg:
		- color_name: string representing the color name.
	Output:
		- color: list with two elements:
			color[1] = lightest color '100'-variant (RGB-triplet in [0, 255])
			color[2] = darkest color '900'-variant (RGB-triplet in [0, 255]).
	"""
	colors = {
		'red' : [[255, 205, 210], [183, 28, 28]],
		'pink' : [[248, 187, 208], [136, 14, 79]],
		'purple' : [[225, 190, 231], [74, 20, 140]],
		'deep_purple' : [[209, 196, 233], [49, 27, 146]],
		'indigo' : [[197, 202, 233], [26, 35, 126]],
		'blue' : [[187, 222, 251], [13, 71, 161]],
		'light_blue' : [[179, 229, 252], [1, 87, 155]],
		'cyan' : [[178, 235, 242], [0, 96, 100]],
		'teal' : [[178, 223, 219], [0, 77, 64]],
		'green' : [[200, 230, 201], [27, 94, 32]],
		'light_green' : [[220, 237, 200], [51, 105, 30]],
		'lime' : [[240, 244, 195], [130, 119, 23]],
		'yellow' : [[255, 249, 196], [245, 127, 23]],
		'amber' : [[255, 236, 179], [255, 111, 0]],
		'orange' : [[255, 224, 178], [230, 81, 0]],
		'deep_orange' : [[255, 204, 188], [191, 54, 12]],
		'brown' : [[215, 204, 200], [62, 39, 35]],
		'gray' : [[245, 245, 245], [33, 33, 33]],
		'blue_gray' : [[207, 216, 220], [38, 50, 56]]
	}
	if not color_name:
		print('\n=== Colors available are:')
		for key, _ in colors.items():
			print('- ' + key)
		return
	else:
		color = colors[color_name]
		return color

def interp_col(color, n):
	"""Linearly interpolate a color.
	Args:
		- color: list with two elements:
			color[0] = lightest color variant (RGB-triplet in [0, 255])
			color[1] = darkest color variant (RGB-triplet in [0, 255]).
		- n: number of desired output colors (n >= 2).
	Output:
		- lsc: list of n linearly scaled colors.
	"""
	color = [[color[i][j]/255 for j in range(3)] for i in range(2)]
	c = [np.interp(list(range(1, n+1)), [1, n], [color[0][i], color[1][i]]) for i in range(3)]
	lsc = [[c[0][i], c[1][i], c[2][i]] for i in range(n)]
	return lsc

def plot_pairwise(Xdata, xlabels):
	"""Plot X high-dimensional dataset by pairwise plotting its features against each other.
	Args:
		- Xdata: n*m matrix
		- xlabels: list of m strings representing the name of X dataset's features.
	"""
	sample_dim = Xdata.shape[0]
	in_dim = Xdata.shape[1]
	fig, axes = plt.subplots(nrows=in_dim, ncols=in_dim, sharex='col', sharey='row', figsize=(20, 11.3))
	for i, axis in enumerate(axes.flatten()):
		axis.scatter(Xdata[:, i % in_dim], Xdata[:, i // in_dim])
		if i // in_dim == in_dim - 1:
			axis.set_xlabel(xlabels[i % in_dim])
		if i % in_dim == 0:
			axis.set_ylabel(xlabels[i // in_dim])
	for i in range(in_dim):
		for j in range(in_dim):
			if i < j:
				axes[i, j].set_visible(False)
	plt.suptitle('Sample dimension = {} points'.format(sample_dim), x=0.1, y=.95, ha='left', va='top')
	plt.show()
	return

def plot_dataset(Xdata, Ydata, xlabels, ylabels):
	"""Plot Y high-dimensional dataset by pairwise plotting its features against each X dataset's feature.
	Args:
		- Xdata: n*m1 matrix
		- Ydata: n*m2 matrix
		- xlabels: list of m1 strings representing the name of X dataset's features
		- ylabels: list of m2 strings representing the name of Y dataset's features.
	"""
	sample_dim = Xdata.shape[0]
	in_dim = Xdata.shape[1]
	out_dim = Ydata.shape[1]
	fig, axes = plt.subplots(nrows=out_dim, ncols=in_dim, sharex='col', sharey='row', figsize=(20, 11.3))                   
	for i, axis in enumerate(axes.flatten()):
		axis.scatter(Xdata[:, i % in_dim], Ydata[:, i // in_dim])
		inf = min(Xdata[:, i % in_dim])
		sup = max(Xdata[:, i % in_dim])
		mean = 0.5*(inf+sup)
		delta = sup-mean
		if i // in_dim == out_dim - 1:
			axis.set_xlabel(xlabels[i % in_dim])
			axis.set_xlim(left=inf-0.3*delta, right=sup+0.3*delta)
			# axis.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
		if i % in_dim == 0:
			axis.set_ylabel(ylabels[i // in_dim])
	plt.suptitle('Sample dimension = {} points'.format(sample_dim))
	plt.show()
	return

def plot_pairwise_waves(XL, colors, xlabels):
	"""Plot a vector XL of overlapping high-dimensional datasets by means of pairwise components plotting.
	Args:
		- XL: list of L matrices with dimensions n*m_i, for i=1,...,L
		- colors: list of L colors
		- xlabels: list of n strings representing the name of X_is datasets' common features.
	"""
	L = len(XL)
	in_dim = XL[0].shape[1]
	fig, axes = plt.subplots(nrows=in_dim, ncols=in_dim, sharex='col', sharey='row', figsize=(20, 11.3))
	for i, ax in enumerate(axes.flatten()):
		for k in range(L):
			sns.scatterplot(ax=ax, x=XL[k][:, i % in_dim], y=XL[k][:, i // in_dim], color=colors[k], edgecolor=colors[k])
		if i // in_dim == in_dim - 1:
			ax.set_xlabel(xlabels[i % in_dim])
		if i % in_dim == 0:
			ax.set_ylabel(xlabels[i // in_dim])
	for i in range(in_dim):
		for j in range(in_dim):
			if j > i:
				axes[i, j].set_visible(False)
	plt.figlegend(labels=['wave {}'.format(k+1) for k in range(L)], loc='upper center')
	plt.show()
	return

def plot_pvloop(S, RS):
	""" Plot LV volume and pressure curves and pv-loop.
	Args:
		- S: mechanics simulation solution class
		- RS: last heart beat solution class
	"""
	gs = grsp.GridSpec(2, 2)
	fig = plt.figure(figsize=(14, 8))
	for i in range(2):
		ax = fig.add_subplot(gs[i, 0])
		if i == 0:
			ax.plot(RS.t, RS.lv_v, color='b', linewidth=1.0)
			ax.plot(S.t, S.lv_v, color='b', linewidth=2.5)
			plt.xlim(RS.t[0], S.t[-1])
			plt.xlabel('Time [ms]')
			plt.ylabel('LVV [$\mu$L]')
		else:
			ax.plot(RS.t, RS.lv_p, color='b', linewidth=1.0)
			ax.plot(S.t, S.lv_p, color='b', linewidth=2.5)
			plt.xlim(RS.t[0], S.t[-1])
			plt.xlabel('Time [ms]')
			plt.ylabel('LVP [kPa]')

	ax = fig.add_subplot(gs[:, 1])
	ax.plot(S.lv_v, S.lv_p, color='b', linewidth=2.5)
	plt.xlabel('Volume [$\mu$L]')
	plt.ylabel('Pressure [kPa]')
	plt.show()
	return