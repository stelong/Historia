import matplotlib.gridspec as grsp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import metrics

def get_col(color_name):
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
	return colors[color_name]

def interp_col(color, n):
	"""Linearly interpolate a color.
	Inputs:
		- color: list with two elements:
			color[0] = lightest color variant (RGB-triplet in [0, 255])
			color[1] = darkest color variant (RGB-triplet in [0, 255])
		- n: number of desired output colors (n >= 2).
	Output:
		- lsc: list of n linearly scaled colors.
	"""
	color = [[color[i][j]/255 for j in range(3)] for i in range(2)]
	c = [np.interp(list(range(1, n+1)), [1, n], [color[0][i], color[1][i]]) for i in range(3)]
	lsc = [[c[0][i], c[1][i], c[2][i]] for i in range(n)]
	return lsc

def plot_dataset(Xdata, Ydata, xlabels, ylabels):
	sample_dim = Xdata.shape[0]
	in_dim = Xdata.shape[1]
	out_dim = Ydata.shape[1]
	fig, axes = plt.subplots(nrows=out_dim, ncols=in_dim, sharex='col', sharey='row', figsize=(16, 16))                   
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

def plot_overlapping_d(Xdata1, Ydata1, Xdata2, Ydata2, xlabels, ylabels):
	sample_dim1 = Xdata1.shape[0]
	sample_dim2 = Xdata2.shape[0]
	in_dim = Xdata1.shape[1]
	out_dim = Ydata1.shape[1]
	fig, axes = plt.subplots(nrows=out_dim, ncols=in_dim, sharex='col', sharey='row', figsize=(16, 16))                   
	for i, axis in enumerate(axes.flatten()):
		axis.scatter(Xdata1[:, i % in_dim], Ydata1[:, i // in_dim], c='steelblue')
		axis.scatter(Xdata2[:, i % in_dim], Ydata2[:, i // in_dim], c='lightsteelblue')
		inf = min(Xdata1[:, i % in_dim])
		sup = max(Xdata1[:, i % in_dim])
		mean = 0.5*(inf+sup)
		delta = sup-mean
		if i // in_dim == out_dim - 1:
			axis.set_xlabel(xlabels[i % in_dim])
			axis.set_xlim(left=inf-0.3*delta, right=sup+0.3*delta)
			axis.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
		if i % in_dim == 0:
			axis.set_ylabel(ylabels[i // in_dim])
	plt.suptitle('Sample dimension = {} points'.format(sample_dim1))
	plt.show()
	return

def plot_pairwise(Xdata, xlabels):
	sample_dim = Xdata.shape[0]
	in_dim = Xdata.shape[1]
	fig, axes = plt.subplots(nrows=in_dim, ncols=in_dim, sharex='col', sharey='row', figsize=(16, 16))
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

def plot_pairwise_red(X, M, c, xlabels, rat):
	perc = 0.87
	c1 = ['lightsteelblue', 'rosybrown']
	c2 = ['steelblue', 'brown']
	q = 1
	if rat == 'sham':
		q = 0
	in_dim = X.shape[1]
	fig, axes = plt.subplots(nrows=in_dim, ncols=in_dim, sharex='col', sharey='row', figsize=(16, 16))
	for i, ax in enumerate(axes.flatten()):
		sns.scatterplot(ax=ax, x=X[:, i % in_dim], y=X[:, i // in_dim], color=c1[q], edgecolor=c1[q])
		sns.scatterplot(ax=ax, x=M[:, i % in_dim], y=M[:, i // in_dim], color=c2[q], edgecolor=c2[q])
		sns.scatterplot(ax=ax, x=c[:, i % in_dim], y=c[:, i // in_dim], color='k', edgecolor='k')
		if i // in_dim == in_dim - 1:
			ax.set_xlabel(xlabels[i % in_dim])
		if i % in_dim == 0:
			ax.set_ylabel(xlabels[i // in_dim])
	for i in range(in_dim):
		for j in range(in_dim):
			if j > i:
				axes[i, j].set_visible(False)
	plt.figlegend(labels=['initial parameter space', 'after wave 1', 'centroid'], loc='upper center')
	plt.suptitle('Percentage of space reduction = {} %'.format(100-perc), x=0.1, y=0.95, ha='left', va='top')
	plt.show()

def plot_pairwise_waves(XL, colors, xlabels):
	mat_dim = len(XL)
	in_dim = XL[0].shape[1]
	fig, axes = plt.subplots(nrows=in_dim, ncols=in_dim, sharex='col', sharey='row', figsize=(20, 11.3))
	for i, ax in enumerate(axes.flatten()):
		for k in range(mat_dim):
			sns.scatterplot(ax=ax, x=XL[k][:, i % in_dim], y=XL[k][:, i // in_dim], color=colors[k], edgecolor=colors[k])

		if i // in_dim == in_dim - 1:
			ax.set_xlabel(xlabels[i % in_dim])
		if i % in_dim == 0:
			ax.set_ylabel(xlabels[i // in_dim])
	for i in range(in_dim):
		for j in range(in_dim):
			if j > i:
				axes[i, j].set_visible(False)
	plt.figlegend(labels=['wave {}'.format(k) for k in range(mat_dim)], loc='upper center')
	# plt.suptitle('Percentages of sequential space reduction = {} %'.format([np.round(100*(1-XL[i].shape[0]/XL[0].shape[0]), decimals=2) for i in range(1, mat_dim)]), x=0.1, y=0.95, ha='left', va='top')
	# plt.show()
	plt.savefig('history_sham.png', format='png')

def plot_obs_vs_pred(X_test, Y_true, Y_pred, xlabels, ylabels):
	sample_dim = X_test.shape[0]
	in_dim = X_test.shape[1]
	out_dim = Y_true.shape[1]
	fig, axes = plt.subplots(nrows=out_dim, ncols=in_dim, sharex='col', sharey='row', figsize=(16, 16))                   
	for i, axis in enumerate(axes.flatten()):
		axis.scatter(X_test[:, i % in_dim], Y_true[:, i // in_dim], facecolors='none', edgecolor='steelblue')
		axis.scatter(X_test[:, i % in_dim], Y_pred[:, i // in_dim], facecolors='none', edgecolor='brown')
		if i // in_dim == out_dim - 1:
			axis.set_xlabel(xlabels[i % in_dim])
		if i % in_dim == 0:
			axis.set_ylabel(ylabels[i // in_dim])
	plt.figlegend(labels=['Observed LV features', 'Enforced LV features'], loc='upper center')
	plt.show()
	return

def plot_pvloop(S, RS):
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
	# plt.savefig('pv_loop_sham.pdf', format='pdf', dpi=1000)
	plt.show()


#-----------------------------------------------------
# # SVM
# def plot_accuracy_demo(self, X_test, y_test):
# 		score = self.accuracy(X_test, y_test)
# 		y_pred = self.predict(X_test)
# 		CM = metrics.confusion_matrix(y_test, y_pred)
# 		plt.figure(figsize=(8, 8))
# 		sns.heatmap(CM, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r')
# 		plt.xlabel('Predicted label')
# 		plt.ylabel('Actual label')
# 		plt.title('Accuracy score: {0}'.format(score), size=15)
# 		plt.show()

# # SOLVER
# def plot_calcium(self, scene='do_not_show'):
# 	plt.plot(self.t, self.ca)
# 	if scene == 'show':
# 		plt.xlim(self.t[0], self.t[-1])
# 		plt.xlabel('Time [ms]')
# 		plt.ylabel('Intracellular calcium [$\mu$M]')
# 		plt.show()
#
# def plot_solution(self, index, scene='do_not_show'):
# 	plt.plot(self.t, self.Y[index, :])
# 	if scene == 'show':
# 		plt.xlim(self.t[0], self.t[-1])
# 		plt.show()

# # EP_UTILS
# def plot(self, visbio=False, scene='do_not_show'):
# 	plt.plot(self.t, self.ca, zorder=1)
# 	if visbio:
# 		plt.scatter(self.bio[3], self.bio[1], c='r', zorder=3)
# 		plt.scatter(self.bio[2]+self.bio[3], f(self.bio[2]+self.bio[3], *self.a), c='r', zorder=3)
# 		plt.axvline(x=self.bio[3], c='r', linestyle='dashed', linewidth=0.8, zorder=2)
# 		plt.axvline(x=self.bio[2]+self.bio[3], c='r', linestyle='dashed', linewidth=0.8, zorder=2)
# 		plt.axhline(y=self.bio[0], c='r', linestyle='dashed', linewidth=0.8, zorder=2)
# 		plt.axhline(y=self.bio[1], c='r', linestyle='dashed', linewidth=0.8, zorder=2)
# 	if scene == 'show':
# 		plt.xlim(self.t[0], self.t[-1])
# 		plt.xlabel('Time [ms]')
# 		plt.ylabel('Intracellular calcium [$\mu$M]')
# 		plt.show()

# # MECH_UTILS - LVV
# plt.plot(S.t, S.lv_v, c='k', zorder=1)
# plt.axvline(x=t2, c='r', linestyle='dashed', linewidth=0.8, zorder=2)
# plt.axvline(x=t3, c='r', linestyle='dashed', linewidth=0.8, zorder=2)
# plt.axvline(x=t4, c='r', linestyle='dashed', linewidth=0.8, zorder=2)
# plt.axvline(x=t5, c='r', linestyle='dashed', linewidth=0.8, zorder=2)
# plt.scatter(t2, S.lv_v[ind_r[1]], c='r', zorder=3)
# plt.scatter(t3, S.lv_v[ind_r[4]], c='r', zorder=3)
# plt.scatter(t4, S.lv_v[ind_r[5]], c='r', zorder=3)
# plt.scatter(t5, S.lv_v[ind_r[7]], c='r', zorder=3)
# plt.xlim(self.t[0], self.t[-1])
# plt.ylim(np.min(self.lv_v)-0.1*np.min(self.lv_v), np.max(self.lv_v)+0.15*np.max(self.lv_v))
# plt.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
# plt.tick_params(axis='y', which='both', left=False, labelleft=False)
# plt.xlabel('Time')
# plt.ylabel('Left ventricular volume')
# plt.show()
# plt.savefig('lvv.pdf', format='pdf', dpi=1000)

# # MECH_UTILS - LVP
# plt.plot(S.t, S.lv_p, c='k', zorder=1)
# plt.axvline(x=self.t[ind_m], c='r', linestyle='dashed', linewidth=0.8, zorder=2)
# plt.axhline(y=m, c='r', linestyle='dashed', linewidth=0.8, zorder=2)
# plt.plot(self.t, [q4*(self.t[i]-self.t[np.argmin(dP)])+self.lv_p[np.argmin(dP)] for i in range(len(self.t))], c='r', linestyle='dashed', linewidth=0.8, zorder=2)
# plt.plot(self.t, [q3*(self.t[i]-self.t[np.argmax(dP)])+self.lv_p[np.argmax(dP)] for i in range(len(self.t))], c='r', linestyle='dashed', linewidth=0.8, zorder=2)
# plt.scatter(self.t[ind_m], m, c='r', zorder=3)
# plt.scatter(self.t[np.argmin(dP)], self.lv_p[np.argmin(dP)], c='r', zorder=3)
# plt.scatter(self.t[np.argmax(dP)], self.lv_p[np.argmax(dP)], c='r', zorder=3)
# plt.xlim(self.t[0], S.t[-400])
# plt.ylim(-0.5, m+0.28*m)
# plt.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
# plt.tick_params(axis='y', which='both', left=False, labelleft=False)
# plt.xlabel('Time')
# plt.ylabel('Left ventricular pressure')
# plt.show()
# plt.savefig('lvp.pdf', format='pdf', dpi=1000)