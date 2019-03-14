import matplotlib.pyplot as plt

def plot_dataset(Xdata, Ydata, xlabels, ylabels):
	sample_dim = Xdata.shape[0]
	in_dim = Xdata.shape[1]
	out_dim = Ydata.shape[1]
	fig, axes = plt.subplots(nrows=out_dim, ncols=in_dim, sharex='col', sharey='row', figsize=(16, 16))                   
	for i, axis in enumerate(axes.flatten()):
		axis.scatter(Xdata[:, i % in_dim], Ydata[:, i // in_dim])
		if i // in_dim == out_dim - 1:
			axis.set_xlabel(xlabels[i % in_dim])
		if i % in_dim == 0:
			axis.set_ylabel(ylabels[i // in_dim])
	plt.suptitle('Sample dimension = {} points'.format(sample_dim))
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

#-----------------------------------

# import matplotlib.gridspec as grsp
# import seaborn as sns
# from sklearn import metrics

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

# # MECH_UTILS

# plt.plot(M.t, M.lv_v, c='k', zorder=1)
# plt.axvline(x=t2, c='r', linestyle='dashed', linewidth=0.8, zorder=2)
# plt.axvline(x=t3, c='r', linestyle='dashed', linewidth=0.8, zorder=2)
# plt.axvline(x=t4, c='r', linestyle='dashed', linewidth=0.8, zorder=2)
# plt.axvline(x=t5, c='r', linestyle='dashed', linewidth=0.8, zorder=2)
# plt.scatter(t2, M.lv_v[ind_r[1]], c='r', zorder=3)
# plt.scatter(t3, M.lv_v[ind_r[2]], c='r', zorder=3)
# plt.scatter(t4, M.lv_v[ind_r[3]], c='r', zorder=3)
# plt.scatter(t5, M.lv_v[ind_r[4]], c='r', zorder=3)
# plt.xlim(self.t[0], M.t[-256])
# plt.ylim(np.min(self.lv_v)-0.1*np.min(self.lv_v), np.max(self.lv_v)+0.15*np.max(self.lv_v))
# plt.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
# plt.tick_params(axis='y', which='both', left=False, labelleft=False)
# plt.xlabel('Time')
# plt.ylabel('Left ventricular volume')
# plt.savefig('lvv.pdf', format='pdf', dpi=1000)

#----------------------------------------------------

# plt.plot(M.t, M.lv_p, c='k', zorder=1)
# plt.axvline(x=self.t[ind_m], c='r', linestyle='dashed', linewidth=0.8, zorder=2)
# plt.axhline(y=m, c='r', linestyle='dashed', linewidth=0.8, zorder=2)
# plt.plot(self.t, [q4*(self.t[i]-self.t[np.argmin(dP)])+self.lv_p[np.argmin(dP)] for i in range(len(self.t))], c='r', linestyle='dashed', linewidth=0.8, zorder=2)
# plt.plot(self.t, [q3*(self.t[i]-self.t[np.argmax(dP)])+self.lv_p[np.argmax(dP)] for i in range(len(self.t))], c='r', linestyle='dashed', linewidth=0.8, zorder=2)
# plt.scatter(self.t[ind_m], m, c='r', zorder=3)
# plt.scatter(self.t[np.argmin(dP)], self.lv_p[np.argmin(dP)], c='r', zorder=3)
# plt.scatter(self.t[np.argmax(dP)], self.lv_p[np.argmax(dP)], c='r', zorder=3)
# plt.xlim(self.t[0], M.t[-400])
# plt.ylim(-0.5, m+0.28*m)
# plt.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
# plt.tick_params(axis='y', which='both', left=False, labelleft=False)
# plt.xlabel('Time')
# plt.ylabel('Left ventricular pressure')
# plt.savefig('lvp.pdf', format='pdf', dpi=1000)

#----------------------------------------------------

# def plot(self, M):
# 	gs = grsp.GridSpec(2, 2)
# 	fig = plt.figure(figsize=(14, 8))
# 	for i in range(2):
# 		ax = fig.add_subplot(gs[i, 0])
# 		if i == 0:
# 			ax.plot(M.t, M.lv_v, color='b', linewidth=1.0)
# 			ax.plot(self.t, self.lv_v, color='b', linewidth=2.5)
# 			plt.xlim(M.t[0], self.t[-1])
# 			plt.xlabel('Time [ms]')
# 			plt.ylabel('LVV [$\mu$L]')
# 		else:
# 			ax.plot(M.t, M.lv_p, color='b', linewidth=1.0)
# 			ax.plot(self.t, self.lv_p, color='b', linewidth=2.5)
# 			plt.xlim(M.t[0], self.t[-1])
# 			plt.xlabel('Time [ms]')
# 			plt.ylabel('LVP [kPa]')

# 		ax = fig.add_subplot(gs[:, 1])
# 		ax.plot(self.lv_v, self.lv_p, color='b', linewidth=2.5)
# 		plt.xlabel('Volume [$\mu$L]')
# 		plt.ylabel('Pressure [kPa]')
# 		plt.show()