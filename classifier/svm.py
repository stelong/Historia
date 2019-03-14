import numpy as np
import pickle
from scipy.stats import uniform
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def lhd(I, n_samp):
	n_par = I.shape[0]
	D = np.zeros(shape=(n_samp, n_par))
	H = np.zeros(shape=(n_samp, n_par))
	dp = 1./n_samp
	for j in range(n_par):
		for i in range(n_samp):
			a = i*dp
			D[i, j] = uniform.rvs(loc=a, scale=dp, size=1)

		D[:, j] = np.random.permutation(D[:, j])
		H[:, j] = I[j, 0] + (I[j, 1] - I[j, 0])*D[:, j]
	return H

class SVMCla:
	def __init__(self):
		self.X = []
		self.y = []
		self.scaler = []
		self.cla = []

	def fit(self, X, y):
		self.X = X
		self.y = y

		pipe = Pipeline([
			('scaler', StandardScaler()),
			('cla', SVC(cache_size=1000, class_weight='balanced'))
		])
		param_grid = [{'cla__kernel': ['linear'], 'cla__C': [1, 10, 100, 1000]},
		{'cla__kernel': ['poly'], 'cla__C': [1, 10, 100, 1000], 'cla__degree': [2, 3, 4, 5], 'cla__gamma': [1e-1, 1e-2, 1e-3, 1e-4]},
		{'cla__kernel': ['rbf'], 'cla__C': [1, 10, 100, 1000], 'cla__gamma': [1e-1, 1e-2, 1e-3, 1e-4]}]

		gs = GridSearchCV(pipe, param_grid, scoring='accuracy', n_jobs=-1,
			cv=5, error_score=0, return_train_score=False)
		gs.fit(self.X, self.y)

		best_model = gs.best_estimator_
		self.scaler = best_model.steps[0][1]
		self.cla = best_model.steps[1][1]

	def predict(self, X_new):
		return self.cla.predict(self.scaler.transform(X_new))

	def accuracy(self, X_test, y_test):
		return self.cla.score(self.scaler.transform(X_test), y_test)

	def hlc_sample(self, n_points):
		in_dim = self.X.shape[1]
		I = np.hstack((np.asarray([np.min(self.X[:, i]) for i in range(in_dim)]).reshape(-1, 1), np.asarray([np.max(self.X[:, i]) for i in range(in_dim)]).reshape(-1, 1)))
		D = np.zeros((1, in_dim), dtype=float)
		while D[1:, :].shape[0] < n_points:
			H = lhd(I, n_points)
			for i in range(n_points):
				if self.predict(H[i, :].reshape(1, -1))[0]:
					D = np.vstack((D, H[i, :]))
					if D[1:, :].shape[0] == n_points:
						break
		return D[1:, :]

	def save(self, name):
		with open(name + '.pkl', 'wb') as f:
			pickle.dump(self, f)

	def load(self, name):
		with open(name + '.pkl', 'rb') as f:
			vars = pickle.load(f)
		self.X = vars.X
		self.y = vars.y
		self.scaler = vars.scaler
		self.cla = vars.cla