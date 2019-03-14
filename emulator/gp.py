import numpy as np
import pickle
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

def mare(Y_true, Y_pred):
	sample_dim = Y_true.shape[0]
	out_dim = Y_true.shape[1]
	if out_dim == 1:
		return np.linalg.norm((Y_pred - Y_true)/Y_true, ord=1)/sample_dim
	else:
		e = np.zeros((sample_dim,), dtype=float)
		for i in range(sample_dim):
			e[i] = np.linalg.norm((Y_pred[i, :] - Y_true[i, :])/Y_true[i, :], ord=1)/out_dim
		return np.sum(e)/sample_dim

class GPEmul:
	def __init__(self):
		self.X = []
		self.Y = []
		self.mean = []
		self.gp = []

	def fit(self, X, Y):
		self.X = X
		self.Y = Y

		in_dim = self.X.shape[1]

		pipe = Pipeline([
    		('poly', PolynomialFeatures()),
        	('lr', LinearRegression(n_jobs=-1))
		])

		param_grid1 = {'poly__degree': [1, 2, 3, 4, 5]}
		gs1 = GridSearchCV(pipe, param_grid1,
			n_jobs=-1, iid=False, cv=5, return_train_score=False)
		gs1.fit(self.X, self.Y)
		self.mean = gs1.best_estimator_

		residuals = self.Y - self.mean.predict(self.X)

		param_grid2 = {'kernel': [Matern(length_scale=in_dim*[1.0], nu=i) for i in [1.5, 2.5]] + [RBF(length_scale=in_dim*[1.0])]}
		gs2 = GridSearchCV(GaussianProcessRegressor(n_restarts_optimizer=10), param_grid2,
			n_jobs=-1, iid=False, cv=5, return_train_score=False)
		gs2.fit(self.X, residuals)
		self.gp = gs2.best_estimator_

	def predict(self, X_new, with_std=True):
		if not with_std:
			return self.mean.predict(X_new) + self.gp.predict(X_new)
		else: 
			res, std = self.gp.predict(X_new, return_std=True)
			return self.mean.predict(X_new) + res, std

	def accuracy(self, Y_true, Y_pred):
		return mare(Y_true, Y_pred)

	def save(self, name):
		with open(name + '.pkl', 'wb') as f:
			pickle.dump(self, f)

	def load(self, name):
		with open(name + '.pkl', 'rb') as f:
			vars = pickle.load(f)
		self.X = vars.X
		self.Y = vars.Y
		self.mean = vars.mean
		self.gp = vars.gp