import numpy as np
import pickle
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern, RBF
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

class GPEmul:
	"""This class implements a scikit-learn instance of a Gaussian Process (GP) regressor.
	Firstly, a linear regression model (counting up to 3rd order interactions) is used to predict the mean of the data m(X). Then, a GP is trained on the residuals Y - m(X).
	The GP's 'kernel' component is automatically chosen to be the best scoring in a five-fold cross-validation test among all the possible kernel choices from a given grid.
	"""
	def __init__(self):
		self.X = []
		self.Y = []
		self.mean = []
		self.gp = []

	def fit(self, X, Y):
		"""Fit a GP regressor on the training set (X, Y).
		Args:
			- X: (n, m1)-shaped matrix
			- Y: (n, m2)-shaped matrix.
		"""
		self.X = X
		self.Y = Y

		in_dim = self.X.shape[1]

		pipe = Pipeline([
    		('poly', PolynomialFeatures()),
        	('lr', LinearRegression(n_jobs=-1))
		])

		param_grid1 = {'poly__degree': [1, 2, 3]}
		gs1 = GridSearchCV(pipe, param_grid1,
			n_jobs=-1, iid=False, cv=5, return_train_score=False)
		gs1.fit(self.X, self.Y)
		self.mean = gs1.best_estimator_

		residuals = self.Y - self.mean.predict(self.X)

		param_grid2 = {'kernel': [C()*Matern(length_scale=in_dim*[1.0], nu=i) for i in [1.5, 2.5]] + [C()*RBF(length_scale=in_dim*[1.0])]}
		gs2 = GridSearchCV(GaussianProcessRegressor(n_restarts_optimizer=10), param_grid2,
			n_jobs=-1, iid=False, cv=5, return_train_score=False)
		gs2.fit(self.X, residuals)
		self.gp = gs2.best_estimator_

	def predict(self, X_new, with_std=True):
		"""Predict the output of the mapping F:X->Y for each input point in a new set.
		Arg:
			X_new: (n, m1)-shaped matrix.
		Kwarg:
			with_std: boolean, defining whether to output also the std or not.
		Output:
			- (n, m2)-shaped matrix of GP mean values, if with_std=False
			- (n,)-, (n,)-shaped vectors of respectively GP mean and GP std values, if with_std=True.
			NOTE: for the 'with_std=True' case, the GP must be trained on a single-component output, e.g. (X, Y[:, i]).
		"""
		if not with_std:
			return self.mean.predict(X_new) + self.gp.predict(X_new)
		else: 
			res, std = self.gp.predict(X_new, return_std=True)
			return self.mean.predict(X_new) + res, std

	def save(self, name):
		"""Save the emulator into a binary file.
		Arg:
			name: string representing the output file name.
		"""
		with open(name + '.pkl', 'wb') as f:
			pickle.dump(self, f)

	@classmethod
	def load(cls, name):
		"""Load the emulator from a binary file.
		Arg:
			name: string representing the input file name.
		"""
		with open(name + '.pkl', 'rb') as f:
			emul_obj = pickle.load(f)
		return emul_obj