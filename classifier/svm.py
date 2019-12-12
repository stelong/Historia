from Historia.shared import design_utils as desu
import numpy as np
import pickle
from scipy.stats import uniform
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

class SVMCla:
	"""This class implements a scikit-learn instance of a Support Vector Machine (SVM) binary classifier.
	The classifier's ('kernel', 'C', 'degree' (only for 'poly' kernels), 'gamma') components set is automatically chosen
	to be the best scoring in a five-fold cross-validation test among all the possible combinations coming from a given grid.
	"""
	def __init__(self):
		self.X = []
		self.y = []
		self.scaler = []
		self.cla = []

	def fit(self, X, y):
		"""Fit a SVM binary classifier on the training set (X, y).
		Args:
			- X: (n, m)-shaped matrix
			- y: (n,)-shaped vector of integers in {0, 1}.
		"""
		self.X = X
		self.y = y

		pipe = Pipeline([
			('scaler', StandardScaler()),
			('cla', SVC(cache_size=1000, class_weight='balanced'))
		])
		param_grid = [
			{'cla__kernel': ['linear'], 'cla__C': [1, 10, 100, 1000]},
			{'cla__kernel': ['poly'], 'cla__C': [1, 10, 100, 1000], 'cla__degree': [2, 3, 4, 5], 'cla__gamma': [1e-1, 1e-2, 1e-3, 1e-4]},
			{'cla__kernel': ['rbf'], 'cla__C': [1, 10, 100, 1000], 'cla__gamma': [1e-1, 1e-2, 1e-3, 1e-4]}
		]

		gs = GridSearchCV(pipe, param_grid, scoring='accuracy', n_jobs=-1,
			cv=5, error_score=0, return_train_score=False)
		gs.fit(self.X, self.y)

		best_model = gs.best_estimator_
		self.scaler = best_model.steps[0][1]
		self.cla = best_model.steps[1][1]

	def predict(self, X_new):
		"""Predict either '0' or '1' labels for each point in a new set.
		Arg:
			X_new: (n, m)-shaped matrix.
		Output:
			(n,)-shaped vector of integers in {0, 1}.
		"""
		return self.cla.predict(self.scaler.transform(X_new))

	def accuracy(self, X_test, y_test):
		"""Compute the mean accuracy in predicting a test set.
		Args:
			- X_test: (n, m)-shaped matrix
			- y_test: (n,)-shaped vector of integers in {0, 1}.
		Output:
			scalar in [0, 1], representing the fraction of correctly guessed labels out of all the labels.
		"""
		return self.cla.score(self.scaler.transform(X_test), y_test)

	def hlc_sample(self, n):
		"""Semple n '1'-labeled points.
		Arg:
			n: positive integer, representing the number of points we want to sample.
		Output:
			D: (n, m)-shaped matrix, containing points sampled from a Latin hypercube that were mapped into '1' by the classifier.
		"""
		in_dim = self.X.shape[1]
		I = desu.get_minmax(self.X)
		D = np.zeros((1, in_dim), dtype=float)
		while D[1:, :].shape[0] < n:
			H = desu.lhd_int(I, n)
			for i in range(n):
				if self.predict(H[i, :].reshape(1, -1))[0]:
					D = np.vstack((D, H[i, :]))
					if D[1:, :].shape[0] == n:
						break
		return D[1:, :]

	def save(self, name):
		"""Save the classifier into a binary file.
		Arg:
			name: string representing the output file name.
		"""
		with open(name + '.pkl', 'wb') as f:
			pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

	@classmethod
	def load(cls, name):
		"""Load the classifier from a binary file.
		Arg:
			name: string representing the input file name.
		"""
		with open(name + '.pkl', 'rb') as f:
			clf_obj = pickle.load(f)
		return clf_obj