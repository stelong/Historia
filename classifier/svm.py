import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn import pipeline
from sklearn import preprocessing
from sklearn import svm
from sklearn import model_selection
from scipy.stats import uniform

class SVMCla:
	def __init__(self):
		self.X = []
		self.y = []
		self.scaler = []
		self.cla = []

	def fit(self, X, y):
		self.X = X
		self.y = y

		pipe = pipeline.Pipeline([
			('scaler', preprocessing.StandardScaler()),
			('cla', svm.SVC(cache_size=1000, class_weight='balanced'))
		])
		param_grid = [{'cla__kernel': ['linear'], 'cla__C': [1, 10, 100, 1000]},
		{'cla__kernel': ['poly'], 'cla__C': [1, 10, 100, 1000], 'cla__degree': [2, 3, 4, 5], 'cla__gamma': [1e-1, 1e-2, 1e-3, 1e-4]},
		{'cla__kernel': ['rbf'], 'cla__C': [1, 10, 100, 1000], 'cla__gamma': [1e-1, 1e-2, 1e-3, 1e-4]}]

		gs = model_selection.GridSearchCV(pipe, param_grid, scoring='accuracy', n_jobs=-1,
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
	
		p_min = np.zeros(shape=(in_dim,), dtype=float)
		p_max = np.zeros(shape=(in_dim,), dtype=float)
		for i in range(in_dim):
			p_min[i] = np.min(self.X[:, i]) 
			p_max[i] = np.max(self.X[:, i])

		np.random.seed()
		
		D = np.zeros(shape=(1, in_dim), dtype=float)
		while D.shape[0] - 1 < n_points:
			zu = uniform.rvs(loc=0, scale=1, size=in_dim)
			d = np.zeros(shape=(1, in_dim), dtype=float)

			for i in range(in_dim):
				d[0, i] = p_min[i] + (p_max[i] - p_min[i])*zu[i]

			pred = self.predict(d)
			if pred:
				D = np.vstack((D, d))

		return D[1:, :]

	def save(self, name):
		with open(name + '.pkl', 'wb') as f:
			pickle.dump(self, f)
		f.close()

	def plot_accuracy_demo(self, X_test, y_test):
		score = self.accuracy(X_test, y_test)
		y_pred = self.predict(X_test)
		cm = metrics.confusion_matrix(y_test, y_pred)
		
		plt.figure(figsize=(9,9))
		sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
		plt.ylabel('Actual label');
		plt.xlabel('Predicted label');
		all_sample_title = 'Accuracy score: {0}'.format(score)
		plt.title(all_sample_title, size = 15);
		plt.show()