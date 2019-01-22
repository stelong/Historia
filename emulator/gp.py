import numpy as np
from sklearn import pipeline
from sklearn import model_selection
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern

class GPEmul:
	def __init__(self):
		self.X = []
		self.Y = []
		self.poly = []
		self.lr = []
		self.mean = []
		self.gp = []

	def fit(self, X, Y):
		self.X = X
		self.Y = Y

		in_dim = self.X.shape[1]

		pipe = pipeline.Pipeline([
    		('poly', preprocessing.PolynomialFeatures()),
        	('lr', linear_model.LinearRegression(n_jobs=-1))
		])

		param_grid1 = {'poly__degree': [1, 2, 3, 4, 5]}
		gs1 = model_selection.GridSearchCV(pipe, param_grid1, n_jobs=-1, iid=False,
			cv=5, return_train_score=False)
		gs1.fit(self.X, self.Y)

		self.poly = gs1.best_estimator_.steps[0][1]
		self.lr = gs1.best_estimator_.steps[1][1]

		X_ = self.poly.transform(self.X)
		self.mean = self.lr.predict(X_)
		Y_ = self.Y - self.mean

		param_grid2 = {'kernel': [Matern(length_scale=in_dim*[1.0], nu=i) for i in [1.5, 2.5]] + [RBF(length_scale=in_dim*[1.0])],
			'alpha': [1e-15, 1e-10, 1e-5, 1e-0]}
		gs2 = model_selection.GridSearchCV(GaussianProcessRegressor(n_restarts_optimizer=10), param_grid2,
			n_jobs=-1, iid=False, cv=5, return_train_score=False)
		gs2.fit(self.X, Y_)
		self.gp = gs2.best_estimator_

	def predict(self, X_new):
		return self.gp.predict(X_new) + self.mean

	def accuracy(self, X_test, Y_test):
		return self.gp.score(X_test, Y_test - self.mean)




# # simple wrapper, because abstract class requires a single class argument
# # -----------------------------------------------------------------------

# # this is the very simple wrapper...
# class model():
#     def __init__(self, mean, gp, X, Y):
#         self.mean = mean
#         self.gp = gp
#         self.X = X
#         self.Y = Y

# for i in range(out_dim):

#     # gp & mean
#     ee = emulators[i]
#     bb = best_models[i]
#     testModel = model(bb, ee, Xdata, Ydata[:,i])

#     # pickle the object, so I don't have to keep rerunning code...
#     import pickle
#     with open('testModel' + str(i) + '.pkl', 'wb') as pickle_file:
#         pickle.dump(testModel, pickle_file)