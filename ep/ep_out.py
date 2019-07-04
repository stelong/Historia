import numpy as np
from scipy.optimize import brentq, curve_fit

class PhenCalcium:
	"""This class implements the calcium transient phenomenological, 4-parameter non-linear equation.
	"""
	def __init__(self):
		self.valid = 1
		self.t = []
		self.ca = []
		self.a = []
		self.bio = []

	def fit(self, t, ca):
		"""Fit the phenomenological equation to the calcium curve (t, ca).
		Args:
			- t: (n,)-shaped vector, representing the time points at which we saved the solution obtained through the 'solver.py' module.
			- ca: (n,)-shaped vector, representing the calcium concentration obtained through the 'solver.py' module.
		"""
		self.t = t[4:] - 4
		ca = ca[4:]

		a0 = A_output(ca)
		a0[2] = np.log(2)/a0[2]

		self.a = curve_fit(f, self.t[1:], ca[1:], p0=a0)[0]
		self.ca = np.insert(f(self.t[1:], *self.a), 0, self.a[0])

	def get_bio(self):
		"""Compute the 4 calcium biomarkers under study. This is analytically done when possible (DCa, PCa), otherwise numerically done (TP, RT50).
		"""
		DCa = self.a[0]
		TP = PCa = RT50 = -1

		try:
			TP = brentq(df, self.t[1], self.t[-1], args=self.a)
		except:
			pass

		if TP != -1:
			PCa = f(TP, *self.a)
			try:
				RT50 = brentq(fm, TP, self.t[-1], args=(self.a, DCa, PCa)) - TP
			except:
				pass
		
		self.bio = [DCa, PCa, RT50, TP]
		if self.bio[2] == -1 or self.bio[0] >= self.bio[1]:
			self.valid = 0

	def build_ca(self, t, a):
		"""Build a calcium curve evaluating the phenomenological equation into (t, a).
		Args:
			- t: (n,)-shaped vector, representing the time points at which we want to observe the calcium concentration.
			- a: list of 4 elements, representing the parameters to be plugged-into the phenomenological equation.
		"""
		self.a = a
		self.t = t[4:] - 4
		self.ca = np.insert(f(self.t[1:], *self.a), 0, self.a[0])

	def check_ca(self):
		"""Check whether the calcium repolarizes or not.
		"""
		pi = self.ca[0]
		pm = f(self.bio[2]+self.bio[3], *self.a)
		dh = (pm - pi)/2
		if self.ca[-1] > pi+dh:
			self.valid = 0

	def print_ca(self, f, stim):
		"""Print to an opened file the entire calcium transiet with a precise syntax for 3D mechanics simulations.
		Args:
			- f: opened file object, usually a *.txt file
			- stim: strictly positive integer, a tag for the specific calcium curve we are saving.
		"""
		f.write('ca{} 165 1 '.format(stim))
		np.savetxt(f, self.ca.reshape(1, -1), fmt='%f')

# useful function 1
def A_output(ca):
	N = len(ca)
	t = np.linspace(0, N-1, N)

	DCa = ca[0]
	imax = np.argmax(ca)
	PCa = ca[imax]
	TP = t[imax]

	j = imax + 1
	RT50 = -1
	while j < N:
		if ca[j] <= 0.5*(PCa + DCa):
			RT50 = t[j] - TP
			break
		j += 1
		
	return [DCa, PCa, RT50, TP]

# phenomenological equation
def f(t, *a):
	return (a[1] - a[0]) * np.power(a[3]/t + np.exp(a[2]*(t - a[3])), -1.0) + a[0]

# useful function 2
def df(t, a):
	return a[2]*np.exp(a[2]*(t - a[3])) - a[3]/np.power(t, 2) 

# useful function 3
def fm(t, a, pmin, pmax):
	return f(t, *a) - 0.5*(pmin + pmax)