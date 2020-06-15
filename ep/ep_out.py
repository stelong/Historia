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
		self.t = t[9:] - 9
		ca = ca[9:]

		a0 = A_output(ca)
		a0[2] = a0[0]

		self.a = curve_fit(f, self.t, ca, p0=a0)[0]
		self.ca = f(self.t, *self.a)

	def get_bio(self):
		"""Compute the 4 calcium biomarkers under study. This is analytically done when possible (DCa, AMPL), otherwise numerically done (Tpeak, RT50).
		"""
		DCa = self.a[0]
		Tpeak = PCa = RT50 = -1

		try:
			Tpeak = brentq(df, self.t[0], self.t[-1], args=self.a)
		except:
			pass

		if Tpeak != -1:
			PCa = f(Tpeak, *self.a)
			try:
				RT50 = brentq(fm, Tpeak, self.t[-1], args=(self.a, DCa, PCa)) - Tpeak
			except:
				pass
		
		AMPL = DCa + PCa
		self.bio = [DCa, AMPL, RT50, Tpeak]
		if self.bio[2] == -1:
			self.valid = 0

	def build_ca(self, t, a):
		"""Build a calcium curve evaluating the phenomenological equation into (t, a).
		Args:
			- t: (n,)-shaped vector, representing the time points at which we want to observe the calcium concentration.
			- a: list of 4 elements, representing the parameters to be plugged-into the phenomenological equation.
		"""
		self.t = t
		self.a = a
		self.ca = f(self.t, *self.a)


def A_output(ca):
	N = len(ca)
	t = np.linspace(0, N-1, N)

	DCa = ca[0]
	imax = np.argmax(ca)
	PCa = ca[imax]
	Tpeak = t[imax]

	j = imax + 1
	RT50 = -1
	while j < N:
		if ca[j] <= 0.5*(PCa + DCa):
			RT50 = t[j] - Tpeak
			break
		j += 1
		
	return [DCa, PCa, RT50, Tpeak]


def f1(t, a):
	return t/a[3]

def f2(t, a):
	T = 166
	return (1 + a[2])*np.power(a[2]/(1 + a[2]), (t - a[3])/(T - a[3])) - a[2]

def f(t, *a):
	return (a[1] - a[0])*np.power(np.power(f1(t, a), -1) + np.power(f2(t, a), -1), -1) + a[0]

def df(t, a):
	T = 166
	return 1/(1 + a[2])*(1/(T - a[3]))*np.log((1 + a[2])/a[2])*np.power((1 + a[2])/a[2], (t - a[3])/(T - a[3])) - a[3]/np.power(t, 2)

def fm(t, a, pmin, pmax):
	return f(t, *a) - 0.5*(pmin + pmax)