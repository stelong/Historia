import numpy as np
from scipy.optimize import brentq, curve_fit

class PhenCalcium:
	"""This class implements the calcium transient phenomenological, 6-parameter non-linear equation.
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

		outs = A_output(ca)
		a0 = [outs[-1], t[4]+outs[-1], 1, outs[0], outs[1], 0.01]

		self.a = curve_fit(f, self.t, ca, p0=a0)[0]
		self.ca = f(self.t, *self.a)

	def get_bio(self):
		"""Compute the 4 calcium biomarkers under study.
		"""
		T = 166
		DCa = self.a[3]
		AMPL = self.a[4]
		Tpeak = self.a[0]
		try:
			RT50 = compute_alg(Tpeak, self.a[1], self.a[2], T, DCa, AMPL, self.a[-1])[-1]
		except:
			RT50 = -1

		self.bio = [DCa, AMPL, Tpeak, RT50]
		if self.bio[2] == -1:
			self.valid = 0

	def check_ca(self):
		T = 166
		t1 = self.a[1] - self.a[0]
		d, gamma = compute_alg(self.a[0], self.a[1], self.a[2], T, self.a[3], self.a[4], self.a[5])[:2]
		point = f2(t1, self.a[0], self.a[1], self.a[2], T, self.a[3], self.a[4], self.a[5], d, gamma)
		if not (self.a[3] < point and point < self.a[3] + 0.16*self.a[4] and 2 < t1 and t1 < 6):
			self.valid = 0

	def build_ca(self, t, a):
		"""Build a calcium curve evaluating the phenomenological equation into (t, a).
		Args:
			- t: (n,)-shaped vector, representing the time points at which we want to observe the calcium concentration.
			- a: list of 6 elements, representing the parameters to be plugged-into the phenomenological equation.
		"""
		self.t = t
		self.a = a
		self.ca = f(self.t, *self.a)


def A_output(ca):
	N = len(ca)
	t = np.linspace(0, N-1, N)

	DCA = ca[0]
	imax = np.argmax(ca)
	PCA = ca[imax]
	TP = t[imax]

	j = imax + 1
	RT50 = -1
	while j < N:
		if ca[j] <= 0.5*(PCA + DCA):
			RT50 = t[j] - TP
			break
		j += 1

	AMPL = PCA - DCA
	Tpeak = TP - t[4]

	return [DCA, AMPL, RT50, Tpeak]
	

def f1(t, Delta_t1, tp, Delta_t2, T, DCA, AMPL, beta, d, gamma):
	return (d*np.power(-Delta_t1, 2) + AMPL)*(1 - np.power(1 - t/(tp - Delta_t1), 1/3)) + DCA

def f2(t, Delta_t1, tp, Delta_t2, T, DCA, AMPL, beta, d, gamma):
	return d*np.power(t - tp, 2) + (DCA + AMPL)

def f3(t, Delta_t1, tp, Delta_t2, T, DCA, AMPL, beta, d, gamma):
	return gamma*(np.exp(-beta*t) - np.exp(-beta*T)) + DCA

def compute_alg(Delta_t1, tp, Delta_t2, T, DCA, AMPL, beta):
	d = -beta*AMPL/(beta*np.power(Delta_t2, 2) + 2*(Delta_t2)*(1 - np.exp(-beta*(T - (tp + Delta_t2)))))
	gamma = AMPL/(np.exp(-beta*(tp + Delta_t2))*(1 + 0.5*beta*(Delta_t2)) - np.exp(-beta*T))
	RT50 = -np.log(AMPL/(2*gamma) + np.exp(-beta*T))/beta - tp
	return d, gamma, RT50

def f(t, *p):
	Delta_t1 = p[0]
	tp = p[1]
	Delta_t2 = p[2]
	T = 166
	DCA = p[3]
	AMPL = p[4]
	beta = p[5]
	
	d, gamma, _ = compute_alg(Delta_t1, tp, Delta_t2, T, DCA, AMPL, beta)
	
	y = []
	for ti in t:
		if ti < tp-Delta_t1:
			y.append(f1(ti, Delta_t1, tp, Delta_t2, T, DCA, AMPL, beta, d, gamma))
		elif tp-Delta_t1 <= ti < tp+Delta_t2:
			y.append(f2(ti, Delta_t1, tp, Delta_t2, T, DCA, AMPL, beta, d, gamma))
		elif tp+Delta_t2 <= ti:
			y.append(f3(ti, Delta_t1, tp, Delta_t2, T, DCA, AMPL, beta, d, gamma))
	return np.array(y)
