import numpy as np
from scipy.optimize import brentq, curve_fit


D = 0.05
T_EP = 170
NSKIP = 8
T = T_EP - NSKIP
T_MECH = 167
BETA0 = 0.01


class PhenCalcium:
	def __init__(self):
		self.t = []
		self.ca = []
		self.p = []
		self.bio = []


	def fit(self, t, ca):
		n_skip = NSKIP
		self.t = t[n_skip:] - n_skip
		ca = ca[n_skip:]

		p0 = A_output(ca)
		p0[-1] = BETA0

		self.p = curve_fit(f, self.t, ca, p0=p0)[0]


	def get_bio(self):
		DCA = f(self.t, *self.p)[-1]

		AMPL = TP = RT50 = -1
		try:
			TP = brentq(lambda t: der(t, self.p), 0, T)
		except:
			pass

		if TP != -1:
			AMPL = f(TP, *self.p) - DCA

			try:
				RT50 = brentq(lambda t: f(t, *self.p) - 0.5*(f(TP, *self.p) + DCA), TP, T) - TP
			except:
				pass
		
		self.bio = np.array([DCA, AMPL, TP, RT50])


	def build_ca(self, t, p):
		self.p = p
		n_points = T_MECH - len(t)
		t_init = np.linspace(0, n_points-1, n_points)
		self.t = np.concatenate((t_init, t + n_points))
		self.ca = np.concatenate((line(t, self.p, t_init), f(t, *self.p)))
		

#----------------------------------------------------------------


def f1(t, p):
	TP = p[2]

	return t/TP + D


def f2(t, p):
	TP = p[2]
	beta = p[3]

	return (1 + D)*( (np.exp(-beta*(t - T)) - 1) / (np.exp(-beta*(TP - T)) - 1) )


def f(t, *p):
	DCA = p[0]
	AMPL = p[1]

	return AMPL * np.power(np.power(f1(t, p), -1) + np.power(f2(t, p), -1), -1) + DCA


def der(t, p):
	TP = p[2]
	beta = p[3]

	return beta * (np.exp(beta*(T - TP)) - 1) * np.exp(beta*(T - t)) * np.power(TP*D + t, 2) - TP*(D + 1) * np.power(np.exp(beta*(T - t)) - 1, 2)


def A_output(ca):
	N = len(ca)
	t = np.linspace(0, N-1, N)

	DCa = ca[-1]
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
		
	return [DCa, PCa-DCa, TP, RT50]


def line(t, p, t_init):
	n_points = T_MECH - len(t)

	return f(t, *p)[-1] - (f(t, *p)[-1] - f(t, *p)[0])/n_points * t_init