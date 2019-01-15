import numpy as np
from scipy import optimize
from scipy.optimize import curve_fit, brentq
import matplotlib.pyplot as plt

class PhenCalcium:
	def __init__(self, t, ca):
		self.conv = 1
		self.t = t[4:-1] - 4
		self.ca = ca[4:-1]
		self.a1 = []
		self.a = []
		self.bio = []

	def fit(self):
		a0 = A_output(self.ca)
		a0[2] = np.log(2)/a0[2]

		self.a1 = curve_fit(f, shift_for_a_sec(self.t), self.ca, p0=a0)[0]
		self.a = np.copy(self.a1)

	def get_biomarkers(self):
		DCa = self.a[0]
		try:
			TP = brentq(df1, self.t[1], self.t[-1], args=self.a)
		except:
			TP = -1
		if TP != -1:
			PCa = f(TP, *self.a)
		else:
			PCa = -1
		try:
			RT50 = brentq(df2, TP, self.t[-1], args=(self.a, TP)) - TP
		except:
			RT50 = -1
		self.bio = np.array([DCa, PCa, RT50, TP])

		if self.bio[3] == -1 or self.bio[2] == -1:
			self.conv = 0
		elif self.bio[0] < 0 or self.bio[1] < 0 or self.bio[2] < 0 or self.bio[3] < 0:
			self.conv = 0
		elif self.bio[0] >= self.bio[1]:
			self.conv = 0

	def build_ca(self):
		if self.conv:
			self.ca = f(self.t, *self.a)

	def plot(self, visbio=False, scene='do_not_show'):
		plt.plot(self.t, f(self.t, *self.a))
		if visbio:
			plt.scatter(self.bio[3], self.bio[1], c='r')
			plt.scatter(self.bio[2]+self.bio[3], f(self.bio[2]+self.bio[3], *self.a), c='r')
			plt.axvline(x=self.bio[3], c='r', linestyle='dashed')
			plt.axvline(x=self.bio[2]+self.bio[3], c='r', linestyle='dashed')
			plt.axhline(y=self.bio[0], c='r', linestyle='dashed')
			plt.axhline(y=self.bio[1], c='r', linestyle='dashed')
		if scene == 'show':
			plt.xlim(self.t[0], self.t[-1])
			plt.xlabel('Time [ms]')
			plt.ylabel('Intracellular calcium [$\mu$M]')
			plt.show()
	
def A_output(ca):
	N = len(ca)
	t = np.linspace(0, N-1, N)

	DCa = np.min(ca)
	PCa = np.max(ca)
	i_max = np.argmax(ca)
	TP = t[i_max]

	for i in range(i_max, N):
		if ca[i] <= 0.5*(DCa + PCa):
			RT50 = t[i] - TP
			break

	return [DCa, PCa, RT50, TP]

def shift_for_a_sec(ts):
	return [x + 1e-16 for x in ts]

def f(t, *a):
	return (a[1] - a[0]) * np.power(np.power(t/a[3], -1.0) + np.exp(a[2]*(t - a[3])), -1.0) + a[0]

def df1(t, a):
	return np.power(np.power(t/a[3], -1.0) + np.exp(a[2]*(t - a[3])), -2.0) * (-a[3]*np.power(t, -2.0) + a[2]*np.exp(a[2]*(t - a[3])))

def df2(t, a, b):
	return f(t, *a) - 0.5*(f(b, *a) + f(0, *a))