import numpy as np
from scipy import optimize
from scipy.optimize import curve_fit, brentq
import matplotlib.pyplot as plt

class TruncCalcium:
	def __init__(self, t, ca):
		self.t = t[4:-1] - 4
		self.ca = ca[4:-1]
		self.a1 = []
		self.bio = []

	def get_cabiomarkers(self):
		a0 = A_output(self.ca)
		a0[2] = np.log(2)/a0[2]

		self.a1 = curve_fit(f, shift_for_a_sec(self.t), self.ca, p0=a0)[0]

		DCa = self.a1[0]
		TP = brentq(df1, self.t[1], self.t[-1], args=self.a1)
		PCa = f(TP, *self.a1)

		try:
			RT50 = brentq(df2, TP, self.t[-1], args=(self.a1, TP)) - TP
		except:
			RT50 = -1

		self.bio = [DCa, PCa, RT50, TP]

	def visualize_biomarkers(self, scene='do_not_show'):
		if len(self.a1) == 0 and len(self.bio) == 0:
			print('Error: first get calcium biomarkers!')
		else:
			plt.plot(self.t, f(self.t, *self.a1))
			plt.scatter(self.bio[3], self.bio[1], c='r')
			plt.scatter(self.bio[2]+self.bio[3], f(self.bio[2]+self.bio[3], *self.a1), c='r')
			plt.axvline(x=self.bio[3], c='r', linestyle='dashed')
			plt.axvline(x=self.bio[2]+self.bio[3], c='r', linestyle='dashed')
			plt.axhline(y=self.bio[0], c='r', linestyle='dashed')
			plt.axhline(y=self.bio[1], c='r', linestyle='dashed')
			if scene == 'show':
				plt.xlim(self.t[0], self.t[-1])
				plt.xlabel('Time [ms]')
				plt.ylabel('Truncated Calcium + Biomarkers')
				plt.show()

class PhenCalcium:
	def __init__(self, t, a):
		self.t = t
		self.ca = np.zeros((167,), dtype=float)
		self.a = a

	def F(self):
		if check_ca(self.a):
			self.ca = 

	
def shift_for_a_sec(ts):
	return [x + 1e-16 for x in ts]

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

def f(t, *a):
	return (a[1] - a[0]) * np.power(np.power(t/a[3], -1.0) + np.exp(a[2]*(t - a[3])), -1.0) + a[0]

def df1(t, a):
	return np.power(np.power(t/a[3], -1.0) + np.exp(a[2]*(t - a[3])), -2.0) * (-a[3]*np.power(t, -2.0) + a[2]*np.exp(a[2]*(t - a[3])))

def df2(t, a, b):
	return f(t, *a) - 0.5*(f(b, *a) + f(0, *a))

def check_ca(a):
	if a[2] == -1:
		conv = 0
	elif a[0] < 0 or a[0] < 0 or a[0] < 0 or a[0] < 0:
		conv = 0
	elif a[0] >= a[1]:
		conv = 0
	else:
		conv = 1

	return conv