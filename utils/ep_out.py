import numpy as np
from scipy import optimize
from scipy.optimize import curve_fit, brentq
import matplotlib.pyplot as plt

class CalciumBiomarkers():
	def __init__(self, ts, cas, b, a):
		self.ts = ts
		self.cas = cas
		self.a = a
		self.bio = bio

def get_cabiomarkers(t, ca):
	ts = t[4:-1] - 4
	cas = ca[4:-1]

	a0 = A_output(cas)
	a0[2] = np.log(2)/a0[2]

	a1 = curve_fit(F, shift_for_a_sec(ts), cas, p0=a0)[0]

	DCa = a1[0]
	TP = brentq(df1, ts[1], ts[-1], args=a1)
	PCa = F(TP, *a1)

	try:
		RT50 = brentq(df2, TP, ts[-1], args=(a1, TP)) - TP
	except:
		RT50 = -1

	biomarkers = [DCa, PCa, RT50, TP]

	return CalciumBiomarkers(ts, cas, a1, biomarkers)

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

def F(t, *a):
	return (a[1] - a[0]) * np.power(np.power(t/a[3], -1.0) + np.exp(a[2]*(t - a[3])), -1.0) + a[0]

def df1(t, a):
	return np.power(np.power(t/a[3], -1.0) + np.exp(a[2]*(t - a[3])), -2.0) * (-a[3]*np.power(t, -2.0) + a[2]*np.exp(a[2]*(t - a[3])))

def df2(t, a, b):
	return F(t, *a) - 0.5*(F(b, *a) + F(0, *a))

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

def visualize_biomarkers(ts, a, bio, scene='do_not_show'):
	plt.plot(ts, F(ts, *a))
	plt.scatter(bio[3], bio[1], c='r')
	plt.scatter(bio[2]+bio[3], F(bio[2]+bio[3], *a), c='r')
	plt.axvline(x=bio[3], c='r', linestyle='dashed')
	plt.axvline(x=bio[2]+bio[3], c='r', linestyle='dashed')
	plt.axhline(y=bio[0], c='r', linestyle='dashed')
	plt.axhline(y=bio[1], c='r', linestyle='dashed')

	if scene == 'show':
		plt.xlim(ts[0], ts[-1])
		plt.xlabel('Time [ms]')
		plt.ylabel('Truncated Calcium + Biomarkers')
		plt.show()

	return