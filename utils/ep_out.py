import numpy as np
from math import *
from scipy import optimize
from scipy.optimize import curve_fit, brentq

class CalciumBiomarkers():
	def __init__(self, ts, cas, a):
		self.ts = ts
		self.cas = cas
		self.a = a

def get_cabiomarkers(t, ca):
	ts = t[4:-1] - 4
	cas = ca[4:-1]

	a0 = A_output(cas)
	a0[2] = log(2)/a0[2]

	a1 = curve_fit(F, shift_for_a_sec(ts), cas, p0=a0)[0]

	DCa = a1[0]
	TP = brentq(df1, 1, 167, args=a1)
	PCa = F(TP, a1[0], a1[1], a1[2], a1[3])

	try:
		RT50 = brentq(df2, TP, 167, args=(a1, TP)) - TP
	except:
		RT50 = -1

	biomarkers = [DCa, PCa, RT50, TP]

	return CalciumBiomarkers(ts, cas, [biomarkers, a1])

def shift_for_a_sec(ts):
	return [x + 1e-16 for x in ts]

def A_output(ca):
	N = len(ca)
	t = np.linspace(0, N-1, N)

	DCa = min(ca)
	PCa = max(ca)
	i_max = np.argmax(ca)
	TP = t[i_max]

	for i in range(i_max, N):
		if ca[i] <= 0.5*(DCa + PCa):
			RT50 = t[i] - TP
			break

	return [DCa, PCa, RT50, TP]

def F(t, a1, a2, a3, a4):
	return (a2 - a1) * np.power(np.power(t/a4, -1.0) + np.exp(a3*(t - a4)), -1.0) + a1

def df1(t, a):
	return np.power(np.power(t/a[3], -1.0) + np.exp(a[2]*(t - a[3])), -2.0) * (-a[3]*np.power(t, -2.0) + a[2]*np.exp(a[2]*(t - a[3])))

def df2(t, a, b):
	return F(t, a[0], a[1], a[2], a[3]) - 0.5*(F(b, a[0], a[1], a[2], a[3]) + F(0, a[0], a[1], a[2], a[3]))