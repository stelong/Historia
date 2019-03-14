import numpy as np
from scipy.optimize import brentq, curve_fit

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

		self.a1 = curve_fit(f, self.t[1:], self.ca[1:], p0=a0)[0]
		self.a = np.copy(self.a1)
		self.ca = np.insert(f(self.t[1:], *self.a1), 0, self.a1[0]) 

	def get_bio(self):
		DCa = self.a[0]
		try:
			TP = brentq(df, self.t[1], self.t[-1], args=self.a)
		except:
			TP = -1
		if TP != -1:
			PCa = f(TP, *self.a)
		else:
			PCa = -1
		if TP != -1 and PCa != -1:
			try:
				RT50 = brentq(fm, TP, self.t[-1], args=(self.a, DCa, PCa)) - TP
			except:
				RT50 = -1
		else:
			RT50 = -1
		
		self.bio = [DCa, PCa, RT50, TP]

	def build_ca(self):
		self.ca = np.insert(f(self.t[1:], *self.a), 0, self.a[0])

	def check_ca(self):
		pi = self.ca[0]
		pm = f(self.bio[2]+self.bio[3], *self.a)
		dh = (pm - pi)/2
		if self.bio[2] == -1 or self.bio[0] >= self.bio[1] or self.ca[-1] > pi+dh:
			self.conv = 0
		else:
			self.conv = 1

	def print_ca(self, f, stim):
		f.write('ca{} 166 1 '.format(stim))
		np.savetxt(f, self.ca.reshape(1, -1), fmt='%f')

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

def f(t, *a):
	return (a[1] - a[0]) * np.power(a[3]/t + np.exp(a[2]*(t - a[3])), -1.0) + a[0]

def df(t, a):
	return a[2]*np.exp(a[2]*(t - a[3])) - a[3]/np.power(t, 2) 

def fm(t, a, pmin, pmax):
	return f(t, *a) - 0.5*(pmin + pmax)

def divide_et_impera(name_in):
	with open(name_in + '.txt', 'r') as f:
		lines = f.readlines()
		for i in range(4):
			with open(name_in + '_' + str(i+1) + '.txt', 'w') as fi:
				for c, line in enumerate(lines):
					if c // 200 == i:
						fi.write(line)
	return

def append_to_header(rat, ca_traces, number):
	with open('utils/headers/' + rat + '/MeshFlatBase.in', 'r') as f:
		with open('MeshFlatBase_' + str(number) + '.in', 'w') as fh:
			for line in f:
				fh.write(line)
			with open(ca_traces + '_' + str(number) + '.txt', 'r') as fc:
				for line in fc:
					fh.write(line)
	return