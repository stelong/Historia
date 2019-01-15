import numpy as np
from utils import math_tools as mt
import matplotlib.pyplot as plt
import matplotlib.gridspec as grsp

class LeftVentricle:
	def __init__(self):
		self.conv = 0
		self.t = []
		self.phase = []
		self.lv_v = []
		self.lv_p = []
		self.f = []

	def get_lvfeatures(self, M, nc):
		if M.conv:
			cp = ph_counter(M.phase)

			if cp[3] >= nc + 1:
				self.conv = 1

			if self.conv:
				M1 = isl_ranges(np.where(np.asarray(M.phase) == 1)[0], cp[0])
				M3 = isl_ranges(np.where(np.asarray(M.phase) == 3)[0], cp[2])

				ind = np.sort(np.concatenate((np.reshape(M1, cp[0]*2), np.reshape(M3, cp[2]*2)), axis=0))
				ind_r = ind[-6:-1]
				interval = list(range(ind_r[0], ind_r[-1]))

				self.t = [M.t[i] for i in interval]
				self.phase = [M.phase[i] for i in interval]
				self.lv_p = [M.lv_p[i] for i in interval]
				self.lv_v = [M.lv_v[i] for i in interval]

				t1 = M.t[ind_r[0]]
				t2 = M.t[ind_r[1]]
				t3 = M.t[ind_r[2]]
				t4 = M.t[ind_r[3]]
				t5 = M.t[ind_r[4]]

				dP = mt.der(self.t, self.lv_p)
				m = max(self.lv_p)
				ind_m = list(self.lv_p).index(m)

				ps1 = list(np.where(np.asarray(self.phase) == 1)[0])
				lvv1 = [self.lv_v[i] for i in ps1]

				p1 = max(lvv1)         # EDV    (end diastolic volume)
				p2 = min(self.lv_v)    # ESV    (end systolic volume)
				p3 = 100*(p1 - p2)/p1  # EF     (ejection fraction)
				p4 = t2 - t1           # ICT    (isovolumetric contraction time)
				p5 = t3 - t2           # ET     (ejection time)
				p6 = t4 - t3           # IRT    (isovolumetric relaxation time)
				p7 = t5 - t4           # Tdiast (diastolic time)

				q1 = m                          # PeakP (maximum pressure)
				q2 = self.t[ind_m] - self.t[0]  # Tpeak (time to peak)
				q3 = max(dP)                    # maxdP (maximum pressure gradient value) 
				q4 = min(dP)                    # mindP (minimum pressure gradient value)

				# self.t = [x - self.t[0] for x in self.t]

				self.f = [p1, p2, p3, p4, p5, p6, p7, q1, q2, q3, q4]

	def plot(self, M):
		gs = grsp.GridSpec(2, 2)
		fig = plt.figure(figsize=(14, 8))
		for i in range(2):
			ax = fig.add_subplot(gs[i, 0])
			if i == 0:
				ax.plot(M.t, M.lv_v, color='b', linewidth=1.0)
				ax.plot(self.t, self.lv_v, color='b', linewidth=2.5)
				plt.xlim(M.t[0], self.t[-1])
				plt.xlabel('Time [ms]')
				plt.ylabel('LVV [$\mu$L]')
			else:
				ax.plot(M.t, M.lv_p, color='b', linewidth=1.0)
				ax.plot(self.t, self.lv_p, color='b', linewidth=2.5)
				plt.xlim(M.t[0], self.t[-1])
				plt.xlabel('Time [ms]')
				plt.ylabel('LVP [kPa]')

		ax = fig.add_subplot(gs[:, 1])
		ax.plot(self.lv_v, self.lv_p, color='b', linewidth=2.5)
		plt.xlabel('Volume [$\mu$L]')
		plt.ylabel('Pressure [kPa]')
		plt.show()

# --------------------

def ph_counter(phase):
	n = len(phase)

	cp = [0, 0, 0, 0]
	for i in range(4):
		j = 0
		while j < n - 1:
			if phase[j] == i + 1:
				cp[i] = cp[i] + 1
				j = j + 1
				while phase[j] == i + 1 and j < n - 1:
					j = j + 1
			else:
				j = j + 1

	return cp

def isl_ranges(l, n_isl):
	len_l = len(l)
	islands = 0
	i = 1

	M = np.zeros(shape=(n_isl, 2), dtype=int)
	M[0, 0] = l[0]
	M[-1, -1] = l[-1]

	while i < len_l:
		if l[i] != l[i-1] + 1:
			M[islands, 1] = l[i-1]
			islands = islands + 1
			M[islands, 0] = l[i]
		i = i + 1

	return M