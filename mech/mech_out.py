import numpy as np
from Historia.shared import math_utils as mu
import matplotlib.pyplot as plt

class LeftVentricle:
	def __init__(self):
		self.conv = 0
		self.t = []
		self.phase = []
		self.lv_v = []
		self.lv_p = []
		self.f = []

	def get_lvfeatures(self, S, nc):
		if S.conv:
			v = []
			if S.phase[0] != S.phase[1]:
				v = S.phase[1:]
			elif S.phase[-2] != S.phase[-1]:
				v = S.phase[:-1]
			else:
				v = S.phase
			
			cp = ph_counter(v)

			if cp[3] >= nc + 1:
				self.conv = 1

			if self.conv:
				M = []
				for i in range(4):
					M.append(isl_ranges(np.where(np.asarray(v) == i+1)[0], cp[i]))

				ind = np.sort(np.concatenate([np.reshape(M[i], cp[i]*2) for i in range(4)], axis=0))
	
				last = np.argmax([M[i][-1, 1] for i in range(4)])
				if last == 3:
					rmv = 7
				elif last == 2:
					rmv = 5
				elif last == 1:
					rmv = 3
				else:
					rmv = 1

				ind = ind[:-rmv]
				ind_r = ind[-9:]
				interval = list(range(ind_r[0], ind_r[-1]))

				self.t = [S.t[i] for i in interval]
				self.phase = [S.phase[i] for i in interval]
				self.lv_p = [S.lv_p[i] for i in interval]
				self.lv_v = [S.lv_v[i] for i in interval]

				t1 = S.t[ind_r[0]]
				t2 = S.t[ind_r[1]]
				t3 = S.t[ind_r[4]]
				t4 = S.t[ind_r[5]]
				t5 = S.t[ind_r[7]]

				dP = mu.der(self.t, self.lv_p)
				m = max(self.lv_p)
				ind_m = list(self.lv_p).index(m)

				ps1 = list(np.where(np.asarray(self.phase) == 1)[0])
				lvv1 = [self.lv_v[i] for i in ps1]

				p1 = max(lvv1)         # EDV    (end-diastolic volume)
				p2 = min(self.lv_v)    # ESV    (end-systolic volume)
				p3 = 100*(p1 - p2)/p1  # EF     (ejection fraction)
				p4 = t2 - t1           # IVCT   (isovolumetric contraction time)
				p5 = t3 - t2           # ET     (ejection time)
				p6 = t4 - t3           # IVRT   (isovolumetric relaxation time)
				p7 = t5 - t3           # Tdiast (diastolic time)

				q1 = m                          # PeakP (peak pressure)
				q2 = self.t[ind_m] - self.t[0]  # Tpeak (time to peak pressure)
				q3 = max(dP)                    # maxdP (maximum pressure rise rate) 
				q4 = min(dP)                    # mindP (maximum pressure decay rate)

				self.f = [p1, p2, p3, p4, p5, p6, p7, q1, q2, q3, q4]

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