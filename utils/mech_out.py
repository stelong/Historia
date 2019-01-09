import numpy as np
from utils import math_tools as mt

class LeftVentricularFeatures():
	def __init__(self, ts, ps, lv_vs, lv_ps, f):
		self.ts = ts
		self.ps = ps
		self.lv_vs = lv_vs
		self.lv_ps = lv_ps
		self.f = f

def get_lvfeatures(t, phase, lv_v, lv_p):
	cp = ph_counter(phase)

	M1 = isl_ranges(np.where(np.asarray(phase) == 1)[0], cp[0])
	M3 = isl_ranges(np.where(np.asarray(phase) == 3)[0], cp[2])

	ind = np.sort(np.concatenate((np.reshape(M1, cp[0]*2), np.reshape(M3, cp[2]*2)), axis=0))
	ind_r = ind[-6:-1]
	interval = list(range(ind_r[0], ind_r[-1]))

	ts = [t[i] for i in interval]
	ps = [phase[i] for i in interval]
	lv_ps = [lv_p[i] for i in interval]
	lv_vs = [lv_v[i] for i in interval]

	t1 = t[ind_r[0]]
	t2 = t[ind_r[1]]
	t3 = t[ind_r[2]]
	t4 = t[ind_r[3]]
	t5 = t[ind_r[4]]

	dP = mt.der(ts, lv_ps)
	m = max(lv_ps)
	ind_m = list(lv_ps).index(m)

	ps1 = list(np.where(np.asarray(ps) == 1)[0])
	lvv1 = [lv_vs[i] for i in ps1]

	p1 = max(lvv1)         # EDV    (end diastolic volume)
	p2 = min(lv_vs)        # ESV    (end systolic volume)
	p3 = 100*(p1 - p2)/p1  # EF     (ejection fraction)
	p4 = t2 - t1           # ICT    (isovolumetric contraction time)
	p5 = t3 - t2           # ET     (ejection time)
	p6 = t4 - t3           # IRT    (isovolumetric relaxation time)
	p7 = t5 - t4           # Tdiast (diastolic time)

	q1 = m                  # PeakP (maximum pressure)
	q2 = ts[ind_m] - ts[0]  # Tpeak (time to peak)
	q3 = max(dP)            # maxdP (maximum pressure gradient value) 
	q4 = min(dP)            # mindP (minimum pressure gradient value)

	# ts = [x - ts[0] for x in ts]

	features = [p1, p2, p3, p4, p5, p6, p7, q1, q2, q3, q4]

	return LeftVentricularFeatures(ts, ps, lv_vs, lv_ps, features)

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

def check_conv(conv, phase, n_cycle):
	if conv:
		cp = ph_counter(phase)
		if cp[3] < n_cycle + 1:
			conv = 0

	return conv

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