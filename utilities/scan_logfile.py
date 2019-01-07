import re
import numpy as np

class LogValues():
	def __init__(self, conv, t, phase, lv_v, lv_p, p):
		self.conv = conv
		self.t = t
		self.phase = phase
		self.lv_v = lv_v
		self.lv_p = lv_p
		self.p = p

class LeftVentricleFeatures():
	def __init__(self, edv, esv, ef, ict, et, irt, tdiast, peakp, tpeak, maxdp, mindp):
		self.edv = edv
		self.esv = esv
		self.ef = ef
		self.ict = ict
		self.et = et
		self.irt = irt
		self.tdiast = tdiast
		self.peakp = peakp
		self.tpeak = tpeak
		self.maxdp = maxdp
		self.mindp = mindp

def extract_info(tag):
	p0 = re.compile('LV\sCavity\svolume\s=\s(\d+\.\d+)')
	p1 = re.compile('(?<=\*\sT\s->\s)\d+\.\d+')
	p2 = re.compile('Phase\s=\s([1-4]).*?Volume\s=\s([-]?\d+\.\d+).*?LVP\s=\s(\d+\.\d+)')
	p3 = re.compile('[=]+\sSimulation\scompleted\ssuccessfully[\s\S]+')

	par1 = re.compile('[-]c1\s+=\s+(\S+)$')
	par2 = re.compile('[-]p\s+=\s+(\S+)$')
	par3 = re.compile('[-]ap\s+=\s+(\S+)$')
	par4 = re.compile('[-]z\s+=\s+(\S+)$')
	par5 = re.compile('[-]ca50\s+=\s+(\S+)$')
	par6 = re.compile('[-]kxb\s+=\s+(\S+)$')
	par7 = re.compile('[-]koff\s+=\s+(\S+)$')
	par8 = re.compile('[-]Tref\s+=\s+(\S+)$')

	c1 = p = ap = z = ca50 = kxb = koff = Tref = lv_v0 = conv = 0

	t = []
	phase = []
	lv_dv = []
	lv_p = []

	f = open(tag, 'r')
	line = f.readlines()
	n = len(line)

	i = 0
	while i < n:
		mp1 = re.search(par1, line[i])
		if mp1 == None:
			i = i + 1
		else:
			c1 = float(mp1.groups()[0])
			break

	while i < n:
		mp2 = re.search(par2, line[i])
		if mp2 == None:
			i = i + 1
		else:
			p = float(mp2.groups()[0])
			break

	while i < n:
		mp3 = re.search(par3, line[i])
		if mp3 == None:
			i = i + 1
		else:
			ap = float(mp3.groups()[0])
			break

	while i < n:
		mp4 = re.search(par4, line[i])
		if mp4 == None:
			i = i + 1
		else:
			z = float(mp4.groups()[0])
			break

	while i < n:
		mp5 = re.search(par5, line[i])
		if mp5 == None:
			i = i + 1
		else:
			ca50 = float(mp5.groups()[0])
			break

	while i < n:
		mp6 = re.search(par6, line[i])
		if mp6 == None:
			i = i + 1
		else:
			kxb = float(mp6.groups()[0])
			break

	while i < n:
		mp7 = re.search(par7, line[i])
		if mp7 == None:
			i = i + 1
		else:
			koff = float(mp7.groups()[0])
			break

	while i < n:
		mp8 = re.search(par8, line[i])
		if mp8 == None:
			i = i + 1
		else:
			Tref = float(mp8.groups()[0])
			break

	while i < n:
		m0 = re.search(p0, line[i])
		if m0 == None:
			i = i + 1
		else:
			lv_v0 = float(m0.groups()[0])
			break

	while i < n:
		for m1 in re.finditer(p1, line[i]):
			t.append(float(m1.group()))

		for m2 in re.finditer(p2, line[i]):
			phase.append(int(m2.groups()[0]))
			lv_dv.append(float(m2.groups()[1]))
			lv_p.append(float(m2.groups()[2]))

		if re.match(p3, line[i]) != None:
			conv = 1
			break

		i = i + 1

	parameters = [c1, p, ap, z, ca50, kxb, koff, Tref]

	lv_v = [x + lv_v0 for x in lv_dv]

	return LogValues(conv, t, phase, lv_v, lv_p, parameters)

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

def display_allout(t, phase, lv_v, lv_p):
	cp = ph_counter(phase)

	M1 = isl_ranges(np.where(np.asarray(phase) == 1)[0], cp[0])
	M3 = isl_ranges(np.where(np.asarray(phase) == 3)[0], cp[2])

	ind = np.sort(np.concatenate((np.reshape(M1, cp[0]*2), np.reshape(M3, cp[2]*2)), axis=0))
	ind_r = ind[-6:-1]
	interval = list(range(ind_r[0], ind_r[-1]))

	ts = [t[i] for i in interval]
	ps = [phase[i] for i in interval]
	LVP = [lv_p[i] for i in interval]
	LVV = [lv_v[i] for i in interval]

	t1 = t[ind_r[0]]
	t2 = t[ind_r[1]]
	t3 = t[ind_r[2]]
	t4 = t[ind_r[3]]
	t5 = t[ind_r[4]]

	dP = der1(ts, LVP)
	m = max(LVP)
	ind_m = list(LVP).index(m)

	ps1 = list(np.where(np.asarray(ps) == 1)[0])
	lvv1 = [LVV[i] for i in ps1]

	p1 = max(lvv1)        # EDV    (end diastolic volume)
	p2 = min(LVV)         # ESV    (end systolic volume)
	p3 = 100*(p1 - p2)/p1 # EF     (ejection fraction)
	p4 = t2 - t1          # ICT    (isovolumetric contraction time)
	p5 = t3 - t2          # ET     (ejection time)
	p6 = t4 - t3          # IRT    (isovolumetric relaxation time)
	p7 = t5 - t4          # Tdiast (diastolic time)

	q1 = m                  # PeakP (maximum pressure)
	q2 = ts[ind_m] - ts[0]  # Tpeak (time to peak)
	q3 = max(dP)            # maxdP (maximum pressure gradient value) 
	q4 = min(dP)            # mindP (minimum pressure gradient value)

	return LeftVentricleFeatures(p1, p2, p3, p4, p5, p6, p7, q1, q2, q3, q4)

def der1(t, y):
	N = len(t)
	dt = (t[-1] - t[0])/N

	p2 = [-1.0/2.0, 1.0/2.0]
	p4 = [1.0/12.0, -2.0/3.0, 2.0/3.0, -1.0/12.0]
	p6 = [-1.0/60.0, 3.0/20.0, -3.0/4.0, 3.0/4.0, -3.0/20.0, 1.0/60.0]
	p8 = [1.0/280.0, -4.0/105.0, 1.0/5.0, -4.0/5.0, 4.0/5.0, -1.0/5.0, 4.0/105.0, -1.0/280.0]

	dy = np.zeros(shape=(N,), dtype=float)

	dy[0] = ( y[1] - y[0] )/dt
	dy[-1] = ( y[-1] - y[-2] )/dt

	dy[1] = ( p2[0]*y[0] + p2[1]*y[2] )/dt
	dy[-2] = ( p2[0]*y[-3] + p2[1]*y[-1] )/dt

	dy[2] = ( p4[0]*y[0] + p4[1]*y[1] + p4[2]*y[3] + p4[3]*y[4] )/dt
	dy[-3] = ( p4[0]*y[-5] + p4[1]*y[-4] + p4[2]*y[-2] + p4[3]*y[-1] )/dt

	dy[3] = ( p6[0]*y[0] + p6[1]*y[1] + p6[2]*y[2] + p6[3]*y[4] + p6[4]*y[5] + p6[5]*y[6] )/dt
	dy[-4] = ( p6[0]*y[-7] + p6[1]*y[-6] + p6[2]*y[-5] + p6[3]*y[-3] + p6[4]*y[-2] + p6[5]*y[-1] )/dt

	for i in range(4, N-4):
		dy[i] = ( p8[0]*y[i-4] + p8[1]*y[i-3] + p8[2]*y[i-2] + p8[3]*y[i-1] + p8[4]*y[i+1] + p8[5]*y[i+2] + p8[6]*y[i+3] + p8[7]*y[i+4] )/dt

	return dy