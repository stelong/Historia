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

	return M1, M3