import re


class LogValues:
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

	for line in open(tag, 'r'):
		for m1 in re.finditer(p1, line):
			t.append(float(m1.group()))

		for m2 in re.finditer(p2, line):
			phase.append(int(m2.groups()[0]))
			lv_dv.append(float(m2.groups()[1]))
			lv_p.append(float(m2.groups()[2]))

		if re.match(p3, line) is not None:
			conv = 1

		if lv_v0 != 0:
			continue

		m0 = re.search(p0, line)
		if m0 is not None:
			lv_v0 = float(m0.groups()[0])
		if Tref != 0:
			continue
		mp8 = re.search(par8, line)
		if mp8 is not None:
			Tref = float(mp8.groups()[0])
		if koff != 0:
			continue
		mp7 = re.search(par7, line)
		if mp7 is not None:
			koff = float(mp7.groups()[0])
		if kxb != 0:
			continue
		mp6 = re.search(par6, line)
		if mp6 is not None:
			kxb = float(mp6.groups()[0])
		if ca50 != 0:
			continue
		mp5 = re.search(par5, line)
		if mp5 is not None:
			ca50 = float(mp5.groups()[0])
		if z != 0:
			continue
		mp4 = re.search(par4, line)
		if mp4 is not None:
			z = float(mp4.groups()[0])
		if ap != 0:
			continue
		mp3 = re.search(par3, line)
		if mp3 is not None:
			ap = float(mp3.groups()[0])
		if p != 0:
			continue
		mp2 = re.search(par2, line)
		if mp2 is not None:
			p = float(mp2.groups()[0])
		if c1 != 0:
			continue
		mp1 = re.search(par1, line)
		if mp1 is not None:
			c1 = float(mp1.groups()[0])

	parameters = [c1, p, ap, z, ca50, kxb, koff, Tref]

	lv_v = [x + lv_v0 for x in lv_dv]

	return LogValues(conv, t, phase, lv_v, lv_p, parameters)
