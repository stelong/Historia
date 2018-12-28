import re

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

	for line in open(tag, 'r'):
		if lv_v0 == 0:
			m0 = re.search(p0, line)
			if m0 != None:
				lv_v0 = float(m0.groups()[0])

			elif Tref == 0:
				mp8 = re.search(par8, line)
				if mp8 != None:
					Tref = float(mp8.groups()[0])
				elif koff == 0:
					mp7 = re.search(par7, line)
					if mp7 != None:
						koff = float(mp7.groups()[0])
					elif kxb == 0:
						mp6 = re.search(par6, line)
						if mp6 != None:
							kxb = float(mp6.groups()[0])
						elif ca50 == 0:
							mp5 = re.search(par5, line)
							if mp5 != None:
								ca50 = float(mp5.groups()[0])
							elif z == 0:
								mp4 = re.search(par4, line)
								if mp4 != None:
									z = float(mp4.groups()[0])
								elif ap == 0:
									mp3 = re.search(par3, line)
									if mp3 != None:
										ap = float(mp3.groups()[0])
									elif p == 0:
										mp2 = re.search(par2, line)
										if mp2 != None:
											p = float(mp2.groups()[0])
										elif c1 == 0:
											mp1 = re.search(par1, line)
											if mp1 != None:
												c1 = float(mp1.groups()[0])

		for m1 in re.finditer(p1, line):
			t.append(float(m1.group()))

		for m2 in re.finditer(p2, line):
			phase.append(int(m2.groups()[0]))
			lv_dv.append(float(m2.groups()[1]))
			lv_p.append(float(m2.groups()[2]))

		if re.match(p3, line) != None:
			conv = 1

	parameters = [c1, p, ap, z, ca50, kxb, koff, Tref]

	lv_v = [x + lv_v0 for x in lv_dv]

	return LogValues(conv, t, phase, lv_v, lv_p, parameters)

#--------