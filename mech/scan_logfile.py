import re

class MECHSolution:
	"""This class implements the solution of Land's 3D rat heart mechanics model*.
	*https://physoc.onlinelibrary.wiley.com/doi/full/10.1113/jphysiol.2012.231928
	"""
	def __init__(self, tag):
		"""Plug-in the logfile.
		Arg:
			tag: string representing the absolute path to the logfile.
		"""
		self.tag = tag
		self.conv = 0
		self.t = []
		self.phase = []
		self.lv_v = []
		self.lv_p = []
		self.p = []

	def extract_loginfo(self):
		"""Scan the logfile searching for: parameters, time, LV volume, LV pressure, cardiac cycle phases vectors.
		"""
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
		par8 = re.compile('[-]beta1\s+=\s+(\S+)$')
		par9 = re.compile('[-]Tref\s+=\s+(\S+)$')

		c1 = p = ap = z = ca50 = kxb = koff = beta1 = Tref = lv_v0 = conv = 0

		f = open(self.tag, 'r')
		line = f.readlines()
		n = len(line)

		lv_dv = []

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
				beta1 = float(mp8.groups()[0])
				break

		while i < n:
			mp9 = re.search(par9, line[i])
			if mp9 == None:
				i = i + 1
			else:
				Tref = float(mp9.groups()[0])
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
				self.t.append(float(m1.group()))

			for m2 in re.finditer(p2, line[i]):
				self.phase.append(int(m2.groups()[0]))
				lv_dv.append(float(m2.groups()[1]))
				self.lv_p.append(float(m2.groups()[2]))

			if re.match(p3, line[i]) != None:
				self.conv = 1
				break

			i = i + 1

		self.p = [p, ap, z, c1, ca50, kxb, koff, beta1, Tref]
		self.lv_v = [x + lv_v0 for x in lv_dv]

		f.close()