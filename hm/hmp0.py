import abstracthm as ahm
from Historia.emulator import gp
from Historia.shared import design_utils as desu
from Historia.shared import train_utils as trut
import numpy as np
import sys

def main():
	rat = sys.argv[1]
	cutoff = 5.5
	maxno = 1
	waveno = 1
	active_out_feat = [0, 1, 2, 4, 5, 7, 8, 11, 12]

	s1 = 'match/' + rat + '/'
	zs = desu.read_txt(s1 + 'zs', 'float64')[active_out_feat]
	vs = desu.read_txt(s1 + 'vs', 'float64')[active_out_feat]

	s2 = 'data/' + rat + '/w' + str(waveno)
	X = desu.read_txt(s2 + '_in', 'float64')
	Y = desu.read_txt(s2 + '_out', 'float64')

	s3 = 'trained/emulators/' + rat + '/w' + str(waveno) + '/emul'
	trut.train_emulator(X, Y, list(range(13)), s3, feats='single')

	emulator = []
	for i in active_out_feat:
		emul = gp.GPEmul()
		emul.load(s3 + '_' + str(i))
		emulator.append(emul)

	AHM = []
	for i in range(len(active_out_feat)):
		AHM.append(ahm.sclEmulator(emulator[i]))

	subwave = []
	for i in range(len(active_out_feat)):
		subwave.append(ahm.Subwave(AHM[i], zs[i], vs[i], name='y['+str(i)+']'))

	I = desu.get_minmax(emulator[0].X)

	N = 200000
	q = 10000
	n = int(N/q)

	tests = np.zeros((0, emulator[0].X.shape[1]), dtype=float)
	for i in range(n):
		H = desu.lhd_int(I, q)
		tests = np.vstack((tests, H))

	W = ahm.Wave(subwaves=subwave, cm=cutoff, maxno=maxno, tests=tests)
	W.findNROY()
	W.save('waves/' + rat + '/wave_' + str(waveno) + '.pkl')

if __name__ == '__main__':
	main()