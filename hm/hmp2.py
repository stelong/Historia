import abstracthm as ahm
from Historia.emulator import gp
from Historia.shared import design_utils as desu
from Historia.shared import mech_utils as mecu
from Historia.shared import train_utils as trut
import numpy as np
import sys

def add_points(n_points, NROY, zs, vs, rat, cutoff, maxno, waveno, active_out_feat):
	while NROY.shape[0] < n_points:
		I = desu.get_minmax(NROY)
		SCALE = 0.1*np.array([I[i, 1]-I[i, 0] for i in range(NROY.shape[1])])

		s = 'trained/emulators/' + rat + '/w' + str(waveno) + '/emul'
		emulator = []
		for i in active_out_feat:
			emul = gp.GPEmul()
			emul.load(s + '_' + str(i))
			emulator.append(emul)

		minmax = desu.get_minmax(emulator[0].X)
		minlist = minmax[:, 0]
		maxlist = minmax[:, 1]

		temp = np.random.normal(loc=NROY, scale=SCALE)
		while True:
			l = []
			for i in range(temp.shape[0]):
				diff1 = temp[i, :] - minlist
				diff2 = maxlist - temp[i, :]
				if np.sum(np.sign(diff1)) != temp.shape[1] or np.sum(np.sign(diff2)) != temp.shape[1]:
					l.append(i)
			if l:
				temp[l, :] = np.random.normal(loc=NROY[l, :], scale=SCALE)
			else:
				break

		AHM = []
		for i in range(len(active_out_feat)):
			AHM.append(ahm.sclEmulator(emulator[i]))

		subwave = []
		for i in range(len(active_out_feat)):
			subwave.append(ahm.Subwave(AHM[i], zs[i], vs[i], name='y['+str(active_out_feat[i])+']'))

		W = ahm.Wave(subwaves=subwave, cm=cutoff, maxno=maxno)

		I, mI = W.calcImp(temp)
		NIMP, _, _ = W.findNIMP(I, mI)    
		NEW_TESTS = temp[NIMP]
		NROY = np.vstack((NROY, NEW_TESTS))

	return NROY

def main():
	rat = sys.argv[1]
	cutoff = 5.0
	maxno = 1
	waveno = int(sys.argv[2])
	active_out_feat = [0, 1, 4, 5, 7, 8, 11, 12]

	s1 = 'match/' + rat + '/'
	zs = desu.read_txt(s1 + 'zs', 'float64')[active_out_feat]
	vs = desu.read_txt(s1 + 'vs', 'float64')[active_out_feat]

	print('\n=== Extracting LV features from simulated points...')

	s2_in = 'output/' + rat + '/w' + str(waveno) + '/'
	dim = 256
	nc = 4
	
	if rat == 'sham':
		ibc = 0.9748959188096324
	else:
		ibc = 0.7547397589738333

	s2_out = 'data/' + rat + '/w' + str(waveno)
	mecu.extract_features_xhm(s2_in, dim, s2_out, nc, ibc)

	print('\n=== Building the new training set for the GPs...')

	s3 = 'data/' + rat + '/w' + str(waveno)
	X = desu.read_txt(s3 + '_in', 'float64')
	Y = desu.read_txt(s3 + '_out', 'float64')

	for i in range(waveno-1, 0, -1):
		X = np.vstack((desu.read_txt('data/' + rat + '/w' + str(i) + '_in', 'float64'), X))

	for i in range(waveno-1, 0, -1):
		Y = np.vstack((desu.read_txt('data/' + rat + '/w' + str(i) + '_out', 'float64'), Y))

	print('\n=== Training the GPs...')

	s4 = 'trained/emulators/' + rat + '/w' + str(waveno) + '/emul'
	trut.train_emulator(X, Y, active_out_feat, s4, feats='single')

	emulator = []
	for i in active_out_feat:
		emul = gp.GPEmul()
		emul.load(s4 + '_' + str(i))
		emulator.append(emul)

	print('\n=== Building new subwaves...')

	AHM = []
	for i in range(len(active_out_feat)):
		AHM.append(ahm.sclEmulator(emulator[i]))

	subwave = []
	for i in range(len(active_out_feat)):
		subwave.append(ahm.Subwave(AHM[i], zs[i], vs[i], name='y['+str(active_out_feat[i])+']'))

	print('\n=== Loading NROY...')

	s5 = 'data/' + rat + '/w' + str(waveno) + '_nimp_f_tests'
	tests = desu.read_txt(s5, 'float64')

	print('\n=== Adding new non-implausible points to NROY...')

	N = 50000
	tests = add_points(N, tests, zs, vs, rat, cutoff+0.5, maxno, waveno-1, active_out_feat)

	s7 = 'data/' + rat + '/w' + str(waveno) + '_tests'
	desu.write_txt(tests, '%f', s7)

	print('\n=== Running History Matching: wave{}...'.format(waveno))

	W = ahm.Wave(subwaves=subwave, cm=cutoff, maxno=maxno, tests=tests)
	W.findNROY()
	W.save('waves/' + rat + '/wave_' + str(waveno) + '.pkl')

	print('\n=== Done. Simulation completed.')

if __name__ == '__main__':
	main()