import abstracthm as ahm
import diversipy as dp
from emulator import gp
import numpy as np
import pickle
import sys
from mech import scan_logfile as slf
from utils import mech_out as glv
from utils import design_tools as dt

def extract_features(path):
	l = []	
	F = np.zeros(shape=(1, 11), dtype=float)
	for i in range(256):
		tag = path + str(i+1) + '/output_log.txt'
		S = slf.MECHSolution(tag)
		S.extract_loginfo()
		V = glv.LeftVentricle()
		V.get_lvfeatures(S, 2)
		if V.conv:
			l.append(i)
			F = np.vstack((F, np.asarray(V.f)))
	return l, F[1:, :]

def main():
	rat = 'sham'
	cutoff = 3.0
	maxno = 1
	waveno = int(sys.argv[1])
	active_out_feat = 2

	if rat == 'sham':
		zs = [508.8, 154.6]
		vs = [1521.85, 273.27]
	else:
		zs = [466.5, 125.6]
		vs = [1376.41, 547.56]

	print('\n[4]=== Extracting LV features from simulated points...')
	path = 'output/' + rat + '/'
	inds, Y = extract_features(path)
	with open('data/' + rat + '/w' + str(waveno) + '_out.txt', 'w') as f:
		np.savetxt(f, Y, fmt='%f')
	f.close()

	print('\n[5]=== Building the new training set for the GPs...')
	X_ = np.loadtxt('data/' + rat + '/w' + str(waveno) + '_allin.txt', dtype=float)
	with open('data/' + rat + '/w' + str(waveno) + '_in.txt', 'w') as f:
		np.savetxt(f, X_[inds, :], fmt='%f')
	f.close()
	X = X_[inds, :]
	for i in range(waveno-1, 0, -1):
		X = np.vstack((np.loadtxt('data/' + rat + '/w' + str(i) + '_in.txt', dtype=float), X))

	for i in range(waveno-1, 0, -1):
		Y = np.vstack((np.loadtxt('data/' + rat + '/w' + str(i) + '_out.txt', dtype=float), Y))

	print('\n[6]=== Training the GPs...')
	emulator = []
	for i in range(active_out_feat):
		emul = gp.GPEmul()
		emul.fit(X, Y[:, i])
		emul.save('trained_emulators/' + rat + '/w' + str(waveno) + '_emul' + str(i+1))
		emulator.append(emul)

	print('\n[7]=== Building new subwaves...')
	AHM = []
	for i in range(active_out_feat):
		AHM.append(ahm.sclEmulator(emulator[i]))

	subwave = []
	for i in range(active_out_feat):
		subwave.append(ahm.Subwave(AHM[i], zs[i], vs[i], name='y['+str(i)+']'))

	print('\n[8]=== Running HM: wave{}...'.format(waveno))
	tests = np.loadtxt('data/' + rat + '/w' + str(waveno) + '_tests.txt', dtype=float)
	W = ahm.Wave(subwaves=subwave, cm=cutoff, maxno=maxno, tests=tests)
	W.findNROY()
	W.save('waves/' + rat + '/wave_' + str(waveno) + '.pkl')

	print('\n=== Done. Simulation completed.')

# ------------------------

if __name__ == '__main__':
	main()
