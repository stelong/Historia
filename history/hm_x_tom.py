import abstracthm as ahm
import diversipy as dp
import emulator as gp
import numpy as np
import pickle
import sys
from mech import scan_logfile as slf
from utils import mech_out as glv
from utils import design_tools as dt

def diff(A, B):
	B = set(B)
	return [item for item in A if item not in B]

def extract_features(path):	
	F = np.zeros(shape=(1, 11), dtype=float)
	for i in range(100):
		tag = path + str(i+1) + '/output_log.txt'
		S = slf.MECHSolution(tag)
		S.extract_loginfo()
		V = glv.LeftVentricle()
		V.get_lvfeatures(S, 2)
		if V.conv:
			F = np.vstack((F, np.asarray(V.f)))
	np.savetxt(f, F[1:, :], fmt='%f')
	return F

def main():
	rat = 'ab'
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

	Wp = ahm.Wave([], cutoff)
	Wp.load('waves/' + rat + '/wave_' + str(waveno-1) + '.pkl')
	N = 80
	X = dp.subset.psa_select(Wp.NROY, N)
	l = []
	for i in range(N):
		l.append(np.where(Wp.NROY == X[i])[0][0])
	nl = diff(range(Wp.NROY.shape[0]), l)
	tests = Wp.NROY[nl, :]

	dt.putlab(X, 'options.txt')
	out_path = rat + '_out/'

	# ------------ RUN MECHANICS -------------
	# 					.
	#					.
	#					.
	# ----------------------------------------

	path = out_path + 'options/'
	Y = extract_features(path)

	emulator = []
	for i in range(active_out_feat):
		emul = gp.GPEmul()
		emul.fit(X, Y[:, i])
		emul.save('trained_emulators/mech/' + rat + '/w' + str(waveno) + '_emul_' + str(i+1))
		emulator.append(emul)

	AHM = []
	for i in range(active_out_feat):
		AHM.append(ahm.sclEmulator(emulator[i]))

	subwave = []
	for i in range(active_out_feat):
		subwave.append(ahm.Subwave(AHM[i], zs[i], vs[i], name='y['+str(i)+']'))

	W = ahm.Wave(subwaves=subwave, cm=cutoff, maxno=maxno, tests=tests)
	W.findNROY()
	W.save('waves/' + rat + '/wave_' + str(waveno) + '.pkl')

# ------------------------

if __name__ == '__main__':
	main()