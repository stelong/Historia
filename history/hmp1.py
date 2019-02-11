import abstracthm as ahm
import diversipy as dp
from emulator import gp
import numpy as np
import pickle
import sys
from mech import scan_logfile as slf
from utils import mech_out as glv
from utils import design_tools as dt

def diff(A, B):
	B = set(B)
	return [item for item in A if item not in B]

def main():
	rat = 'sham'
	cutoff = 3.0
	maxno = 1
	waveno = int(sys.argv[1])
	active_out_feat = 2

	Wp = ahm.Wave([], cutoff)
	Wp.load('waves/' + rat + '/wave_' + str(waveno-1) + '.pkl')
	N = 256
	X = dp.subset.psa_select(Wp.NROY, N)
	with open('data/' + rat + '/w' + str(waveno) + '_in.txt', 'w') as f:
		np.savetxt(f, X, fmt='%f')
	f.close()

	dt.putlab(X, 'options/' + rat + '/w' + str(waveno) + '_opt.txt')

	l = []
	for i in range(N):
		l.append(np.where(Wp.NROY == X[i, :])[0][0])
	nl = diff(range(Wp.NROY.shape[0]), l)
	tests = Wp.NROY[nl, :]
	with open('data/' + rat + '/w' + str(waveno) + '_tests.txt', 'w') as f:
		np.savetxt(f, tests, fmt='%f')
	f.close()

# ------------------------

if __name__ == '__main__':
	main()