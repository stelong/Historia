import abstracthm as ahm
import diversipy as dp
from Historia.shared import design_utils as desu
from Historia.shared import indices_utils as indu
import numpy as np
import sys

def main():
	rat = sys.argv[1]
	cutoff = 5.5
	maxno = 1
	waveno = int(sys.argv[2])
	active_out_feat = [0, 1, 2, 4, 5, 7, 8, 11, 12]

	print('\n=== Extracting points to be simulated from NROY...')

	Wp = ahm.Wave([], cutoff)
	Wp.load('waves/' + rat + '/wave_' + str(waveno-1) + '.pkl')
	X = Wp.NROY
	
	cont = 0
	if X.shape[0] < 256:
		print('\n=== Look, I think you should stop: #NINP = {}'.format(X.shape[0]))
		SX = X
	else:
		cont = 1
		n = 256
		SX = dp.subset.psa_select(X, n)

	s1 = 'data/' + rat + '/w' + str(waveno) + '_nimp'
	desu.write_txt(SX, '%f', s1)

	print('\n=== Creating option file for the simulation...')

	s2 = 'options/' + rat + '/w' + str(waveno) + '_opt'
	desu.putlab(SX, s2)

	if cont:
		print('\n=== Saving tests points for wave{}...'.format(waveno))

		_, nl = indu.whereq_whernot(X, SX)
		tests = X[nl]

		s3 = 'data/' + rat + '/w' + str(waveno) + '_nimp_f_tests'
		desu.write_txt(tests, '%f', s3)

		print('\n=== Done. Now run the mechanics.')

	else:
		print('\n=== History Matching concluded. Simulate the last remaining points.')

if __name__ == '__main__':
	main()