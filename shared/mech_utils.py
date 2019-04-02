from Historia.shared import design_utils as desu
from Historia.mech import mech_out as glv
from Historia.mech import scan_logfile as slf
import numpy as np

def extract_features(path_in, dim, path_out):
	nc = 2
	XA = np.zeros(shape=(1, 8), dtype=float)
	X = np.zeros(shape=(1, 8), dtype=float)

	YA = []
	Y = np.zeros(shape=(1, 11), dtype=float)

	for i in range(dim):
		tag = path_in + str(i+1) + '/output_log.txt'
		S = slf.MECHSolution(tag)
		try:
			S.extract_loginfo()
		except FileNotFoundError:
			print('\n=== [Index: {}] Logfile not found! Don\'t worry about it, I will fix everything for you.'.format(i))
			continue

		XA = np.vstack((XA, np.asarray(S.p)))

		RS = glv.LeftVentricle()
		RS.get_lvfeatures(S, nc)
		YA.append(RS.conv)

		if RS.conv:
			X = np.vstack((X, np.asarray(S.p)))
			Y = np.vstack((Y, np.asarray(RS.f)))

	desu.write_txt(XA[1:], '%f', path_out + '_inputs')
	desu.write_txt(X[1:], '%f', path_out + '_conly_inputs')
	desu.write_txt(YA, '%d', path_out + '_outputs')
	desu.write_txt(Y[1:], '%f', path_out + '_conly_outputs')
	return

def extract_features_xhm(path_in, dim, path_out):
	nc = 2
	X = np.zeros(shape=(1, 8), dtype=float)
	Y = np.zeros(shape=(1, 11), dtype=float)

	for i in range(dim):
		tag = path_in + str(i+1) + '/output_log.txt'
		S = slf.MECHSolution(tag)
		try:
			S.extract_loginfo()
		except FileNotFoundError:
			continue

		RS = glv.LeftVentricle()
		RS.get_lvfeatures(S, nc)

		if RS.conv:
			X = np.vstack((X, np.asarray(S.p)))
			Y = np.vstack((Y, np.asarray(RS.f)))

	desu.write_txt(X[1:], '%f', path_out + '_in')
	desu.write_txt(Y[1:], '%f', path_out + '_out')
	return
