from Historia.shared import design_utils as desu
from Historia.mech import mech_out as glv
from Historia.mech import scan_logfile as slf
import numpy as np

def extract_features(path_in, dim, path_out):
	XA = np.zeros(shape=(1, 8), dtype=float)
	X = np.zeros(shape=(1, 8), dtype=float)

	YA = []
	Y = np.zeros(shape=(1, 11), dtype=float)

	for i in range(dim):
		tag = path_in + str(i+1) + '/output_log.txt'
		try:
			S = slf.MECHSolution(tag)
		except FileNotFoundError:
			print('\n=== Logfile not found! Don''t worry about it, I will fix everything for you.')
			continue

		S.extract_loginfo()
		XA = np.vstack((XA, np.asarray(S.p)))

		V = glv.LeftVentricle()
		V.get_lvfeatures(S, 2)
		YA.append(V.conv)

		if V.conv:
			X = np.vstack((X, np.asarray(S.p)))
			Y = np.vstack((Y, np.asarray(V.f)))

	desu.write_txt(XA[1:], '%f', path_out + '_inputs')
	desu.write_txt(X[1:], '%f', path_out + '_conly_inputs')
	desu.write_txt(YA, '%d', path_out + '_outputs')
	desu.write_txt(Y[1:], '%f', path_out + '_conly_outputs')
	return
