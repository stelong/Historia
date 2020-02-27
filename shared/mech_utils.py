from Historia.mech import mech_out as glv
from Historia.mech import scan_logfile as slf
from Historia.shared import design_utils as desu
import numpy as np

def extract_features(path_in, dim, path_out, nc, ibc):
	XA = np.zeros((0, 8), dtype=float)
	X = np.zeros((0, 8), dtype=float)
	YA = []
	Y = np.zeros((0, 13), dtype=float)
	l = []
	for i in range(dim):
		tag = path_in + str(i+1) + '/output_log.txt'
		S = slf.MECHSolution(tag)
		try:
			S.extract_loginfo()
		except FileNotFoundError:
			print('\n=== [Index: {}] Logfile not found!'.format(i+1))
			continue

		XA = np.vstack((XA, np.array(S.p)))

		RS = glv.LeftVentricle()
		RS.get_lvfeatures(S, nc, ibc)

		YA.append(RS.conv)
		if RS.conv:
			l.append(i+1)
			X = np.vstack((X, np.array(S.p)))
			Y = np.vstack((Y, np.array(RS.f)))

	desu.write_txt(XA, '%f', path_out + '_inputs')
	desu.write_txt(X, '%f', path_out + '_conly_inputs')
	desu.write_txt(YA, '%d', path_out + '_outputs')
	desu.write_txt(Y, '%f', path_out + '_conly_outputs')
	desu.write_txt(l, '%d', path_out + '_lconv')
	return