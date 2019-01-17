import numpy as np
from mech import scan_logfile as slf
from utils import math_tools as mt
from utils import mech_out as glv

def main():

	# for j in range(50):
	# 	tag = '/home/sl18/Desktop/AB_7/options1/' + str(j+1) + '/output_log.txt'
	# 	M = slf.MECHSolution(tag)
	# 	M.extract_loginfo()

	# 	V = glv.LeftVentricle()
	# 	V.get_lvfeatures(M, 2)
		
	# 	if V.conv:
	# 		V.plot(M)

	# ------------------------------------------------------------------------------

	file_in = 'AB_8_in.txt'
	with open(file_in, 'w') as f1:
		for k in range(4):
			s = 'data/AB_8/options' + str(k+1) + '/'
			P = np.zeros(shape=(1, 7), dtype=float)
		
			for j in range(200):
				tag = s + str(j+1) + '/output_log.txt'
				S = slf.MECHSolution(tag)
				S.extract_loginfo()
				V = glv.LeftVentricle()
				V.get_lvfeatures(S, 2)
				if V.conv:
					P = np.vstack((P, np.asarray(S.p)))
			np.savetxt(f1, P[1:, :], fmt='%f')
	f1.close()							
	
	file_out = 'AB_8_out.txt'
	with open(file_out, 'w') as f2:
		for k in range(4):
			s = 'data/AB_8/options' + str(k+1) + '/'
			M = np.zeros(shape=(1, 11), dtype=float)

			for j in range(200):
				tag = s + str(j+1) + '/output_log.txt'
				S = slf.MECHSolution(tag)
				S.extract_loginfo()
				V = glv.LeftVentricle()
				V.get_lvfeatures(S, 2)
				if V.conv:
					M = np.vstack((M, np.asarray(V.f)))
			np.savetxt(f2, M[1:, :], fmt='%f')
	f2.close()

# ------------------------

if __name__ == '__main__':
	main()