import numpy as np
from mech import scan_logfile as slf
from utils import mech_out as glv

def looknsave_all(name):
	folder = 'data/' + name 
	file_name = name + '_inputs.txt'
	vc = []
	with open(file_name, 'w') as f:
		for k in range(4):
			s = folder + '/options' + str(k+1) + '/'
			M = np.zeros(shape=(1, 8), dtype=float)
			for j in range(200):
				tag = s + str(j+1) + '/output_log.txt'
				S = slf.MECHSolution(tag)
				S.extract_loginfo()
				M = np.vstack((M, np.asarray(S.p)))
				V = glv.LeftVentricle()
				V.get_lvfeatures(S, 4)
				vc.append(V.conv)
			np.savetxt(f, M[1:, :], fmt='%f')

	file_name = name + '_outputs.txt'
	with open(file_name, 'w') as f:
		np.savetxt(f, vc, fmt='%d')

	return

def looknsave_conly(name, suffix):
	folder = 'data/' + name 
	file_name = name + '_conly_' + suffix + '.txt'

	with open(file_name, 'w') as f:
		for k in range(4):
			s = folder + '/options' + str(k+1) + '/'
			if suffix == 'inputs':
				dim = 8
			else:
				dim = 11
			M = np.zeros(shape=(1, dim), dtype=float)
			for j in range(200):
				tag = s + str(j+1) + '/output_log.txt'
				S = slf.MECHSolution(tag)
				S.extract_loginfo()
				V = glv.LeftVentricle()
				V.get_lvfeatures(S, 4)
				if V.conv:
					if suffix == 'inputs':
						M = np.vstack((M, np.asarray(S.p)))
					else:
						M = np.vstack((M, np.asarray(V.f)))
			np.savetxt(f, M[1:, :], fmt='%f')

	return

def main():
	name = 'SHAM_20'

	looknsave_conly(name, 'inputs')
	looknsave_conly(name, 'outputs')
	looknsave_all(name)

# ------------------------

if __name__ == '__main__':
	main()