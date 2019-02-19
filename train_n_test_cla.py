from classifier import svm
import numpy as np
import pickle

def main():
	cla = svm.SVMCla()

	inFile1 = 'data/SHAM_20/SHAM_20_inputs.txt'
	inFile2 = 'data/SHAM_20/SHAM_20_outputs.txt'
	inFile3 = 'data/SHAM_20/a1.txt'
	
	mech = np.loadtxt(inFile1, dtype=float)
	a1 = np.loadtxt(inFile3, dtype=float)

	X = np.hstack((a1, mech))
	y = np.loadtxt(inFile2, dtype=float)

	cla.fit(X, y)
	cla.save('cla')

	H = cla.hlc_sample(800)
	T = [M for M in np.hsplit(H, 3)]

	with open('a1.txt', 'w') as f:
		np.savetxt(f, T[0], fmt='%f')

	with open('mech.txt', 'w') as f:
		np.savetxt(f, np.hstack((T[1], T[2])), fmt='%f')

#-------------------------

if __name__ == "__main__":
    main()