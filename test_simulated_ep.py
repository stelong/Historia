import numpy as np

def get_biomarkers(name):
	X = np.loadtxt('data/' + name + '/inputs.txt', dtype=float)
	Y = np.loadtxt('data/' + name + '/outputs.txt', dtype=float)

	sample_dim = X.shape[0]
	in_dim = X.shape[1]
	out_dim = Y.shape[1]
	bio_dim = 4

	l = []
	for i in range(sample_dim):
		if np.sum(Y[i, :]) == 0:
			l.append(0)
		else:
			l.append(1)

	for i in range(sample_dim):
		if l[i]:
			y = Y[i, :]

			imax = np.argmax(y)
			if y[imax] <= y[0] or y[imax] <= y[-1]:
				l[i] = 0

	t = np.linspace(0, 170, 171)

	P = np.zeros((1, in_dim), dtype=float)
	B = np.zeros((1, bio_dim), dtype=float)
	for i in range(sample_dim):
		if l[i]:
			P = np.vstack((P, X[i, :]))
			y = Y[i, :]

			DCa = y[0]
			imax = np.argmax(y)
			PCa = y[imax]
			TP = t[imax]

			j = imax + 1
			while j < 171:
				if y[j] <= 0.5*(PCa + DCa):
					RT50 = t[j] - TP
					break
				j += 1

			B = np.vstack((B, np.array([DCa, PCa, RT50, TP])))

	with open(name + '_inputs.txt', 'w') as f:
		np.savetxt(f, X, fmt='%f')
	f.close()

	with open(name + '_outputs.txt', 'w') as f:
		np.savetxt(f, l, fmt='%d')
	f.close()

	with open(name + '_conly_inputs.txt', 'w') as f:
		np.savetxt(f, P[1:, :], fmt='%f')
	f.close()

	with open(name + '_conly_outputs.txt', 'w') as f:
		np.savetxt(f, B[1:, :], fmt='%f')
	f.close()

def main():
	name = 'EP_3'
	get_biomarkers(name)

#-------------------------

if __name__ == "__main__":
	main()