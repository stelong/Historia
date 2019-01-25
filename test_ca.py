import numpy as np
import matplotlib.pyplot as plt
from utils.math_tools import der
np.set_printoptions(formatter={"float_kind": lambda x: "%g" % x})

def main():
	X = np.loadtxt('data/EP_3/ep_3_in.txt', dtype=float)
	Y = np.loadtxt('data/EP_3/ep_3_out.txt', dtype=float)
	l = np.loadtxt('data/EP_3/ep_3_conv.txt', dtype=int)

	t = np.linspace(0,170,171)
	for i in range(1000):
		if l[i]:
			y = Y[i, :]

			imax = np.argmax(y)
			if y[imax] <= y[0] or y[imax] <= y[-1]:
				l[i] == 0

	P = np.zeros((1, 11), dtype=float)
	B = np.zeros((1, 4), dtype=float)
	for i in range(1000):
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

	with open('EP_3_conly_inputs.txt', 'w') as f:
		np.savetxt(f, P[1:, :], fmt='%f')
	f.close()

	with open('EP_3_conly_outputs.txt', 'w') as f:
		np.savetxt(f, B[1:, :], fmt='%f')
	f.close()


#-------------------------

if __name__ == "__main__":
	main()