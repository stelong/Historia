from ep import solver as s
from ep.models import Gattoni_SHAM6Hz as sham
from ep.models import Gattoni_AB6Hz as ab
from utils import ep_out as e
from utils import design_tools as des
from classifier import svm as c
import numpy as np
import matplotlib.pyplot as plt
import time

def main():

	# n = 10
	# p0 = sham.initParams()
	# E = np.array([[50, 50], [50, 50], [50, 50], [50, 50], [50, 50], [50, 50], [50, 50], [50, 50], [50, 50], [50, 50], [50, 50]])
	# H = des.lhd(p0, E, n)

	# lista = []
	# B = np.zeros((1, 4), dtype=float)
	# for i in range(n):
	# 	print(i)
	# 	S = s.EPSolution(sham)
	# 	S.run2sc(H[i, :])
	# 	S.plot_ca(scene='show')
		# C = e.PhenCalcium(S.t, S.ca)
		# C.fit()
		# C.get_biomarkers()
		# if C.conv:
		# 	lista.append(i)
		# 	B = np.vstack((B, np.asarray(C.a1)))

	# P = np.copy(H[lista, :])
	# F = np.copy(B[1:, :])

	# with open('inputs.txt', 'w') as f:
	# 	np.savetxt(f, P, fmt='%f')
	# f.close()

	# with open('outputs.txt', 'w') as f:
	# 	np.savetxt(f, F, fmt='%f')
	# f.close()

	# ---------------------------------------

	start_time = time.time()
	p0 = ab.initParams()
	S = s.EPSolution(ab)
	S.run2sc(p0)
	elapsed_time = time.time() - start_time
	print(elapsed_time)

	# ---------------------------------------

	# j = 1
	# with open('ciao.txt', 'w') as f:
	# 	for i in range(100):
	# 		C.a[j] = C.a1[j] - 0.01*(i+1)*C.a1[j]
	# 		C.get_biomarkers()
	# 		if C.conv:
	# 			C.build_ca()
	# 			C.print_ca(f, i+1)
	# 		else:
	# 			break
	# f.close()

	# ---------------------------------------

	# j = 0
	# converged = []
	# for i in range(100):
	# 	C.a[j] = C.a1[j] + 0.01*(i+1)*C.a1[j]
	# 	C.get_biomarkers()
	# 	if C.conv:
	# 		converged.append(i)
	# 	else:
	# 		break

	# print(converged)

	# for i in converged:
	# 	C.a[j] = C.a1[j] + 0.01*(i+1)*C.a1[j]
	# 	C.get_biomarkers()
	# 	if i == converged[-1]:
	# 		C.plot(scene='show')
	# 	else:
	# 		C.plot()

	# SHAM:       -            +
	# 0: [0,...,99] , [0,...,99] 
	# 1: [0,...,80] , [0,...,99]
	# 2: [0,...,65] , [0,...,99]
	# 3: [0,...,98] , [0,...,99]
	#
	# a1 = [0.41766805  2.23158618  0.02263841 16.71244691]
	# bio = [0.41766805  1.38518875 47.48711431 24.79500712]

	# AB:         -            +
	# 0: [0,...,99] , [0,...,99] 
	# 1: [0,...,92] , [0,...,99]
	# 2: [0,...,69] , [0,...,99]
	# 3: [0,...,98] , [0,...,99]
	#
	# a1 = [0.11361605  1.74023038  0.02455431 13.46650655]
	# bio = [0.11361605  0.99557524 42.96436418 21.27732917]

#-------------------------

if __name__ == "__main__":
    main()