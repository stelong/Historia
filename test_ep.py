from ep.models import Gattoni_6Hz as odesys
from ep.models import SHAM_init as sham
from ep.models import AB_init as ab
from ep import solver as sol
import numpy as np
from utils import ep_out as ep
from utils import design_tools as des
import matplotlib.pyplot as plt
import time
np.set_printoptions(formatter={'all':lambda x: '{:f}'.format(x)})

class TimeCounter:
    def __init__(self, f):
        self.time = 0
        self.f = f

    def __call__(self, *args, **kwargs):
        start = time.time()
        result = self.f(*args, **kwargs)
        elapsed = time.time() - start
        self.time += elapsed
        print(f"Spent {self.time} in {self.f.__name__} so far.")
        return result

@TimeCounter
def main():
	S = sol.EPSolution(odesys, ab)
	p0 = ab.initParams()
	S.run2ss(p0)
	S.plot_calcium(scene='show')

	# bio = ep.A_output(S.ca)
	# plt.figure()
	# plt.plot(S.t, S.ca, c='k', zorder=1)
	# plt.axvline(x=bio[3], c='r', linestyle='dashed', linewidth=0.8, zorder=2)
	# plt.axvline(x=bio[2]+bio[3], c='r', linestyle='dashed', linewidth=0.8, zorder=2)
	# plt.axhline(y=bio[0], c='r', linestyle='dashed', linewidth=0.8, zorder=2)
	# plt.axhline(y=bio[1], c='r', linestyle='dashed', linewidth=0.8, zorder=2)
	# plt.scatter(bio[3], bio[1], c='r', zorder=3)
	# plt.scatter(bio[2]+bio[3], S.ca[int(bio[2]+bio[3])], c='r', zorder=3)
	# plt.xlim(S.t[0], S.t[-1])
	# plt.ylim(0.9*np.min(S.ca), 1.1*np.max(S.ca))
	# frame = plt.gca()
	# frame.axes.get_xaxis().set_visible(False)
	# frame.axes.get_yaxis().set_visible(False)
	# plt.savefig('calcium_biomarkers.pdf', format='pdf', dpi=1000)

	# C = ep.PhenCalcium(S.t, S.ca)
	# C.fit()
	# C.get_bio()
	# C.build_ca()
	# C.check_ca()
	# C.plot(visbio=True, scene='show')

	#--------

	# p0 = C.a1
	# E = np.array([[100, 100], [100, 100], [100, 100], [100, 100]])
	# N = 10
	# H = des.lhd(p0, E, N)

	# l = []
	# nl = []
	# for j in range(N):
	# 	C.a = H[j, :]
	# 	C.get_bio()
	# 	C.build_ca()
	# 	C.check_ca()
	# 	if C.conv:
	# 		l.append(j)
	# 		# C.plot(visbio=False)
	# 	else:
	# 		nl.append(j)
	# 		# C.plot(visbio=False)
	# plt.show()

	#--------

	# p0 = C.a1
	# E = np.array([[100, 100], [100, 100], [100, 100], [100, 100]])
	# N = 800
	# H = des.lhd(p0, E, N)

	# with open('calcium.txt', 'w') as f:
	# 	H = np.zeros((1, 4), dtype=float)
	# 	B = np.zeros((1, 4), dtype=float)
	# 	stim = 0
	# 	while H[1:, :].shape[0] < N:
	# 		Hp = des.lhd(p0, E, N)
	# 		for i in range(N):
	# 			C.a = Hp[i, :]
	# 			C.get_bio()
	# 			C.build_ca()
	# 			C.check_ca()
	# 			if C.conv:
	# 				H = np.vstack((H, Hp[i, :]))
	# 				B = np.vstack((B, np.asarray(C.bio)))
	# 				stim += 1
	# 				print('[i={}, stim={}]'.format(i, stim))
	# 				if stim % 200 == 0:
	# 					pstim = 200
	# 				else:
	# 					pstim = stim % 200 
	# 				C.print_ca(f, pstim)
	# 				if H[1:, :].shape[0] == N:
	# 					break
	# f.close()

	# with open('a1.txt', 'w') as f:
	# 	np.savetxt(f, H[1:, :], fmt='%f')
	# f.close()

	# with open('bio.txt', 'w') as f:
	# 	np.savetxt(f, B[1:, :], fmt='%f')
	# f.close()

	#--------

	# des.divide_et_impera('calcium')
	# des.append_to_header('sham', 'calcium', 1)

	#--------

	# j = 0
	# converged = []
	# for i in range(100):
	# 	C.a[j] = C.a1[j] - 0.01*(i+1)*C.a1[j]
	# 	C.get_bio()
	# 	C.build_ca()
	# 	C.check_ca()
	# 	if C.conv:
	# 		converged.append(i)
	# 	else:
	# 		break

	# print(converged)

	# for i in converged:
	# 	C.a[j] = C.a1[j] - 0.01*(i+1)*C.a1[j]
	# 	C.get_bio()
	# 	C.build_ca()
	# 	C.check_ca()
	# 	if i == converged[-1]:
	# 		C.plot(scene='show')
	# 	else:
	# 		C.plot()

	# SHAM:       -            +
	# 0: [0,...,98] , [0,...,99] 
	# 1: [0,...,80] , [0,...,99]
	# 2: [0,...,64] , [0,...,99]
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