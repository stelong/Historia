import matplotlib.pyplot as plt
import matplotlib.gridspec as grsp
from utilities import scan_logfile as slf
import numpy as np

# n_par = 7
# n_var = 100

# s3 = '/output_log.txt'
# for i in range(n_par):
# 	M = np.zeros(shape=(1, 11), dtype=float)
# 	P = np.zeros(shape=(1, 7), dtype=float)
# 	s1 = 'data/SHAM/j' + str(i+1) + '/'
# 	for j in range(n_var):
# 		s2 = str(j+1)
# 		tag = s1 + s2 + s3

# 		S = slf.extract_info(tag)
# 		conv = slf.check_conv(S.conv, S.phase, 4)
# 		if conv:
# 			P = np.vstack((P, np.asarray(S.p)))
# 			V = slf.display_allout(S.t, S.phase, S.lv_v, S.lv_p)
# 			M = np.vstack((M, np.asarray(V.f)))

# 	name1 = str(i+1) + '_out.txt'
# 	with open(name1, 'w') as f1:
# 		np.savetxt(f1, M[1:, :], fmt='%f')
# 	f1.close()

# 	name2 = str(i+1) + '_in.txt'
# 	with open(name2, 'w') as f2:
# 		np.savetxt(f2, P[1:, :], fmt='%f')
# 	f2.close()

file_in = 'AB_7_in.txt'
with open(file_in, 'w') as f1:
	for k in range(4):
		s = 'data/AB_7/options' + str(k+1) + '/'
		P = np.zeros(shape=(1, 7), dtype=float)
		
		for j in range(100):
			tag = s + str(j+1) + '/output_log.txt'
			S = slf.extract_info(tag)
			conv = slf.check_conv(S.conv, S.phase, 2)
			if conv:
				V = slf.display_allout(S.t, S.phase, S.lv_v, S.lv_p)
				P = np.vstack((P, np.asarray(S.p)))
		np.savetxt(f1, P[1:, :], fmt='%f')
f1.close()							
	
file_out = 'AB_7_out.txt'
with open(file_out, 'w') as f2:
	for k in range(4):
		s = 'data/AB_7/options' + str(k+1) + '/'
		M = np.zeros(shape=(1, 11), dtype=float)

		for j in range(100):
			tag = s + str(j+1) + '/output_log.txt'
			S = slf.extract_info(tag)
			conv = slf.check_conv(S.conv, S.phase, 2)
			if conv:
				V = slf.display_allout(S.t, S.phase, S.lv_v, S.lv_p)
				M = np.vstack((M, np.asarray(V.f)))
		np.savetxt(f2, M[1:, :], fmt='%f')
f2.close()

# gs = grsp.GridSpec(2, 2)
# fig = plt.figure(figsize=(14, 8))
# for i in range(2):
# 	ax = fig.add_subplot(gs[i, 0])
# 	if i == 0:
# 		ax.plot(S.t, S.lv_v, color='b', linewidth=1.0)
# 		ax.plot(V.ts, V.lv_vs, color='b', linewidth=2.5)
# 		plt.xlim(0, V.ts[-1])
# 	else:
# 		ax.plot(S.t, S.lv_p, color='b', linewidth=1.0)
# 		ax.plot(V.ts, V.lv_ps, color='b', linewidth=2.5)
# 		plt.xlim(0, V.ts[-1])

# ax = fig.add_subplot(gs[:, 1])
# ax.plot(V.lv_vs, V.lv_ps, color='b', linewidth=2.5)
# plt.show()