#
#
#
import matplotlib.pyplot as plt
import matplotlib.gridspec as grsp
from utilities import scan_logfile as slf
import numpy as np

# nf = 4
# nsf = 200
#
# cv = []
# for i in range(nf):
# 	for j in range(nsf):
# 		tag = 'data/AB_6/options' + str(i+1) + '/' + str(j+1) + '/output_log.txt'
# 		S = slf.extract_info(tag)
# 		cv.append(S.conv)
# 
# print(cv) # convergences vector

tag = 'data/output_log.txt'
S = slf.extract_info(tag)


# gs = grsp.GridSpec(2, 2)

# fig = plt.figure(figsize=(14, 8))
# for i in range(2):
# 	ax = fig.add_subplot(gs[i, 0])
# 	if i == 0:
# 		ax.plot(S.t, S.lv_v)
# 	else:
# 		ax.plot(S.t, S.lv_p)

# ax = fig.add_subplot(gs[:, 1])
# ax.plot(S.lv_v, S.lv_p)

# plt.show()


M1, M3 = slf.display_allout(S.t, S.phase, S.lv_v, S.lv_p)
print(M1)
print(M3)