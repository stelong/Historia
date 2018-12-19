#
#
#
import matplotlib.pyplot as plt
from utilities import scan_logfile as slf

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

fig, axes = plt.subplots(1, 3, figsize=(12,4))
for i, a in enumerate(axes.flatten()):
	if i == 0:
		a.plot(S.t, S.lv_v)
	elif i == 1:
		a.plot(S.t, S.lv_p)
	else:
		a.plot(S.lv_v, S.lv_p)

plt.show()