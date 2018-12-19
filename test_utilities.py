
import matplotlib.pyplot as plt
from utilities import scan_logfile as slf

tag = 'data/output_log.txt'

S = slf.extract_info(tag)

plt.plot(S.t, S.lv_v)
plt.show()