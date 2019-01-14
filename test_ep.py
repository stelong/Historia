from ep import solver as s
from ep.models import Gattoni_SHAM6Hz as sham
from ep.models import Gattoni_AB6Hz as ab
from utils import ep_out as e
import numpy as np
import matplotlib.pyplot as plt

def main():

	p = ab.initParams()

	S = s.EPSolution(ab)
	S.run2sc(p)
	S.plot_calcium(scene='show')

	C = e.TruncatedCalcium(S.t, S.ca)
	# C.get_cabiomarkers()
	C.visualize_biomarkers(scene='show')

	

	# aa = np.copy(C.a1)
	# aa[0] = 0.5*aa[0]
	# plt.plot(C.t,e.F(C.t, *aa))
	# plt.plot(C.t,e.F(C.t, *C.a1))
	# plt.show()

#-------------------------

if __name__ == "__main__":
    main()