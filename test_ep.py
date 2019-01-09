from ep import solver as s
from ep.models import Gattoni_SHAM6Hz as sham
from ep.models import Gattoni_AB6Hz as ab
from utils import ep_out as e
import matplotlib.pyplot as plt

def main():

	sol = s.solve_electr2ss(sham)
	C = e.get_cabiomarkers(sol.t,sol.ca)

	ts = C.ts
	cas = C.cas
	bio = C.a[0]
	a1 = C.a[1]

	plt.plot(C.ts, C.cas, c='k')
	plt.plot(ts, e.F(ts, a1[0], a1[1], a1[2], a1[3]))
	plt.xlim(ts[0], ts[-1])
	plt.scatter(bio[3], bio[1], c='r')
	plt.scatter(bio[2]+bio[3], e.F(bio[2]+bio[3], a1[0], a1[1], a1[2], a1[3]), c='r')
	plt.axvline(x=bio[3], c='r', linestyle='dashed')
	plt.axvline(x=bio[2]+bio[3], c='r', linestyle='dashed')
	plt.axhline(y=bio[0], c='r', linestyle='dashed')
	plt.axhline(y=bio[1], c='r', linestyle='dashed')
	plt.show()

#-------------------------

if __name__ == "__main__":
    main()