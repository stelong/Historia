from ep import solver as s
from ep.models import Gattoni_SHAM6Hz as sham
from ep.models import Gattoni_AB6Hz as ab
from utils import ep_out as e

def main():

	sol = s.solve_electr2ss(sham)
	C = e.get_cabiomarkers(sol.t,sol.ca)

	ts = C.ts
	cas = C.cas
	a = C.a
	bio = C.bio

	s.plot_calcium(sol, scene='show')
	e.visualize_biomarkers(ts, a, bio, scene='show')

#-------------------------

if __name__ == "__main__":
    main()