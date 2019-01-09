from ep import solver as s
from ep.models import Gattoni_SHAM6Hz as sham
from ep.models import Gattoni_AB6Hz as ab

def main():

	sol = s.solve_electr2ss(ab)
	s.plot_calcium(sol)

#-------------------------

if __name__ == "__main__":
    main()