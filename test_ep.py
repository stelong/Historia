from ep import Gattoni_SHAM6Hz as sham
from ep import Gattoni_AB6Hz as ab
from scipy.integrate import solve_ivp
import time

def main():
	p = sham.initParams()
	c = sham.initConsts(p)
	y0 = sham.initStates()
	tspan = [0, 170]
	t = range(0, 171, 1)

	start_time = time.time()

	sol = solve_ivp(fun=lambda t, y: sham.model(t, y, c, p), t_span=tspan, y0=y0, t_eval=t)

	elapsed_time = time.time() - start_time
	print(elapsed_time)

	sham.plot_calcium(sol)

#-------------------------

if __name__ == "__main__":
    main()