
import time
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def prey_predator(t, y, p):
	dy1 = (p[0] - p[1]*y[1]) * y[0]
	dy2 = (p[2]*y[0] - p[3]) * y[1]

	return [dy1, dy2]

def main():

	p = [1, 1, 1, 1]
	tspan = [0, 170]
	y0 = [0.5, 0.5]
	t = np.linspace(0, 14.9, 150)

	start_time = time.time()

	sol = solve_ivp(fun=lambda t, y: prey_predator(t, y, p), t_span=tspan, y0=y0, t_eval=t)

	elapsed_time = time.time() - start_time
	print(elapsed_time)

	plt.plot(sol.t, sol.y.T)
	plt.show()

#-------------------------

if __name__ == "__main__":
    main()