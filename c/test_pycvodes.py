
import time
import numpy as np
from pycvodes import integrate_predefined
import matplotlib.pyplot as plt

def f(t, y, dydt):
	dydt[0] = (1 - y[1]) * y[0]
	dydt[1] = (y[0] - 1) * y[1]

def j(t, y, J, dfdt=None, fy=None):
	J[0, 0] = 1 - y[1]
	J[0, 1] = -y[0]
	J[1, 0] = y[1]
	J[1, 1] = y[0] - 1

def main():
	y0 = [0.5, 0.5]
	dt0 = 1e-8
	t0 = 0.0
	atol = 1e-8
	rtol = 1e-8
	tout = np.linspace(0, 14.9, 150)

	start_time = time.time()

	yout, info = integrate_predefined(f, j, y0, tout, atol, rtol, dt0, method='bdf')

	elapsed_time = time.time() - start_time
	print(elapsed_time)

	series = plt.plot(tout, yout)
	plt.show()

#-------------------------

if __name__ == "__main__":
    main()