
import time
import numpy as np
import sympy as sp
from pycvodes import integrate_predefined
import matplotlib.pyplot as plt

def f_app(t, y, p):
	return np.array([
		(p[0] - p[1]*y[1]) * y[0],
		(p[2]*y[0] - p[3]) * y[1]
		])

t = sp.symbols('t')
y = y0, y1 = sp.symbols('y0 y1')
p = p1, p2, p3, p4 = sp.symbols('p1 p2 p3 p4')

Jf = sp.Matrix(f_app(t, y, p)).jacobian(y)
J_func = sp.lambdify((t, y, p), Jf)

def f(t, y, dydt):
	dydt[0] = (1 - y[1]) * y[0]
	dydt[1] = (y[0] - 1) * y[1]

def j(t, y, J, dfdt=None, fy=None):
	J = J_func(t, y, [1, 1, 1, 1])

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