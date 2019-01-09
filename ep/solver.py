import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

class Solution():
	def __init__(self, conv, t, ca):
		self.conv = conv
		self.t = t
		self.ca = ca

def solve_electr2ss(model):
	p = model.initParams()
	c = model.initConsts(p)
	Y0 = model.initStates()
	tspan = [0, 170]
	t = range(0, 171)

	Y = solve_ivp(fun=lambda t, y: model.odesys(t, y, c, p), t_span=tspan, y0=Y0, t_eval=t)

	tre = 10000
	ssnc = b = 1

	while np.linalg.norm(Y.y[:, -1] - Y.y[:, 0], ord=np.inf) > 1e-1:
		Y = solve_ivp(fun=lambda t, y: model.odesys(t, y, c, p), t_span=tspan, y0=Y.y[:, -1], t_eval=t)
		ssnc += 1

		if ssnc == tre:
			b = 0
			break

	if b:
		print('\n=== Number of cycles needed to reach the steady-state: {}\n'.format(ssnc))
	else:
		print('\n=== Error: unable to reach the steady-state!\n')

	return Solution(b, Y.t, 1e3*Y.y[12, :])

def plot_calcium(sol):
	plt.plot(sol.t, sol.ca)
	plt.xlim(sol.t[0], sol.t[-1])
	plt.xlabel('Time [ms]')
	plt.ylabel('Intracellular calcium [$\mu$M]')
	plt.show()

	return