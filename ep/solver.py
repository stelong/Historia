import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

class EPSolution:
	def __init__(self, model):
		self.model = model
		self.conv = 0
		self.t = np.arange(0, 171, 1)
		self.ca = np.zeros((171,), dtype=float)

	def run2sc(self, parameters):
		c = self.model.initConsts(parameters)
		Y0 = self.model.initStates()
		tspan = [0, 170]

		Y = solve_ivp(fun=lambda t, y: self.model.odesys(t, y, c), t_span=tspan, y0=Y0, t_eval=self.t)
		self.ca = 1e3*Y.y[12, :]
		self.conv = 1

	def run2ss(self, parameters):
		c = self.model.initConsts(parameters)
		Y0 = self.model.initStates()
		tspan = [0, 170]
		Y = solve_ivp(fun=lambda t, y: self.model.odesys(t, y, c), t_span=tspan, y0=Y0, t_eval=self.t)

		thre = 10000
		ssnc = b = 1
		while np.linalg.norm(Y.y[:, -1] - Y.y[:, 0], ord=np.inf) > 1e-4:
			Y = solve_ivp(fun=lambda t, y: self.model.odesys(t, y, c), t_span=tspan, y0=Y.y[:, -1], t_eval=self.t)
			ssnc += 1
			if ssnc == thre:
				b = 0
				break
		if b:
			print('\n=== Number of cycles needed to reach the steady-state: {}\n'.format(ssnc))
			self.ca = 1e3*Y.y[12, :]
			self.conv = b
		else:
			print('\n=== Error: unable to reach the steady-state!\n')

	def plot_ca(self, scene='do_not_show'):
		plt.plot(self.t, self.ca)
		if scene == 'show':
			plt.xlim(self.t[0], self.t[-1])
			plt.xlabel('Time [ms]')
			plt.ylabel('Intracellular calcium [$\mu$M]')
			plt.show()