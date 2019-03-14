import numpy as np
from scipy.integrate import solve_ivp

class EPSolution:
	def __init__(self, odesys, model):
		self.odesys = odesys
		self.model = model
		self.t = np.arange(0, 170+1, 1)
		self.Y = np.zeros((170+1, 18), dtype=float)
		self.ca = np.zeros((170+1,), dtype=float)

	def run2sc(self, parameters):
		self.conv = 1
		tspan = [0, 170]
		Y0 = self.model.initStates()
		c = self.model.initConsts(parameters)
		Y = solve_ivp(fun=lambda t, y: self.odesys.f(t, y, c), t_span=tspan, y0=Y0, method='BDF', t_eval=self.t, max_step=1.0)
		self.Y = Y.y
		self.ca = 1e3*Y.y[12, :]

	def run2ss(self, parameters):
		self.conv = 1
		tspan = [0, 170]
		Y0 = self.model.initStates()
		c = self.model.initConsts(parameters)
		Y = solve_ivp(fun=lambda t, y: self.odesys.f(t, y, c), t_span=tspan, y0=Y0, method='BDF', t_eval=self.t)

		thre = 100
		ssnc = 1
		while np.linalg.norm((Y.y[:, -1] - Y.y[:, 0])/Y.y[:, 0], ord=np.inf) > 1e-3:
			Y = solve_ivp(fun=lambda t, y: self.odesys.f(t, y, c), t_span=tspan, y0=Y.y[:, -1], method='BDF', t_eval=self.t)
			ssnc += 1
			if ssnc == thre:
				self.conv = 0
				break
		if self.conv:
			print('\n=== Number of cycles needed to reach the steady-state: {}\n'.format(ssnc))
			self.Y = Y.y
			self.ca = 1e3*Y.y[12, :]			
		else:
			print('\n=== Error: unable to reach the steady-state!\n')