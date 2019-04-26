import numpy as np
from scipy.integrate import solve_ivp

class EPSolution:
	def __init__(self, model, rat):
		self.rat = rat
		self.model = model

	def run2sc(self, parameters, nbeats):
		self.conv = 1
		self.nbeats = nbeats
		self.t = np.arange(nbeats*170)
		tspan = [0, nbeats*170-1]
		Y0 = self.rat.initStates()
		c = self.model.initConsts(parameters)
		try:
			Y = solve_ivp(fun=lambda t, y: self.model.computeRates(t, y, c), t_span=tspan, y0=Y0, method='BDF', t_eval=self.t, max_step=1.0)
			self.Y = Y.y
			self.ca = 1e3*Y.y[12, :]
		except:
			print('=== ODE solver failed!')
			self.conv = 0
			self.Y = []
			self.ca = []
			
	def run2ss(self, parameters):
		self.conv = 1
		self.t = np.arange(170)
		tspan = [0, 170-1]
		Y0 = self.rat.initStates()
		c = self.model.initConsts(parameters)
		try:
			Y = solve_ivp(fun=lambda t, y: self.model.computeRates(t, y, c), t_span=tspan, y0=Y0, method='BDF', t_eval=self.t)

			thre = 100
			ssnc = 1
			while np.linalg.norm((Y.y[:, -1] - Y.y[:, 0])/Y.y[:, 0], ord=np.inf) > 1e-3:
				Y = solve_ivp(fun=lambda t, y: self.model.computeRates(t, y, c), t_span=tspan, y0=Y.y[:, -1], method='BDF', t_eval=self.t)
				ssnc += 1
				if ssnc == thre:
					print('\n=== Error: unable to reach the steady-state!\n')
					self.conv = 0
					self.Y = []
					self.ca = []
					break

			if self.conv:
				print('\n=== Number of cycles needed to reach the steady-state: {}\n'.format(ssnc))
				self.Y = Y.y
				self.ca = 1e3*Y.y[12, :]
		except:
			print('=== ODE solver failed!')
			self.conv = 0