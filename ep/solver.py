from Historia.ep.model import initialize as init
import numpy as np
from scipy.integrate import solve_ivp

class EPSolution:
	"""This class implements the solution of Gattoni's model* ODEs system.
	*https://physoc.onlinelibrary.wiley.com/doi/10.1113/JP273879
	"""
	def __init__(self, model, rat, hz):
		self.model = model
		self.rat = rat
		self.hz = hz

	def run2sc(self, parameters, nbeats):
		"""Run the ODE solver and store the solution at 1ms spaced time points.
		Args:
			- parameters: (n,)-shaped vector to be initialized through the dedicated module 'initialize.py'
			- nbeats: strictly positive integer, representing the number of beats we want to simulate.
		"""
		self.conv = 1
		self.nbeats = nbeats

		stim_period = parameters[16]
		self.t = np.arange(nbeats*stim_period)
		tspan = [0.0, nbeats*stim_period-1.0]

		Y0 = init.states(self.rat, self.hz)
		c = self.model.initConsts(parameters)
		try:
			Y = solve_ivp(fun=lambda t, y: self.model.computeRates(t, y, c), t_span=tspan, y0=Y0, method='BDF', t_eval=self.t, max_step=1.0)
			self.Y = Y.y
			self.v = Y.y[0, :]
			self.ca = 1e3*Y.y[12, :]
		except:
			print('=== ODE solver failed!')
			self.conv = 0
			self.Y = []
			self.v = []
			self.ca = []