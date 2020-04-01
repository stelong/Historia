from Historia.cellcontr.model import Land2012
import numpy as np
from scipy.integrate import solve_ivp

class CONTRSolution:
	"""This class implements the solution of Land et al. (2012) cellular contraction model*.
	*https://physoc.onlinelibrary.wiley.com/doi/full/10.1113/jphysiol.2012.231928
	"""
	def __init__(self, Cai, c_dict):
		self.Cai = Cai
		self.constant = c_dict

	def solver_sol(self, p_dict=None):
		if p_dict is not None:
			for key in p_dict.keys():
				self.constant[key] = p_dict[key]
		
		self.t = np.arange(2*len(self.Cai))
		tspan = [0, self.t[-1]]
		Y0 = Land2012.initStates()
		Y = solve_ivp(fun=lambda t, y: Land2012.computeRates(t, y, self.Cai, self.constant), t_span=tspan, y0=Y0, method='BDF', t_eval=self.t, max_step=1.0)
		self.Y = Y.y

		constant = Land2012.initConstants(self.constant)
		self.T = np.array([Land2012.computeAlgebraics(ti, Yi, self.Cai, constant)[-1] for ti, Yi in zip(self.t, self.Y.T)])

	def steadystate_sol(self, p_dict=None):
		if p_dict is not None:
			for key in p_dict.keys():
				self.constant[key] = p_dict[key]

		TRPN_ss, trpn_EC50, XB_ss, xb_EC50, F_ss, pCa50 = Land2012.computeSteadystate(self.Cai, self.constant)
		self.TRPN = {'TRPN': TRPN_ss, 'EC50': trpn_EC50}
		self.XB = {'XB': XB_ss, 'EC50': xb_EC50}
		self.F = {'F': F_ss, 'pCa50': pCa50}