import numpy as np

def initStates():
	sizeStates = 4
	state = np.zeros((sizeStates,), dtype=float)

	state[0] = 8e-02
	state[1] = 2e-03
	state[2] = 0.0
	state[3] = 0.0
	
	return state

def initConstants(c_dict):
	sizeConstants = 17
	constant = np.zeros((sizeConstants,), dtype=float)

	for i, key in enumerate(c_dict.keys()):
		constant[i] = c_dict[key]

	return constant

def computeAlgebraics(t, y, Cai, constant):
	sizeAlgebraics = 10
	algebraic = np.zeros((sizeAlgebraics,), dtype=float)

	algebraic[0] = t - len(Cai)*np.floor(t/len(Cai))
	
	if not t // len(Cai):
		algebraic[1] = Cai[int(algebraic[0])]
	else:
		algebraic[1] = Cai[-1]

	algebraic[2] = constant[2]*(1.0 + constant[11]*(np.min([1.2, np.max([0.8, constant[0]])]) - 1.0))
	algebraic[3] = np.sqrt(np.power(y[0]/constant[4], constant[8]))
	algebraic[4] = np.min([100.0, 1.0/algebraic[3]])
	algebraic[5] = y[2] + y[3]

	if algebraic[5] <= 0.0:
		algebraic[6] = (constant[12]*algebraic[5] + 1.0)/(1.0 - algebraic[5])
	else:
		algebraic[6] = (1.0 + (constant[12] + 2.0)*algebraic[5])/(1.0 + algebraic[5])

	algebraic[7] = np.max([0.0, 1.0 + constant[10]*(np.min([constant[0], 1.2]) + np.min([np.min([constant[0], 1.2]), 0.87]) - 1.87)])
	algebraic[8] = algebraic[6] * algebraic[7] * y[1]
	algebraic[9] = constant[3] * algebraic[8]

	return algebraic

def computeRates(t, y, Cai, c_dict):
	constant = initConstants(c_dict)
	algebraic = computeAlgebraics(t, y, Cai, constant)

	sizeRates = 4
	rate = np.zeros((sizeRates,), dtype=float)

	rate[0] = constant[6]*np.power(algebraic[1]/algebraic[2], constant[5])*(1.0 - y[0]) - constant[7]*y[0]
	rate[1] = constant[9]*(algebraic[3]*(1.0 - y[1]) - algebraic[4]*y[1])
	rate[2] = constant[13]*constant[1] - constant[15]*y[2]
	rate[3] = constant[14]*constant[1] - constant[16]*y[3]

	return rate

def computeSteadystate(Cai, c_dict):
	constant = initConstants(c_dict)
	constant[2] = constant[2]*(1.0 + constant[11]*(np.min([1.2, np.max([0.8, constant[0]])]) - 1.0))

	r = np.power(Cai/constant[2], constant[5])
	TRPN_ss = r/(constant[7]/constant[6] + r)
	trpn_EC50 = np.power(constant[7]/constant[6], 1.0/constant[5])*constant[2]

	s = np.power(TRPN_ss/constant[4], constant[8])
	XB_ss = s/(1.0 + s)
	xb_EC50 = constant[2]*np.power((constant[7]*constant[4])/(constant[6]*(1.0 - constant[4])), 1.0/constant[5])

	F_ss = constant[3]*XB_ss
	pCa50 = -np.log10(1e-6*constant[2]*np.power((constant[7]*constant[4])/(constant[6]*(1.0 - constant[4])), 1.0/constant[5]))

	return TRPN_ss, trpn_EC50, XB_ss, xb_EC50, F_ss, pCa50