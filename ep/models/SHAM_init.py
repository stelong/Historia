import numpy as np

def initStates():
	sizeStates = 18
	states = np.zeros(shape=(sizeStates,), dtype=float)

	states[0] = -85.5696059816548
	states[1] = 12.960204722764
	states[2] = 0.00194320513549252
	states[3] = 0.824266840236936
	states[4] = 0.722995966352439
	states[5] = 139.994789812443
	states[6] = 0.00140717973440432
	states[7] = 0.954569649396715
	states[8] = 0.300568843000913
	states[9] = 0.00189620611861644
	states[10] = 0.310616913661423
	states[11] = 0.00395453931879583
	states[12] = 0.000463188750506148
	states[13] = 1.0663135643117
	states[14] = 0.915759670501361
	states[15] = 0.0116308821918943
	states[16] = 0.0716996181298885
	states[17] = 0.0462772817360074

	return states

def initParams():
	sizeParameters = 11
	parameters = np.zeros(shape=(sizeParameters,), dtype=float)

	parameters[0] = 50000 # N
	parameters[1] = 1.45e-6 # g_f
	parameters[2] = 2e-5 # g_to
	parameters[3] = 4e-5 # g_K1
	parameters[4] = 5e-6 # g_pCa
	parameters[5] = 0.0234 # g_NCX
	parameters[6] = 0.0007 # g_Na
	parameters[7] = 0.00051 # g_SERCA
	parameters[8] = 1e-6 # g_SRL
	parameters[9] = 1.3e-5 # g_ss
	parameters[10] = 0.00138 # i_NaK_max

	return parameters

def initConsts(parameters):
    sizeConstants = 71
    constants = np.zeros(shape=(sizeConstants,), dtype=float)

    constants[0] = 8314
    constants[1] = 310
    constants[2] = 96487
    constants[3] = 0.0001
    constants[4] = 170
    constants[5] = 3
    constants[6] = -0.0012
    constants[7] = 25850
    constants[8] = 2.585e-5
    constants[9] = 2.098e-6
    constants[10] = parameters[6]
    constants[11] = 140
    constants[12] = parameters[2]
    constants[13] = 0.883
    constants[14] = 0.117
    constants[15] = 5.4
    constants[16] = parameters[9]
    constants[17] = parameters[3]
    constants[18] = parameters[1]
    constants[19] = 0.2
    constants[20] = 8.015e-8
    constants[21] = 1.38e-7
    constants[22] = parameters[10]
    constants[23] = 3.6
    constants[24] = 19
    constants[25] = 22
    constants[26] = 880
    constants[27] = 0.3
    constants[28] = 1.8
    constants[29] = 1.8
    constants[30] = 0.1
    constants[31] = 0.02
    constants[32] = 0.0008
    constants[33] = parameters[0]
    constants[34] = -9
    constants[35] = 7
    constants[36] = 11.5
    constants[37] = 1
    constants[38] = 1550
    constants[39] = 1.17
    constants[40] = 2.4
    constants[41] = 0.05
    constants[42] = 0.012
    constants[43] = 0.065
    constants[44] = 0.00038
    constants[45] = 0.0625
    constants[46] = 14
    constants[47] = 0.01
    constants[48] = 100
    constants[49] = 87.5
    constants[50] = 1.38
    constants[51] = 0.35
    constants[52] = 0.1
    constants[53] = parameters[5]
    constants[54] = parameters[7]
    constants[55] = 0.00069
    constants[56] = parameters[4]
    constants[57] = 0.00035
    constants[58] = 2e-8
    constants[59] = parameters[8]
    constants[60] = 0.04
    constants[61] = 40
    constants[62] = 0.07
    constants[63] = 0.002382
    constants[64] = 0.05
    constants[65] = 0
    constants[66] = 0.00015
    constants[67] = 2100.00
    constants[68] = 1.00000-constants[19]
    constants[69] = constants[36]/constants[37]
    constants[70] = constants[41]/constants[39]

    return constants