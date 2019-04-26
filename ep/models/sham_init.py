import numpy as np

def initStates():
	sizeStates = 18
	states = np.zeros((sizeStates,), dtype=float)

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
	sizeParams = 16
	parameters = np.zeros((sizeParams,), dtype=float)

	parameters[0] = 0.0007 # <--- g_Na
	parameters[1] = 2e-5 # <--- g_to
	parameters[2] = 1.3e-5 # <--- g_ss
	parameters[3] = 4e-5 # <--- g_K1
	parameters[4] = 1.45e-6 # <--- g_f
	parameters[5] = 0.00138 # <--- i_NaK_max
	parameters[6] = 0.0008 # <--- J_L ---> rat
	parameters[7] = 50000 # <--- N
	parameters[8] = -9 # <--- V_L ---> rat
	parameters[9] = 0.00038 # <--- K_L ---> rat
	parameters[10] = 0.0234 # <--- g_NCX
	parameters[11] = 0.00051 # <--- g_SERCA
	parameters[12] = 0.00069 # <--- K_SERCA ---> rat
	parameters[13] = 5e-6 # <--- g_pCa
	parameters[14] = 2e-8 # <--- g_CaB ---> rat
	parameters[15] = 1e-6 # <--- g_SRl

	return parameters