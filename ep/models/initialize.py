import numpy as np

def states(rat, hz):
	sizeStates = 18
	states = np.zeros((sizeStates,), dtype=float)

	if rat == 'sham':
		if hz == 1:
			states[0] = -85.1221681609219
			states[1] = 8.46899983583716
			states[2] = 0.00208137744708665
			states[3] = 0.815520320018128
			states[4] = 0.815471795073686
			states[5] = 142.919492013701
			states[6] = 0.00146331830465093
			states[7] = 0.996934138278418
			states[8] = 0.78841193673441
			states[9] = 0.0019683140031203
			states[10] = 0.416987850222633
			states[11] = 0.00566123148325894
			states[12] = 0.000103020385969363
			states[13] = 0.96268028201207
			states[14] = 0.990016532916529
			states[15] = 0.00845823628523856
			states[16] = 0.00151233172289407
			states[17] = 0.0633670056927004
		elif hz == 6:
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
		else:
			print('\n=== Invalid frequency! Try with either \'1\' Hz or \'6\' Hz.')

	elif rat == 'ab':
		if hz == 1:
			states[0] = -84.8179335690707
			states[1] = 7.86198765026574
			states[2] = 0.00218089269066671
			states[3] = 0.807799685437556
			states[4] = 0.807494936350446
			states[5] = 139.964416466575
			states[6] = 0.00150276538930736
			states[7] = 0.996791125715503
			states[8] = 0.625815901322428
			states[9] = 0.0020195578095622
			states[10] = 0.368560872566697
			states[11] = 0.00484325258805042
			states[12] = 4.53106928940201e-5
			states[13] = 1.22566116672694
			states[14] = 0.989618266628688
			states[15] = 0.00828851292530188
			states[16] = 0.00207588449544264
			states[17] = 0.0669207270187171
		elif hz == 6:
			states[0] = -86.2742001770196
			states[1] = 10.83563050735
			states[2] = 0.00174392847776228
			states[3] = 0.831855538514125
			states[4] = 0.585528587217805
			states[5] = 140.1638657529
			states[6] = 0.00132309740749641
			states[7] = 0.913059928712858
			states[8] = 0.215036691884835
			states[9] = 0.00178817500396492
			states[10] = 0.209855226174927
			states[11] = 0.0025539544569868
			states[12] = 0.000145080891755077
			states[13] = 1.78499523287115
			states[14] = 0.650490049579104
			states[15] = 0.00570189155987283
			states[16] = 0.340821224038804
			states[17] = 0.0590608556435992
		else:
			print('\n=== Invalid frequency! Try with either \'1\' Hz or \'6\' Hz.')

	else:
		print('\n=== Invalid rat phenotype! Try with either \'sham\' rat or \'ab\' rat.')

	return states

def params(rat, hz):
	sizeParams = 17
	parameters = np.zeros((sizeParams,), dtype=float)

	if rat == 'sham':
		parameters[0] = 0.0007 # <--- g_Na ---> rat
		parameters[1] = 2.0e-5 # <--- g_to ---> rat
		parameters[2] = 1.3e-5 # <--- g_ss ---> rat
		parameters[3] = 4.0e-5 # <--- g_K1 ---> rat
		parameters[4] = 1.45e-6 # <--- g_f
		parameters[5] = 0.00138 # <--- i_NaK_max
		parameters[6] = 0.0008 # <--- J_L ---> rat
		parameters[7] = 50000.0 # <--- N
		parameters[8] = -9.0 # <--- V_L ---> rat
		parameters[9] = 0.00038 # <--- K_L ---> rat
		parameters[10] = 0.0234 # <--- g_NCX ---> rat
		parameters[13] = 5.0e-6 # <--- g_pCa
		parameters[14] = 2.0e-8 # <--- g_CaB ---> rat
		parameters[15] = 1.0e-6 # <--- g_SRl

		if hz == 1:
			parameters[11] = 0.000235 # <--- g_SERCA ---> rat & freq
			parameters[12] = 0.0004968 # <--- K_SERCA ---> rat & freq
			parameters[16] = 1000.0 # <--- stim_period ---> freq
		elif hz == 6:
			parameters[11] = 0.00051 # <--- g_SERCA ---> rat & freq
			parameters[12] = 0.00069 # <--- K_SERCA ---> rat & freq
			parameters[16] = 170.0 # <--- stim_period ---> freq
		else:
			print('\n=== Invalid frequency! Try with either \'1\' Hz or \'6\' Hz.')

	elif rat == 'ab':
		parameters[0] = 0.0002 # <--- g_Na ---> rat
		parameters[1] = 1.4e-5 # <--- g_to ---> rat
		parameters[2] = 1.0e-6 # <--- g_ss ---> rat
		parameters[3] = 1.5e-5 # <--- g_K1 ---> rat
		parameters[4] = 1.45e-6 # <--- g_f
		parameters[5] = 0.00138 # <--- i_NaK_max
		parameters[6] = 0.0012 # <--- J_L ---> rat
		parameters[7] = 50000.0 # <--- N
		parameters[8] = -13.0 # <--- V_L ---> rat
		parameters[9] = 0.00016 # <--- K_L ---> rat
		parameters[10] = 0.0456 # <--- g_NCX ---> rat
		parameters[11] = 0.00049 # <--- g_SERCA ---> rat
		parameters[13] = 5.0e-6 # <--- g_pCa
		parameters[14] = 6.0e-9 # <--- g_CaB ---> rat
		parameters[15] = 1.0e-6 # <--- g_SRl

		if hz == 1:
			parameters[12] = 0.00044 # <--- K_SERCA ---> rat & freq
			parameters[16] = 1000.0 # <--- stim_period ---> freq
		elif hz == 6:
			parameters[12] = 0.00025 # <--- K_SERCA ---> rat & freq
			parameters[16] = 170.0 # <--- stim_period ---> freq
		else:
			print('\n=== Invalid frequency! Try with either \'1\' Hz or \'6\' Hz.')

	else:
		print('\n=== Invalid rat phenotype! Try with either \'sham\' rat or \'ab\' rat.')

	return parameters