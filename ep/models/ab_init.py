import numpy as np

def initStates():
    sizeStates = 18
    states = np.zeros((sizeStates,), dtype=float)

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

    return states

def initParams():
    sizeParams = 16
    parameters = np.zeros((sizeParams,), dtype=float)

    parameters[0] = 0.0007 # <--- g_Na
    parameters[1] = 1.4e-5 # <--- g_to
    parameters[2] = 1e-6 # <--- g_ss
    parameters[3] = 1.5e-5 # <--- g_K1
    parameters[4] = 1.45e-6 # <--- g_f
    parameters[5] = 0.00138 # <--- i_NaK_max
    parameters[6] = 0.0012 # <--- J_L ---> rat
    parameters[7] = 50000 # <--- N
    parameters[8] = -13 # <--- V_L ---> rat
    parameters[9] = 0.00016 # <--- K_L ---> rat
    parameters[10] = 0.0456 # <--- g_NCX
    parameters[11] = 0.00049 # <--- g_SERCA
    parameters[12] = 0.00025 # <--- K_SERCA ---> rat
    parameters[13] = 5e-6 # <--- g_pCa
    parameters[14] = 6e-9 # <--- g_CaB ---> rat
    parameters[15] = 1e-6 # <--- g_SRl

    return parameters