import numpy as np

def initStates():
    sizeStates = 18
    states = np.zeros(shape=(sizeStates,), dtype=float)

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
    sizeParameters = 11
    parameters = np.zeros(shape=(sizeParameters,), dtype=float)

    parameters[0] = 50000 # N
    parameters[1] = 1.45e-6 # g_f
    parameters[2] = 1.4e-5 # g_to
    parameters[3] = 1.5e-5 # g_K1
    parameters[4] = 5e-6 # g_pCa
    parameters[5] = 0.0456 # g_NCX
    parameters[6] = 0.0007 # g_Na
    parameters[7] = 0.00049 # g_SERCA
    parameters[8] = 1e-6 # g_SRL
    parameters[9] = 1e-6 # g_ss
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
    constants[32] = 0.0012
    constants[33] = parameters[0]
    constants[34] = -13
    constants[35] = 7
    constants[36] = 11.5
    constants[37] = 1
    constants[38] = 1550
    constants[39] = 1.17
    constants[40] = 2.4
    constants[41] = 0.05
    constants[42] = 0.012
    constants[43] = 0.065
    constants[44] = 0.00016
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
    constants[55] = 0.00025
    constants[56] = parameters[4]
    constants[57] = 0.00035
    constants[58] = 6e-9
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