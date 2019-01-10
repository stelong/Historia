#
# 6Hz paced SHAM rat heart electrophysiological model (Gattoni 2016).
#
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

def computeAlgebraic(constants, states, voi):
    sizeAlgebraic = 85
    algebraic = np.zeros(shape=(sizeAlgebraic,), dtype=float)

    algebraic[8] = 1.00000/(1.00000+np.exp((states[0]+87.5000)/10.3000))
    algebraic[1] = 1.00000/(1.00000+np.exp((states[0]+45.0000)/-6.50000))
    algebraic[11] = 1.36000/((0.320000*(states[0]+47.1300))/(1.00000-np.exp(-0.100000*(states[0]+47.1300)))+0.0800000*np.exp(-states[0]/11.0000))
    algebraic[2] = 1.00000/(1.00000+np.exp((states[0]+76.1000)/6.07000))
    
    if states[0] >= -40.0000:
    	algebraic[12] = 0.453700*(1.00000+np.exp(-(states[0]+10.6600)/11.1000))
    else:
    	algebraic[12] = 3.49000/(0.135000*np.exp(-(states[0]+80.0000)/6.80000)+3.56000*np.exp(0.0790000*states[0])+310000.*np.exp(0.350000*states[0]))
    
    algebraic[3] = 1.00000/(1.00000+np.exp((states[0]+76.1000)/6.07000))
    
    if states[0] >= -40.0000:
    	algebraic[13] = (11.6300*(1.00000+np.exp(-0.100000*(states[0]+32.0000))))/np.exp(-2.53500e-07*states[0])
    else:
    	algebraic[13] = 3.49000/(((states[0]+37.7800)/(1.00000+np.exp(0.311000*(states[0]+79.2300))))*(-127140.*np.exp(0.244400*states[0])-3.47400e-05*np.exp(-0.0439100*states[0]))+(0.121200*np.exp(-0.0105200*states[0]))/(1.00000+np.exp(-0.137800*(states[0]+40.1400))))

    algebraic[14] = 100.000/(45.1600*np.exp(0.0357700*(states[0]+50.0000))+98.9000*np.exp(-0.100000*(states[0]+38.0000)))
    algebraic[4] = 1.00000/(1.00000+np.exp((states[0]+10.6000)/-11.4200))
    algebraic[15] = 20.0000*np.exp(-(np.power((states[0]+70.0000)/25.0000, 2.00000)))+35.0000
    algebraic[5] = 1.00000/(1.00000+np.exp((states[0]+45.3000)/6.88410))
    algebraic[16] = 1300.00*np.exp(-(np.power((states[0]+70.0000)/30.0000, 2.00000)))+35.0000
    algebraic[6] = 1.00000/(1.00000+np.exp((states[0]+45.3000)/6.88410))
    algebraic[17] = 10000.0/(45.1600*np.exp(0.0357700*(states[0]+50.0000))+98.9000*np.exp(-0.100000*(states[0]+38.0000)))
    algebraic[7] = 1.00000/(1.00000+np.exp((states[0]+11.5000)/-11.8200))
    algebraic[18] = 1000.00/(0.118850*np.exp((states[0]+80.0000)/28.3700)+0.562300*np.exp((states[0]+80.0000)/-14.1900))
    algebraic[9] = 1.00000/(1.00000+np.exp((states[0]+138.600)/10.4800))
    algebraic[23] = ((constants[0]*constants[1])/constants[2])*np.log(constants[15]/states[5])
    algebraic[24] = constants[12]*states[6]*(constants[13]*states[7]+constants[14]*states[8])*(states[0]-algebraic[23])
    algebraic[25] = constants[16]*states[9]*states[10]*(states[0]-algebraic[23])
    algebraic[26] = ((0.0480000/(np.exp((states[0]+37.0000)/25.0000)+np.exp((states[0]+37.0000)/-25.0000))+0.0100000)*0.00100000)/(1.00000+np.exp((states[0]-(algebraic[23]+76.7700))/-17.0000))+(constants[17]*(states[0]-(algebraic[23]+1.73000)))/((1.00000+np.exp((1.61300*constants[2]*(states[0]-(algebraic[23]+1.73000)))/(constants[0]*constants[1])))*(1.00000+np.exp((constants[15]-0.998800)/-0.124000)))
    algebraic[31] = constants[21]*(states[0]-algebraic[23])
    algebraic[33] = constants[23]*(np.power(1.00000+constants[24]/states[1], 2.00000))*(1.00000+(constants[25]/states[1])*np.exp(((-constants[27]*constants[2]*states[0])/constants[0])/constants[1]))+(np.power(1.00000+constants[28]/constants[15], 2.00000))*(1.00000+(constants[11]/constants[26])*np.exp(((-(1.00000-constants[27])*constants[2]*states[0])/constants[0])/constants[1]))
    algebraic[34] = (constants[22]*(constants[23]+1.00000))/algebraic[33]
    
    if voi-np.floor(voi/constants[4])*constants[4] >= 0.00000 and voi-np.floor(voi/constants[4])*constants[4] <= constants[5]:
    	algebraic[20] = constants[6]
    else:
    	algebraic[20] = 0.00000
    
    algebraic[28] = constants[18]*states[11]*constants[68]*(states[0]-algebraic[23])
    algebraic[0] = (constants[2]*states[0])/(constants[0]*constants[1])
    algebraic[19] = 2.00000*algebraic[0]
    
    if np.fabs(algebraic[19]) > 1.00000e-09:
    	algebraic[44] = (states[12]+((constants[32]/constants[30])*constants[29]*algebraic[19]*np.exp(-algebraic[19]))/(1.00000-np.exp(-algebraic[19])))/(1.00000+((constants[32]/constants[30])*algebraic[19])/(1.00000-np.exp(-algebraic[19])))
    else:
    	algebraic[44] = (states[12]+(constants[32]/constants[30])*constants[29])/(1.00000+constants[32]/constants[30])
    
    algebraic[46] = (np.power(algebraic[44], 2.00000)+constants[47]*(np.power(constants[43], 2.00000)))/(constants[40]*(np.power(algebraic[44], 2.00000)+np.power(constants[43], 2.00000)))
    algebraic[40] = (np.power(states[12], 2.00000)+constants[47]*(np.power(constants[43], 2.00000)))/(constants[40]*(np.power(states[12], 2.00000)+np.power(constants[43], 2.00000)))
    algebraic[35] = np.exp((states[0]-constants[34])/constants[35])
    algebraic[36] = algebraic[35]/(constants[37]*(algebraic[35]+1.00000))
    algebraic[37] = (np.power(states[12], 2.00000))/(constants[39]*(np.power(states[12], 2.00000)+np.power(constants[43], 2.00000)))
    algebraic[45] = (np.power(algebraic[44], 2.00000))/(constants[39]*(np.power(algebraic[44], 2.00000)+np.power(constants[43], 2.00000)))
    algebraic[52] = (algebraic[36]+constants[69])*((constants[69]+constants[70]+algebraic[45])*(constants[70]+algebraic[37])+algebraic[36]*(constants[70]+algebraic[45]))
    algebraic[53] = (algebraic[36]*constants[70]*(algebraic[36]+constants[69]+constants[70]+algebraic[37]))/algebraic[52]
    algebraic[56] = (constants[69]*constants[70]*(constants[69]+algebraic[36]+constants[70]+algebraic[45]))/algebraic[52]
    algebraic[57] = algebraic[53]*algebraic[46]+algebraic[56]*algebraic[40]
    algebraic[47] = (constants[42]*constants[48]*(np.power(algebraic[44], 2.00000)+constants[47]*(np.power(constants[43], 2.00000))))/(constants[40]*(constants[48]*(np.power(algebraic[44], 2.00000))+constants[47]*(np.power(constants[43], 2.00000))))
    algebraic[41] = (constants[42]*constants[48]*(np.power(states[12], 2.00000)+constants[47]*(np.power(constants[43], 2.00000))))/(constants[40]*(constants[48]*(np.power(states[12], 2.00000))+constants[47]*(np.power(constants[43], 2.00000))))
    algebraic[59] = (algebraic[36]*algebraic[47]+constants[69]*algebraic[41])/(algebraic[36]+constants[69])
    algebraic[42] = (states[12]+(constants[31]/constants[30])*states[13])/(1.00000+constants[31]/constants[30])
    algebraic[43] = (algebraic[42]*(algebraic[35]+constants[45]))/(constants[38]*constants[44]*(algebraic[35]+1.00000))
    algebraic[38] = (states[12]*(algebraic[35]+constants[45]))/(constants[38]*constants[44]*(algebraic[35]+1.00000))
    algebraic[54] = (constants[69]*(algebraic[37]*(constants[69]+constants[70]+algebraic[45])+algebraic[45]*algebraic[36]))/algebraic[52]
    algebraic[65] = algebraic[54]*algebraic[43]+algebraic[56]*algebraic[38]
    algebraic[39] = (constants[46]*(algebraic[35]+constants[45]))/(constants[38]*(constants[46]*algebraic[35]+constants[45]))
    algebraic[67] = algebraic[39]
    algebraic[69] = (constants[69]*algebraic[38])/(algebraic[36]+constants[69])
    algebraic[71] = algebraic[39]
    algebraic[73] = ((1.00000-states[14])-states[15])-states[16]
    algebraic[61] = (constants[70]*algebraic[40])/(constants[70]+algebraic[37])
    algebraic[63] = algebraic[41]
    algebraic[21] = ((constants[0]*constants[1])/constants[2])*np.log(constants[11]/states[1])
    algebraic[22] = constants[10]*(np.power(states[2], 3.00000))*states[3]*states[4]*(states[0]-algebraic[21])
    algebraic[30] = constants[20]*(states[0]-algebraic[21])
    algebraic[74] = (constants[53]*(np.exp(constants[51]*algebraic[0])*(np.power(states[1], 3.00000))*constants[29]-np.exp((constants[51]-1.00000)*algebraic[0])*(np.power(constants[11], 3.00000))*states[12]))/((np.power(constants[11], 3.00000)+np.power(constants[49], 3.00000))*(constants[29]+constants[50])*(1.00000+constants[52]*np.exp((constants[51]-1.00000)*algebraic[0])))
    algebraic[75] = algebraic[74]*constants[2]*constants[8]
    algebraic[27] = constants[18]*states[11]*constants[19]*(states[0]-algebraic[21])
    algebraic[29] = algebraic[27]+algebraic[28]
    algebraic[77] = (constants[56]*states[12])/(constants[57]+states[12])
    algebraic[78] = algebraic[77]*2.00000*constants[2]*constants[8]
    
    if np.fabs(algebraic[19]) > 1.00000e-05:
    	algebraic[51] = (((constants[32]*algebraic[19])/(1.00000-np.exp(-algebraic[19])))*((constants[29]*np.exp(-algebraic[19])-states[12])+(constants[31]/constants[30])*(constants[29]*np.exp(-algebraic[19])-states[13])))/(1.00000+constants[31]/constants[30]+((constants[32]/constants[30])*algebraic[19])/(1.00000-np.exp(algebraic[19])))
    else:
    	algebraic[51] = (((constants[32]*1.00000e-05)/(1.00000-np.exp(-1.00000e-05)))*((constants[29]*np.exp(-1.00000e-05)-states[12])+(constants[31]/constants[30])*(constants[29]*np.exp(-1.00000e-05)-states[13])))/(1.00000+constants[31]/constants[30]+((constants[32]/constants[30])*1.00000e-05)/(1.00000-np.exp(-1.00000e-05)))
    
    if np.fabs(algebraic[19]) > 1.00000e-05:
    	algebraic[50] = (((constants[32]*algebraic[19])/(1.00000-np.exp(-algebraic[19])))*(constants[29]*np.exp(-algebraic[19])-states[12]))/(1.00000+((constants[32]/constants[30])*algebraic[19])/(1.00000-np.exp(-algebraic[19])))
    else:
    	algebraic[50] = (((constants[32]*1.00000e-05)/(1.00000-np.exp(-1.00000e-05)))*(constants[29]*np.exp(-1.00000e-05)-states[12]))/(1.00000+((constants[32]/constants[30])*1.00000e-05)/(1.00000-np.exp(-1.00000e-05)))
    
    algebraic[55] = (algebraic[36]*(algebraic[45]*(algebraic[36]+constants[70]+algebraic[37])+algebraic[37]*constants[69]))/algebraic[52]
    algebraic[66] = algebraic[51]*algebraic[55]+algebraic[50]*algebraic[53]
    algebraic[68] = (algebraic[50]*algebraic[36])/(algebraic[36]+constants[69])
    algebraic[70] = ((states[14]*algebraic[66]+states[15]*algebraic[68])*constants[33])/constants[7]
    algebraic[72] = -algebraic[70]*2.00000*constants[2]*constants[8]
    algebraic[79] = ((constants[0]*constants[1])/(2.00000*constants[2]))*np.log(constants[29]/states[12])
    algebraic[80] = constants[58]*(algebraic[79]-states[0])
    algebraic[81] = -algebraic[80]*2.00000*constants[2]*constants[8]
    algebraic[48] = (constants[31]*(states[13]-states[12]))/(1.00000+constants[31]/constants[30])
    
    if np.fabs(algebraic[19]) > 1.00000e-05:
    	algebraic[49] = (constants[31]*((states[13]-states[12])+(((constants[32]/constants[30])*algebraic[19])/(1.00000-np.exp(-algebraic[19])))*(states[13]-constants[29]*np.exp(-algebraic[19]))))/(1.00000+constants[31]/constants[30]+((constants[32]/constants[30])*algebraic[19])/(1.00000-np.exp(-algebraic[19])))
    else:
    	algebraic[49] = (constants[31]*((states[13]-states[12])+(((constants[32]/constants[30])*1.00000e-05)/(1.00000-np.exp(-1.00000e-05)))*(states[13]-constants[29]*np.exp(-1.00000e-05))))/(1.00000+constants[31]/constants[30]+((constants[32]/constants[30])*1.00000e-05)/(1.00000-np.exp(-1.00000e-05)))
    
    algebraic[58] = algebraic[55]*algebraic[49]+algebraic[48]*algebraic[54]
    algebraic[60] = (algebraic[48]*algebraic[37])/(constants[70]+algebraic[37])
    algebraic[62] = ((states[14]*algebraic[58]+states[16]*algebraic[60])*constants[33])/constants[7]
    algebraic[64] = algebraic[62]
    algebraic[76] = (constants[54]*(np.power(states[12], 2.00000)))/(np.power(constants[55], 2.00000)+np.power(states[12], 2.00000))
    algebraic[82] = constants[59]*(states[13]-states[12])
    algebraic[83] = constants[60]*(constants[62]-states[17])-constants[61]*states[17]*states[12]
    algebraic[84] = np.power(1.00000+(constants[63]*constants[64])/(np.power(constants[63]+states[12], 2.00000))+(constants[65]*constants[66])/(np.power(constants[66]+states[12], 2.00000)), -1.00000)
    algebraic[10] = states[12]
    algebraic[32] = algebraic[30]+algebraic[31]

    return algebraic

def computeRates(constants, algebraic, states, voi):
    sizeRates = 18
    rates = np.zeros(shape=(sizeRates,), dtype=float)
    
    rates[0] = -(algebraic[22]+algebraic[24]+algebraic[25]+algebraic[29]+algebraic[26]+algebraic[30]+algebraic[31]+algebraic[34]+algebraic[81]+algebraic[75]+algebraic[78]+algebraic[72]+algebraic[20])/constants[3]
    rates[1] = (-(algebraic[22]+algebraic[30]+algebraic[75]*3.00000+algebraic[34]*3.00000+algebraic[27])*1.00000)/(constants[8]*constants[2])
    rates[2] = (algebraic[1]-states[2])/algebraic[11]
    rates[3] = (algebraic[2]-states[3])/algebraic[12]
    rates[4] = (algebraic[3]-states[4])/algebraic[13]
    rates[5] = (-(algebraic[20]+algebraic[25]+algebraic[31]+algebraic[24]+algebraic[26]+algebraic[28]+-2.00000*algebraic[34])*1.00000)/(constants[8]*constants[2])
    rates[6] = (algebraic[4]-states[6])/algebraic[14]
    rates[7] = (algebraic[5]-states[7])/algebraic[15]
    rates[8] = (algebraic[6]-states[8])/algebraic[16]
    rates[9] = (algebraic[7]-states[9])/algebraic[17]
    rates[10] = (algebraic[8]-states[10])/constants[67]
    rates[11] = (algebraic[9]-states[11])/algebraic[18]
    rates[12] = algebraic[84]*(((algebraic[64]-algebraic[76])+algebraic[82]+algebraic[83])-(-2.00000*algebraic[75]+algebraic[72]+algebraic[78]+algebraic[81])/(2.00000*constants[8]*constants[2]))
    rates[13] = (constants[8]/constants[9])*((-algebraic[64]+algebraic[76])-algebraic[82])
    rates[14] = -(algebraic[57]+algebraic[65])*states[14]+algebraic[59]*states[15]+algebraic[67]*states[16]
    rates[15] = (algebraic[57]*states[14]-(algebraic[59]+algebraic[69])*states[15])+algebraic[71]*algebraic[73]
    rates[16] = (algebraic[65]*states[14]-(algebraic[67]+algebraic[61])*states[16])+algebraic[63]*algebraic[73]
    rates[17] = algebraic[83]

    return rates

def odesys(voi, states, constants, parameters):
    algebraic = computeAlgebraic(constants, states, voi)
    rates = computeRates(constants, algebraic, states, voi)

    return rates