from Historia.cellcontr import solver as sol
from Historia.shared import plot_utils as plut
import json
import matplotlib.pyplot as plt
import numpy as np

def main():
	rat = 'sham' # choose rat phenotype

	if rat == 'sham':
		calcium = [0.472428, 0.484626, 0.499103, 0.517363, 0.543670, 0.646212, 0.750755, 0.848459, 0.939326, 1.023355, 1.100546, 1.170899, 1.234415, 1.291092, 1.340932, 1.383934, 1.420098, 1.449424, 1.471912, 1.487563, 1.496375, 1.498350, 1.493487, 1.481786, 1.465424, 1.449134, 1.433076, 1.417248, 1.401645, 1.386265, 1.371105, 1.356161, 1.341430, 1.326910, 1.312597, 1.298489, 1.284581, 1.270873, 1.257360, 1.244040, 1.230910, 1.217967, 1.205210, 1.192634, 1.180238, 1.168019, 1.155974, 1.144102, 1.132398, 1.120862, 1.109491, 1.098282, 1.087233, 1.076341, 1.065605, 1.055023, 1.044591, 1.034309, 1.024173, 1.014182, 1.004333, 0.994625, 0.985056, 0.975623, 0.966325, 0.957160, 0.948126, 0.939220, 0.930442, 0.921789, 0.913259, 0.904852, 0.896564, 0.888395, 0.880342, 0.872404, 0.864580, 0.856867, 0.849264, 0.841770, 0.834383, 0.827101, 0.819924, 0.812848, 0.805874, 0.798999, 0.792223, 0.785543, 0.778959, 0.772468, 0.766070, 0.759764, 0.753548, 0.747420, 0.741380, 0.735426, 0.729557, 0.723772, 0.718069, 0.712448, 0.706907, 0.701445, 0.696061, 0.690754, 0.685523, 0.680366, 0.675283, 0.670273, 0.665334, 0.660466, 0.655667, 0.650937, 0.646274, 0.641678, 0.637147, 0.632681, 0.628279, 0.623939, 0.619662, 0.615446, 0.611290, 0.607193, 0.603154, 0.599174, 0.595250, 0.591382, 0.587569, 0.583811, 0.580107, 0.576455, 0.572856, 0.569307, 0.565810, 0.562362, 0.558964, 0.555614, 0.552312, 0.549057, 0.545849, 0.542686, 0.539569, 0.536496, 0.533467, 0.530481, 0.527538, 0.524637, 0.521777, 0.518958, 0.516179, 0.513440, 0.510740, 0.508079, 0.505456, 0.502870, 0.500321, 0.497808, 0.495331, 0.492890, 0.490483, 0.488111, 0.485773, 0.483468, 0.481196, 0.478956, 0.476749, 0.474573, 0.472428]
		color = 'blue'
	else:
		calcium = [0.135408, 0.143914, 0.155570, 0.178474, 0.309279, 0.430894, 0.543392, 0.646771, 0.741033, 0.826177, 0.902204, 0.969112, 1.026903, 1.075576, 1.115132, 1.145569, 1.166889, 1.179092, 1.182176, 1.176143, 1.162072, 1.146941, 1.131996, 1.117236, 1.102658, 1.088259, 1.074038, 1.059992, 1.046120, 1.032418, 1.018885, 1.005520, 0.992319, 0.979280, 0.966403, 0.953684, 0.941122, 0.928715, 0.916461, 0.904358, 0.892404, 0.880598, 0.868937, 0.857420, 0.846045, 0.834810, 0.823714, 0.812754, 0.801930, 0.791239, 0.780680, 0.770251, 0.759951, 0.749778, 0.739730, 0.729806, 0.720004, 0.710323, 0.700762, 0.691318, 0.681991, 0.672779, 0.663680, 0.654694, 0.645818, 0.637052, 0.628394, 0.619843, 0.611397, 0.603055, 0.594816, 0.586679, 0.578642, 0.570704, 0.562864, 0.555121, 0.547473, 0.539919, 0.532459, 0.525090, 0.517812, 0.510625, 0.503525, 0.496513, 0.489588, 0.482748, 0.475993, 0.469320, 0.462730, 0.456221, 0.449793, 0.443444, 0.437172, 0.430979, 0.424861, 0.418820, 0.412852, 0.406958, 0.401137, 0.395388, 0.389709, 0.384101, 0.378561, 0.373090, 0.367687, 0.362350, 0.357078, 0.351872, 0.346730, 0.341652, 0.336636, 0.331681, 0.326788, 0.321956, 0.317183, 0.312468, 0.307812, 0.303213, 0.298671, 0.294185, 0.289754, 0.285378, 0.281056, 0.276787, 0.272571, 0.268407, 0.264294, 0.260231, 0.256219, 0.252257, 0.248343, 0.244477, 0.240659, 0.236888, 0.233164, 0.229486, 0.225853, 0.222264, 0.218720, 0.215220, 0.211763, 0.208348, 0.204976, 0.201645, 0.198355, 0.195106, 0.191897, 0.188727, 0.185596, 0.182504, 0.179451, 0.176434, 0.173455, 0.170513, 0.167607, 0.164737, 0.161902, 0.159102, 0.156337, 0.153606, 0.150908, 0.148244, 0.145613, 0.143014, 0.140447, 0.137912, 0.135408]
		color = 'red'


	c_dict = {}
	with open('parameters.json') as f:
		c_dict = json.load(f)[rat]

	#----------------------------------------------------------------
	# EXAMPLE 1

	C = sol.CONTRSolution(calcium, c_dict)
	C.solver_sol()

	plt.figure()
	plt.plot(C.t, C.T)
	plt.show()

	Cai = np.linspace(1e-1, 1e1, 10000)
	pCai = -np.log10(1e-6*Cai)

	C_ss = sol.CONTRSolution(Cai, c_dict)
	C_ss.steadystate_sol()

	plt.figure()
	plt.plot(Cai, C_ss.TRPN['TRPN'])
	plt.scatter(C_ss.TRPN['EC50'], 0.5*(C_ss.TRPN['TRPN'].min()+C_ss.TRPN['TRPN'].max()))
	plt.gca().invert_xaxis()
	plt.show()
	print(C_ss.TRPN['EC50'])

	plt.figure()
	plt.plot(Cai, C_ss.XB['XB'])
	plt.scatter(C_ss.XB['EC50'], 0.5*(C_ss.XB['XB'].min()+C_ss.XB['XB'].max()))
	plt.gca().invert_xaxis()
	plt.show()
	print(C_ss.XB['EC50'])

	plt.figure()
	plt.plot(pCai, C_ss.F['F'])
	plt.scatter(C_ss.F['pCa50'], 0.5*(C_ss.F['F'].min()+C_ss.F['F'].max()))
	plt.gca().invert_xaxis()
	plt.show()

	#----------------------------------------------------------------
	# EXAMPLE 2

	# C = sol.CONTRSolution(calcium, c_dict)

	# params = ['lambda', 'dlambda_dt', 'ca50ref', 'Tref', 'TRPN50', 'nTRPN', 'kon', 'koff', 'nXB', 'kXB', 'beta0', 'beta1', 'a', 'A1', 'A2', 'alpha1', 'alpha2']

	# idx_par = 2 # choose parameter to alter (e.g. 2 --> 'ca50ref')

	# p0 = C.constant[params[idx_par]]

	# perc = 0.2 # choose percentage of +/- alteration from baseline
	# n_vals = 8
	# p = np.linspace(p0*(1-perc), p0*(1+perc), n_vals)
	# col = plut.interp_col(plut.get_col(color), n_vals+1)

	# fig, axes = plt.subplots(2, 1, figsize=(1.5*8.27, 1.5*11.69))

	# Cai = np.linspace(1e-1, 1e1, 1000)
	# pCai = -np.log10(1e-6*Cai)

	# gray = plut.get_col('gray')[1]

	# C0_ss = sol.CONTRSolution(Cai, c_dict)
	# C0_ss.steadystate_sol()
	# axes[0].plot(pCai, C0_ss.F['F'], c='k', linestyle='dashed', label='{} = {:.4f}'.format(params[idx_par], p0))
	# axes[0].scatter(C0_ss.F['pCa50'], C0_ss.constant['Tref']/2, facecolors='k', zorder=2)
	# axes[0].fill_between([-np.log10(1e-6*np.array(calcium)).max(), -np.log10(1e-6*np.array(calcium)).min()], 0, C0_ss.constant['Tref'], color=gray, alpha=0.15, label='$[Ca^{2+}]_i$ phys. range')

	# C0 = sol.CONTRSolution(calcium, c_dict)
	# C0.solver_sol()
	# axes[1].plot(C0.t, C0.T, c='k', linestyle='dashed', label='{} = {:.4f}'.format(params[idx_par], p0))
	
	# for i, pi in enumerate(p):
	# 	p_dict = {params[idx_par]: pi}
	# 	C0_ss.steadystate_sol(p_dict)
	# 	axes[0].plot(pCai, C0_ss.F['F'], c=col[-(i+1)], label='{} = {:.4f}'.format(params[idx_par], pi))
	# 	axes[0].scatter(C0_ss.F['pCa50'], C0_ss.constant['Tref']/2, facecolors=col[-(i+1)], zorder=2)

	# 	C0.solver_sol(p_dict)
	# 	axes[1].plot(C0.t, C0.T, c=col[-(i+1)], label='{} = {:.4f}'.format(params[idx_par], pi))	

	# axes[0].set_xlim([pCai[-1], pCai[0]])
	# axes[0].invert_xaxis()
	# axes[0].set_xlabel('pCa (log10(M))', fontsize=12)
	# axes[0].set_ylabel('Force (kPa)', fontsize=12)
	# axes[0].set_title('Force vs pCa', fontweight='bold', fontsize=14)
	# axes[0].legend(loc='lower right')

	# axes[1].set_xlim([C0.t[0], C0.t[-1]])
	# axes[1].set_xlabel('Time (ms)', fontsize=12)
	# axes[1].set_ylabel('Force (kPa)', fontsize=12)
	# axes[1].set_title('Force vs Time', fontweight='bold', fontsize=14)
	# axes[1].legend()
	
	# plt.show()

	#--------------------------------------------------------------------------------------------------------
	# EXAMPLE 3: we can possibly vary two or more parameters at a time (uncomment what follows)

	# idx_par1 = 2
	# idx_par2 = 7

	# C = sol.CONTRSolution(calcium, c_dict)

	# p1_0 = C.constant[params[idx_par1]]
	# p2_0 = C.constant[params[idx_par2]]

	# perc = 0.2
	# n_vals = 4
	# p1 = np.linspace(p1_0*(1-perc), p1_0*(1+perc), n_vals)	
	# p2 = np.linspace(p2_0*(1-perc), p2_0*(1+perc), n_vals)

	# xv, yv = np.meshgrid(p1, p2) # 2D grid to be simulated

	# plt.scatter(xv, yv)
	# plt.xlabel(params[idx_par1], fontsize=12)
	# plt.ylabel(params[idx_par2], fontsize=12)
	# plt.show()

	# col = plut.interp_col(plut.get_col('orange'), n_vals*n_vals)

	# fig, axis = plt.subplots(1, 1)
	# i = 0
	# for p1_i in p1:
	# 	for p2_i in p2:
	# 		i += 1
	# 		p_dict = {params[idx_par1]: p1_i, params[idx_par2]: p2_i}
	# 		C.solver_sol(p_dict)
	# 		axis.plot(C.t, C.T, c=col[-i], label='{} = {:.4f}, {} = {:.4f}'.format(params[idx_par1], p1_i, params[idx_par2], p2_i))
	# plt.xlim([C.t[0], C.t[-1]])
	# plt.xlabel('Time (ms)', fontsize=12)
	# plt.ylabel('Force (kPa)', fontsize=12)
	# plt.legend()
	# plt.show()

if __name__ == '__main__':
    main()