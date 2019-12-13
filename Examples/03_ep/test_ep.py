from Historia.ep import ep_out as epo
from Historia.ep import solver as sol
from Historia.ep.model import Gattoni as ep
from Historia.ep.model import initialize as init
from Historia.shared import plot_utils as plut
import matplotlib.pyplot as plt
import numpy as np
import sys

def switch(case):
	return {
		'example1': example1,
		'example2': example2,
		'example3': example3
	}.get(case, default)

def default():
	print('Could not find case! Choose another case.')
	return

def example1():
	## case 'example1': sham and ab comparison
	
	# initialise rat phenotypes vector
	rat = ['sham', 'ab']
	# initialise pacing frequency (choose either 1 or 6) and number of heart beats to simulate
	hz = 6
	nbeats = 1

	# get nice colours for plotting
	color = [plut.get_col('blue')[1], plut.get_col('red')[1]]

	fig, axis = plt.subplots(1, 1)
	# loop over rat phenotype
	for i, r in enumerate(rat):
		# initialise the ode problem
		S = sol.EPSolution(ep, r, hz)
		# initialise the parameter vector
		p0 = init.params(r, hz)
		# run the ode solver
		S.run2sc(p0, nbeats)
		# plot the Voltage component of the solution
		axis.plot(S.t, S.v, c=color[i], label=r)
	axis.set_xlabel('Time (ms)')
	axis.set_ylabel('V (mV)')
	plt.legend()
	plt.title('Example 1 - Voltage')
	plt.show()

	fig, axis = plt.subplots(1, 1)
	# loop over rat phenotype
	for i, r in enumerate(rat):
		# initialise the ode problem
		S = sol.EPSolution(ep, r, hz)
		# initialise the parameter vector
		p0 = init.params(r, hz)
		# run the ode solver
		S.run2sc(p0, nbeats)
		# plot the Calcium concentration component of the solution
		axis.plot(S.t, S.ca, c=color[i], label=r)
	axis.set_xlabel('Time (ms)')
	axis.set_ylabel('[Ca$^{2+}$]$_i$ ($\mu$M)')
	plt.legend()
	plt.title('Example 1 - Calcium transient')
	plt.show()

	fig, axis = plt.subplots(1, 1)
	# loop over rat phenotype
	for i, r in enumerate(rat):
		# initialise the ode problem
		S = sol.EPSolution(ep, r, hz)
		# initialise the parameter vector
		p0 = init.params(r, hz)
		# run the ode solver
		S.run2sc(p0, nbeats)
		# plot the 8th component of the solution
		axis.plot(S.t, S.Y[8, :], c=color[i], label=r)
	axis.set_xlabel('Time (ms)')
	plt.legend()
	plt.title('Example 1 - $Y_8$')
	plt.show()
	return

def example2():
	## case 'example2': single EP parameter perturbation

	# initialise rat phenotype (choose either 'sham' or 'ab'), pacing frequency (choose either 1 or 6) and number of heart beats to simulate
	rat = 'sham'
	hz = 6
	nbeats = 1

	# get nice colours for plotting
	color = plut.get_col('cyan') # call without argument to see the list of available colours
	n = 8
	lsc = plut.interp_col(color, n) # linearly interpolate the chosen colour into n colour variants
	
	# loop over the number of perturbations
	for i in range(n):
		# initialise the ode problem
		S = sol.EPSolution(ep, rat, hz)
		# initialise the parameter vector
		p0 = init.params(rat, hz)
		# perturb one parameter
		p0[11] = p0[11] + (i+1)*0.05*p0[11]
		# run the ode solver
		S.run2sc(p0, nbeats)
		# plot the solution
		plt.plot(S.t, S.ca, c=lsc[i], label='param. +{}%'.format(int(100*(i+1)*0.05)))
	plt.legend()
	plt.title('Example 2 - Perturbing one specific EP parameter')
	plt.show()
	return

def example3():
	## case 'example3': building new Calcium transients from a phenomenological equation

	# initialise rat phenotype (choose either 'sham' or 'ab'), pacing frequency (choose either 1 or 6) and number of heart beats to simulate
	rat = 'sham'
	hz = 6
	nbeats = 1

	# get nice colours for plotting
	blue = plut.get_col('blue')[1]
	red = plut.get_col('red')[1]

	# solve Gattoni et al. (2016) EP model
	S = sol.EPSolution(ep, rat, hz)
	p0 = init.params(rat, hz)
	S.run2sc(p0, nbeats)

	# vector of biomarkers characterising the Calcium transient: [DCa, PCa, RT50, TP]
	bio = epo.A_output(S.ca)

	# plot the Calcium transient
	fig, axis = plt.subplots(1, 1)
	axis.plot(S.t, S.ca, c=blue)
	axis.set_xlabel('Time (ms)')
	axis.set_ylabel('[Ca$^{2+}$]$_i$ ($\mu$M)')
	plt.title('Example 3 - Calcium transient')
	plt.show()

	# fit a phenomenological curve to the simulated Calcium transient
	C = epo.PhenCalcium()
	C.fit(S.t, S.ca)
	C.get_bio()

	# vector of biomarkers characterising the fitted curve
	bio_fit = C.bio

	# biomarkers are close each other (the fit is good)
	print('\n=== Simulated Ca biomarkers: {}'.format(bio))
	print('=== Fitted Ca biomarkers: {}\n'.format(bio_fit))

	# plot the Calcium transient and the fitted phenomenological curve
	fig, axis = plt.subplots(1, 1)
	axis.plot(S.t, S.ca, c=blue, linestyle='dashed', label='original curve')
	axis.plot(C.t, C.ca, c=red, linewidth=2, label='fitted curve')
	axis.set_xlabel('Time (ms)')
	axis.set_ylabel('[Ca$^{2+}$]$_i$ ($\mu$M)')
	plt.title('Example 3 - Simulated Ca transient Vs Fitted Ca transient')
	plt.legend()
	plt.show()

	# we want to build new Calcium transients using the phenomenological equation by perturbing the baseline value of a specific feature
	keys = ['DCa', 'PCa', 'RT50', 'TP'] # Calcium transient biomarkers
	perc = [0.5, 0.6, 0.7, 0.8, 0.9, 1] # scaling prefactors for biomarkers (%)
	idx = 0 # Calcium transient feature to perturb
	labels = ['{}% of {} feature'.format(int(100*p), keys[idx]) for p in perc[:-1]] + ['fitted curve - unperturbed'] # vector of labels for plotting

	lsc = plut.interp_col(plut.get_col('red'), len(perc)) # linearly interpolate the red colour into len(perc) variants

	a0 = np.copy(np.array(C.a)) # store baseline phenomenological equation parameters
	
	fig, axis = plt.subplots(1, 1)
	axis.plot(S.t, S.ca, c=blue, linestyle='dashed', label='original curve')
	for i, p in enumerate(perc):
		a = np.copy(a0)
		a[idx] = p*a[idx]
		C = epo.PhenCalcium()
		C.build_ca(S.t, a) # build a new Calcium transient using the pertubed set of parameters
		C.get_bio()
		if C.valid:
			C.check_ca()
			if C.valid:
				axis.plot(C.t, C.ca, c=lsc[i], linewidth=2, label=labels[i])
	axis.set_xlabel('Time (ms)')
	axis.set_ylabel('[Ca$^{2+}$]$_i$ ($\mu$M)')
	plt.title('Example 3 - Perturbing one specific Ca feature')
	plt.legend()
	plt.show()
	return

def main():
	## Available examples are 'example1', 'example2', 'example3'
	case = sys.argv[1]
	switch(case)()

if __name__ == "__main__":
	main()