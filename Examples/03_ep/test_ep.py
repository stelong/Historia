from Historia.ep import ep_out as epo
from Historia.ep import solver as sol
from Historia.ep.model import Gattoni as ep
from Historia.ep.model import initialize as init
from Historia.shared import plot_utils as plut
import matplotlib.pyplot as plt
import numpy as np

def main():
	## Initialize rat phenotype (choose either 'sham' or 'ab'), pacing frequency (choose either 1 or 6) and number of heart beats to simulate
	rat = 'sham'
	hz = 6
	nbeats = 1

	## Get nice colours for plotting
	blue = plut.get_col('blue')[1]
	red = plut.get_col('red')[1]

	## Solve Gattoni et al. (2016) EP model
	S = sol.EPSolution(ep, rat, hz)
	p0 = init.params(rat, hz)
	S.run2sc(p0, nbeats)

	## Vector of biomarkers characterizing the Calcium transient: [DCa, PCa, RT50, TP]
	bio = epo.A_output(S.ca)

	## Plot the Calcium transient
	fig, axis = plt.subplots(1, 1)
	axis.plot(S.t, S.ca, c=blue)
	axis.set_xlabel('Time (ms)')
	axis.set_ylabel('[Ca$^{2+}$]$_i$ ($\mu$M)')
	plt.show()

	## Fit a phenomenological curve to the simulated Calcium transient
	C = epo.PhenCalcium()
	C.fit(S.t, S.ca)
	C.get_bio()

	## Vector of biomarkers characterizing fitted curve
	bio_fit = C.bio

	## Biomarkers are close each other (the fit is good)
	print(bio)
	print(bio_fit)

	## Plot the Calcium transient and the fitted phenomenological curve
	fig, axis = plt.subplots(1, 1)
	axis.plot(S.t, S.ca, c=blue, linestyle='dashed', label='original curve')
	axis.plot(C.t, C.ca, c=red, linewidth=2, label='fitted curve')
	axis.set_xlabel('Time (ms)')
	axis.set_ylabel('[Ca$^{2+}$]$_i$ ($\mu$M)')
	plt.legend()
	plt.show()

	## We want to build new Calcium transients using the phenomenological equation by perturbing the baseline value of a specific feature
	keys = ['DCa', 'PCa', 'RT50', 'TP'] # Calcium transient biomarkers
	perc = [0.5, 0.6, 0.7, 0.8, 0.9, 1] # scaling prefactors for biomarkers (%)
	idx = 1 # Calcium transient feature to perturb
	labels = ['{}% of {} feature'.format(int(100*p), keys[idx]) for p in perc[:-1]] + ['fitted curve - unperturbed'] # vector of labels for plotting

	lsc = plut.interp_col(plut.get_col('red'), len(perc)) # get a linear interpolation between the lightest and the darkest red colour variants

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
	plt.legend()
	plt.show()

if __name__ == "__main__":
	main()