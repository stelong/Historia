import emulator
import classifier
import pickle
import abstracthm as ahm
import diversipy as dp
import numpy as np

# load dataset, emulators and classifier
# --------------------------------------
Xdata = np.loadtxt('data/mech/ab/inputs.txt', dtype=float)
Ydata = np.loadtxt('data/mech/ab/outputs.txt', dtype=float)

in_dim = Xdata.shape[1]
out_dim = Ydata.shape[1]

emulator = []
for i in range(2):
	with open('trained_emulators/mech/ab/w1_emul' + str(i+1) + '.pkl', 'rb') as f:
		emulator.append(pickle.load(f))
	f.close()

classifier = []
with open('trained_classifiers/ab/cla.pkl', 'rb') as f:
	cla = pickle.load(f)
f.close()

# pass to the new abstract base class
# -----------------------------------
AHM = []
for i in range(2):
    AHM.append(ahm.sclEmulator(emulator[i]))

# okay, let's make some wave objects
# ----------------------------------
trueX = Xdata[0].reshape([1, -1])

# zs = [np.mean(Ydata[:, i]) for i in range(out_dim)]
# vs = 0.01*np.array([np.ptp(Ydata[:, i]) for i in range(out_dim)])

# zs = [np.mean(Ydata[:, i]) for i in range(out_dim)]
# vs = [np.var(Ydata[:, i]) for i in range(out_dim)]

# # sham
# zs = [508.8, 154.6]
# vs = [1521.85, 273.27]

# ab
zs = [466.5, 125.6]
vs = [1376.41, 547.56]

cutoff = 3.0
maxno = 1
waveno = 1
LOAD = False

# subwave
# -------
subwave = []
for i in range(2):
    subwave.append(ahm.Subwave(AHM[i], zs[i], vs[i], name='y['+str(i)+']'))

# MW = ahm.Multivar(subwaves=[subwave1, subwave2], covar=np.array([[vs[0],0],[0,vs[0]]]))

# some test points
# ----------------
tests = cla.hlc_sample(100000)

# create wave object
# ------------------
W = ahm.Wave(subwaves=subwave, cm=cutoff, maxno=maxno, tests=tests)

# do history matching
# -------------------
if not(LOAD):
    W.findNROY()
    W.save('wave_' + str(waveno) + '.pkl')
else: 
    try:
        W.load('wave_' + str(waveno) + '.pkl')
    except FileNotFoundError as e:
        pass
        exit()

# random.seed(a=None, version=2)

# W.load('wave_1.pkl')
# N = 10 * in_dim
# subset = dp.subset.psa_select(W.NROY, N)
# print(subset)
# print(subset.shape)


# plotting
# --------
MINMAXwave1 = AHM[0].minmax()
# ahm.plotImp(W, grid=12, maxno=maxno, NROY=True, NIMP=False, manualRange=MINMAXwave1, vmin=1.0, sims=False, odp=True, points = trueX)
# ahm.plotImp(W, grid=12, maxno=maxno, NROY=True, NIMP=False, manualRange=MINMAXwave1, vmin=1.0, sims=True, odp=False, points = [trueX])
ahm.plotImp(W, grid=12, maxno=maxno, NROY=True, NIMP=True, manualRange=MINMAXwave1, vmin=1.0, sims=True, odp=True, points = trueX, colorbar=True)