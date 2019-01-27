import numpy as N
import pylab as P
import nose
from assimulo.solvers import Dopri5
from assimulo.problem import Explicit_Problem

def run_example(with_plots=True):

    def f(t,y):
        ydot = -y[0]
        return N.array([ydot])

    #Define an Assimulo problem
    exp_mod = Explicit_Problem(f, 4.0)
    
    exp_sim = Dopri5(exp_mod) #Create a Dopri5 solver

    #Simulate
    t, y = exp_sim.simulate(5) #Simulate 5 seconds
    
    #Plot
    if with_plots:
        P.plot(t,y)
        P.title(exp_mod.name)
        P.xlabel('Time')
        P.ylabel('State')
        P.show()
        
    return exp_mod, exp_sim


if __name__=='__main__':
    mod,sim = run_example()