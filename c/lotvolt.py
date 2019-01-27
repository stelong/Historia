import numpy as np
import matplotlib.pyplot as plt
import nose
from assimulo.solvers import CVode
from assimulo.problem import Explicit_Problem

def f(t, y, p):
    dy1 = (p[0] - p[1]*y[1]) * y[0]
    dy2 = (p[2]*y[0] - p[3]) * y[1]

    return np.array([dy1, dy2])

def main():
    y0 = [0.5, 0.5]
    
    model = Explicit_Problem(f, y0, 0)
    
    model.p0 = [1.0, 1.0, 1.0, 1.0]

    sim = CVode(model)
    
    #Sets the solver paramters
    sim.iter = 'Newton'
    sim.discr = 'BDF'
    sim.rtol = 1.e-4
    sim.atol = np.array([1.0e-8, 1.0e-8])
    sim.sensmethod = 'SIMULTANEOUS'
    sim.suppress_sens = False
    sim.report_continuously = True

    t, y = sim.simulate(6, 1001) #Simulate 4 seconds with 400 communication points
    
    # #Basic test
    # nose.tools.assert_almost_equal(y[-1][0], 9.05518032e-01, 4)
    # nose.tools.assert_almost_equal(y[-1][1], 2.24046805e-05, 4)
    # nose.tools.assert_almost_equal(model.p_sol[0][-1][0], -1.8761, 2) #Values taken from the example in Sundials
    # nose.tools.assert_almost_equal(model.p_sol[1][-1][0], 2.9614e-06, 8)
    
    plt.plot(t, y)
    plt.show()


#-------------------------

if __name__ == "__main__":
    main()