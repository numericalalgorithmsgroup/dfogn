# Data fitting example
# https://uk.mathworks.com/help/optim/ug/lsqcurvefit.html#examples

from __future__ import print_function
import numpy as np
import dfogn

tdata = np.array([0.9, 1.5, 13.8, 19.8, 24.1, 28.2, 35.2,
                  60.3, 74.6, 81.3])
ydata = np.array([455.2, 428.6, 124.1, 67.3, 43.2, 28.1, 13.1,
                  -0.4, -1.3, -1.5])

def prediction_error(x):
    return ydata - x[0] * np.exp(x[1] * tdata)

x0 = np.array([100.0, -1.0])

upper = np.array([1e20, 0.0])

soln = dfogn.solve(prediction_error, x0, upper=upper)

# Display output
print(" *** DFO-GN results *** ")
print("Solution xmin = %s" % str(soln.x))
print("Objective value f(xmin) = %.10g" % soln.f)
print("Needed %g objective evaluations" % soln.nf)
print("Exit flag = %g" % soln.flag)
print(soln.msg)

