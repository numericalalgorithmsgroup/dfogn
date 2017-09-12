# DFO-GN example: Solving a nonlinear system of equations
# http://support.sas.com/documentation/cdl/en/imlug/66112/HTML/default/viewer.htm#imlug_genstatexpls_sect004.htm

from __future__ import print_function
import math
import numpy as np
import dfogn

# Want to solve:
#   x1 + x2 - x1*x2 + 2 = 0
#   x1 * exp(-x2) - 1   = 0
def nonlinear_system(x):
    return np.array([x[0] + x[1] - x[0]*x[1] + 2,
                     x[0] * math.exp(-x[1]) - 1.0])

# Warning: if there are multiple solutions, which one
#          DFO-GN returns will likely depend on x0!
x0 = np.array([0.1, -2.0])

soln = dfogn.solve(nonlinear_system, x0)

# Display output
print(" *** DFO-GN results *** ")
print("Solution xmin = %s" % str(soln.x))
print("Objective value f(xmin) = %.10g" % soln.f)
print("Needed %g objective evaluations" % soln.nf)
print("Residual vector = %s" % str(soln.resid))
print("Exit flag = %g" % soln.flag)
print(soln.msg)

