# DFO-GN example: minimize the Rosenbrock function (with bounds)
from __future__ import print_function
import numpy as np
import dfogn

# Define the objective function
def rosenbrock(x):
    return np.array([10.0 * (x[1] - x[0] ** 2), 1.0 - x[0]])

# Define the starting point
x0 = np.array([-1.2, 1.0])

# Define optional bound constraints (a <= x <= b)
a = np.array([-10.0, -10.0])
b = np.array([0.9, 0.85])

soln = dfogn.solve(rosenbrock, x0, lower=a, upper=b)

# Display output
print(" *** DFO-GN results *** ")
print("Solution xmin = %s" % str(soln.x))
print("Objective value f(xmin) = %.10g" % soln.f)
print("Needed %g objective evaluations" % soln.nf)
print("Residual vector = %s" % str(soln.resid))
print("Approximate Jacobian = %s" % str(soln.jacobian))
print("Exit flag = %g" % soln.flag)
print(soln.msg)

