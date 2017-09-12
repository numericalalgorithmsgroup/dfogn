# DFO-GN example: minimize the Rosenbrock function (with noise)
from __future__ import print_function
import numpy as np
import dfogn

# Define the objective function
def rosenbrock(x):
    return np.array([10.0 * (x[1] - x[0] ** 2), 1.0 - x[0]])

def rosenbrock_noisy(x):
    return rosenbrock(x) * (1.0 + 1e-2 * np.random.normal(size=(2,)))

# Define the starting point
x0 = np.array([-1.2, 1.0])

np.random.seed(0)

print("Demonstrate noise in function evaluation:")
for i in range(5):
    print("objfun(x0) = %s" % str(rosenbrock_noisy(x0)))
print("")

soln = dfogn.solve(rosenbrock_noisy, x0)

# Display output
print(" *** DFO-GN results *** ")
print("Solution xmin = %s" % str(soln.x))
print("Objective value f(xmin) = %.10g" % soln.f)
print("Needed %g objective evaluations" % soln.nf)
print("Residual vector = %s" % str(soln.resid))
print("Approximate Jacobian = %s" % str(soln.jacobian))
print("Exit flag = %g" % soln.flag)
print(soln.msg)

# Compare with a derivative-based solver
import scipy.optimize as opt

soln = opt.least_squares(rosenbrock_noisy, x0)
print("")
print(" *** SciPy results *** ")
print("Solution xmin = %s" % str(soln.x))
print("Objective value f(xmin) = %.10g" % (2.0 * soln.cost))
print("Needed %g objective evaluations" % soln.nfev)
print("Exit flag = %g" % soln.status)
print(soln.message)

