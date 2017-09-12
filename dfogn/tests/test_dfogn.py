"""

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.

The development of this software was sponsored by NAG Ltd. (http://www.nag.co.uk)
and the EPSRC Centre For Doctoral Training in Industrially Focused Mathematical
Modelling (EP/L015803/1) at the University of Oxford. Please contact NAG for
alternative licensing.

Copyright 2017, Lindon Roberts

"""

# Ensure compatibility with Python 2
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import unittest

import dfogn


def array_compare(x, y, thresh=1e-14):
    return np.max(np.abs(x - y)) < thresh


def rosenbrock(x):
    return np.array([10.0 * (x[1] - x[0] ** 2), 1.0 - x[0]])

def rosenbrock_jacobian(x):
    return np.array([[-20.0*x[0], 10.0], [-1.0, 0.0]])

class TestRosenbrockGeneric(unittest.TestCase):
    # Minimise the (2d) Rosenbrock function
    def runTest(self):
        # n, m = 2, 2
        x0 = np.array([-1.2, 1.0])
        soln = dfogn.solve(rosenbrock, x0)
        self.assertTrue(array_compare(soln.x, np.array([1.0, 1.0]), thresh=1e-4), "Wrong xmin")
        self.assertTrue(array_compare(soln.resid, rosenbrock(soln.x), thresh=1e-10), "Wrong resid")
        self.assertTrue(array_compare(soln.jacobian, rosenbrock_jacobian(soln.x), thresh=1e-2), "Wrong Jacobian")
        self.assertTrue(abs(soln.f) < 1e-10, "Wrong fmin")


class TestRosenbrockBounds(unittest.TestCase):
    # Minimise the (2d) Rosenbrock function, where x[1] hits the upper bound
    def runTest(self):
        # n, m = 2, 2
        x0 = np.array([-1.2, 0.7])  # standard start point does not satisfy bounds
        lower = np.array([-2.0, -2.0])
        upper = np.array([1.0, 0.9])
        xmin = np.array([0.9486, 0.9])  # approximate
        fmin = np.dot(rosenbrock(xmin), rosenbrock(xmin))
        soln = dfogn.solve(rosenbrock, x0, lower=lower, upper=upper)
        print(soln.x)
        self.assertTrue(array_compare(soln.x, xmin, thresh=1e-2), "Wrong xmin")
        self.assertTrue(array_compare(soln.resid, rosenbrock(soln.x), thresh=1e-10), "Wrong resid")
        self.assertTrue(array_compare(soln.jacobian, rosenbrock_jacobian(soln.x), thresh=1e-2), "Wrong Jacobian")
        self.assertTrue(abs(soln.f - fmin) < 1e-4, "Wrong fmin")


class TestRosenbrockBounds2(unittest.TestCase):
    # Minimise the (2d) Rosenbrock function, where x[0] hits upper bound
    def runTest(self):
        # n, m = 2, 2
        x0 = np.array([-1.2, 0.7])  # standard start point too close to upper bounds
        lower = np.array([-2.0, -2.0])
        upper = np.array([0.9, 0.9])
        xmin = np.array([0.9, 0.81])  # approximate
        fmin = np.dot(rosenbrock(xmin), rosenbrock(xmin))
        soln = dfogn.solve(rosenbrock, x0, lower=lower, upper=upper)
        print(soln.x)
        self.assertTrue(array_compare(soln.x, xmin, thresh=1e-2), "Wrong xmin")
        self.assertTrue(array_compare(soln.resid, rosenbrock(soln.x), thresh=1e-10), "Wrong resid")
        self.assertTrue(array_compare(soln.jacobian, rosenbrock_jacobian(soln.x), thresh=1e-2), "Wrong Jacobian")
        self.assertTrue(abs(soln.f - fmin) < 1e-4, "Wrong fmin")


class TestLinear(unittest.TestCase):
    # Solve min_x ||Ax-b||^2, for some random A and b
    def runTest(self):
        n, m = 2, 5
        np.random.seed(0)  # (fixing random seed)
        A = np.random.rand(m, n)
        b = np.random.rand(m)
        objfun = lambda x: np.dot(A, x) - b
        xmin = np.linalg.lstsq(A, b)[0]
        fmin = np.dot(objfun(xmin), objfun(xmin))
        x0 = np.zeros((n,))
        soln = dfogn.solve(objfun, x0)
        self.assertTrue(array_compare(soln.x, xmin, thresh=1e-2), "Wrong xmin")
        self.assertTrue(array_compare(soln.resid, objfun(soln.x), thresh=1e-10), "Wrong resid")
        self.assertTrue(array_compare(soln.jacobian, A, thresh=1e-2), "Wrong Jacobian")
        self.assertTrue(abs(soln.f - fmin) < 1e-4, "Wrong fmin")
