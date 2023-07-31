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

"""

# Ensure compatibility with Python 2
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import unittest

from dfogn.util import *


def array_compare(x, y, thresh=1e-14):
    return np.max(np.abs(x - y)) < thresh


class TestSumsq(unittest.TestCase):
    def runTest(self):
        n = 10
        x = np.sin(np.arange(n))
        normx = np.sum(x**2)
        self.assertAlmostEqual(normx, sumsq(x), msg='Wrong answer')


class TestEval(unittest.TestCase):
    def runTest(self):
        objfun = lambda x : np.array([10*(x[1]-x[0]**2), 1-x[0]])
        x = np.array([-1.2, 1.0])
        fvec, f = eval_least_squares_objective(objfun, x)
        self.assertTrue(np.all(fvec == objfun(x)), 'Residuals wrong')
        self.assertAlmostEqual(f, sumsq(fvec), 'Sum of squares wrong')


class TestModelValue(unittest.TestCase):
    def runTest(self):
        n = 5
        A = np.arange(n ** 2, dtype=float).reshape((n, n))
        H = np.sin(A + A.T)  # force symmetric
        hess = to_upper_triangular_vector(H)
        vec = np.exp(np.arange(n, dtype=float))
        g = np.cos(3*np.arange(n, dtype=float) - 2.0)
        mval = np.dot(g, vec) + 0.5 * np.dot(vec, np.dot(H, vec))
        self.assertAlmostEqual(mval, calculate_model_value(g, hess, vec), 'Wrong value')


class TestInitFromMatrix(unittest.TestCase):
    def runTest(self):
        n = 3
        nvals = n*(n+1)//2
        A = np.arange(n**2, dtype=float).reshape((n,n))
        hess = to_upper_triangular_vector(A+A.T)  # force symmetric
        self.assertEqual(len(hess), nvals, 'Wrong length')
        self.assertTrue(np.all(hess == np.array([0.0, 4.0, 8.0, 8.0, 12.0, 16.0])),
                        'Wrong initialised values')


class TestToFull(unittest.TestCase):
    def runTest(self):
        n = 7
        A = np.arange(n ** 2, dtype=float).reshape((n, n))
        H = A + A.T  # force symmetric
        hess = to_upper_triangular_vector(H)
        self.assertTrue(np.all(to_full_matrix(n, hess) == H), 'Wrong values')


class TestGetElementGood(unittest.TestCase):
    def runTest(self):
        n = 3
        A = np.arange(n ** 2, dtype=float).reshape((n, n))
        H = A + A.T  # force symmetric
        hess = to_upper_triangular_vector(H)
        for i in range(n):
            for j in range(n):
                self.assertEqual(get_hessian_element(n, hess, i, j), H[i,j],
                                 'Wrong value for (i,j)=(%g,%g): got %g, expecting %g'
                                 % (i, j, get_hessian_element(n, hess, i, j), H[i,j]))


class TestMultGood(unittest.TestCase):
    def runTest(self):
        n = 5
        A = np.arange(n ** 2, dtype=float).reshape((n, n))
        H = np.sin(A + A.T)  # force symmetric
        hess = to_upper_triangular_vector(H)
        vec = np.exp(np.arange(n, dtype=float))
        hs = np.dot(H, vec)
        self.assertTrue(array_compare(right_multiply_hessian(hess, vec), hs, thresh=1e-12), 'Wrong values')
