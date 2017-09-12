"""
Util
==================
A set of useful extra functions for DFO-GN.

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
import logging
import sys

zhang_code_structure = True  # True = Zhang version, False = NAG/BOBYQA version
bbqtr = False  # BOBYQA trust region update or Zhang paper trust region update (only used when zhang_code_structure = False)


def sumsq(x):
    # There are several ways to calculate sum of squares of a vector:
    #   np.dot(x,x)
    #   np.sum(x**2)
    #   np.sum(np.square(x))
    #   etc.
    # Using the timeit routine, it seems like dot(x,x) is ~3-4x faster than other methods
    return np.dot(x, x)


def get_vector_max(x):
    # Get k and x[k] with max value of x
    idx = np.argmax(x)
    return idx, x[idx]


def get_vector_min(x):
    # Get k and x[k] with min value of x
    idx = np.argmin(x)
    return idx, x[idx]


def all_square_distances(xpt, xopt):
    # Return vector of squared Euclidean distances between each row of xpt and xopt
    npt, n = xpt.shape
    assert xopt.size == n, "xpt and xopt have incompatible sizes"
    all_sq_dist = np.zeros((npt,))
    for k in range(npt):
        all_sq_dist[k] = sumsq(xpt[k, :]-xopt)
    return all_sq_dist


def to_upper_triangular_vector(A):
    n = A.shape[0]
    assert A.shape == (n,n), "A must be a square matrix"

    hq = np.zeros((n*(n+1)//2,))
    ih = -1
    for j in range(n):  # j = 0, ..., n-1
        for i in range(j + 1):  # i = 0, ..., j
            ih += 1
            hq[ih] = A[i,j]
    return hq


def to_full_matrix(n, hq):
    assert hq.size == n * (n + 1) // 2, "hq has wrong size given input n"
    A = np.zeros((n,n))
    ih = -1
    for j in range(n):  # j = 0, ..., n-1
        for i in range(j + 1):  # i = 0, ..., j
            ih += 1
            A[i, j] = hq[ih]
            A[j, i] = hq[ih]
    return A


def eval_least_squares_objective(objfun, x, verbose=True, eval_num=0):
    # Evaluate least squares function
    fvec = objfun(x)
    f = sumsq(fvec)  # objective = sum(ri^2) [no 1/2 factor at front]

    if verbose:
        if len(x) < 6:
            logging.info("Function eval %i has f = %.15g at x = " % (eval_num, f) + str(x))
        else:
            logging.info("Function eval %i has f = %.15g at x = [...]" % (eval_num, f))

    return fvec, f


def eval_least_squares_objective_v2(objfun, x, verbose=True, eval_num=0, pt_num=0, full_x_thresh=6):
    # Evaluate least squares function
    fvec = objfun(x)

    try:
        if np.max(np.abs(fvec)) >= np.sqrt(sys.float_info.max):
            f = sys.float_info.max
        else:
            f = sumsq(fvec)  # objective = sum(ri^2) [no 1/2 factor at front]
    except OverflowError:
        f = sys.float_info.max

    if verbose:
        if len(x) < full_x_thresh:
            logging.info("Function eval %i at point %i has f = %.15g at x = " % (eval_num, pt_num, f) + str(x))
        else:
            logging.info("Function eval %i at point %i has f = %.15g at x = [...]" % (eval_num, pt_num, f))

    return fvec, f


def calculate_model_value(gopt, hq, s):
    # Calculate model value (s^T * gopt + 0.5* s^T * H * s) = s^T * (gopt + 0.5 * H*s)
    assert gopt.shape == s.shape, "gopt and s have incompatible sizes"
    Hs = right_multiply_hessian(hq, s)
    return np.dot(s, gopt + 0.5*Hs)


def right_multiply_hessian(hq, s):
    # Multiply H*s where H is Hessian defined by hq
    n = s.size
    assert hq.size == n * (n + 1) // 2, "hq and s have incompatible sizes"
    hs = np.zeros((n,))

    ih = -1
    for j in range(n):  # j = 0, ..., n-1
        for i in range(j + 1):  # i = 0, ..., j
            ih += 1
            if i < j:
                hs[j] += hq[ih] * s[i]
            hs[i] += hq[ih] * s[j]

    return hs


def get_hessian_element(n, hq, i, j):
    # Get element (i,j) of Hessian, for i,j=0,...,n-1
    assert hq.size == n * (n + 1) // 2, "hq has wrong size given input n"
    assert 0 <= i <= n-1, "i must be in 0, ..., n-1"
    assert 0 <= j <= n-1, "j must be in 0, ..., n-1"
    ih = -1
    for k1 in range(n):
        for k2 in range(k1+1):
            ih += 1
            if (k1==i and k2==j) or (k1==j and k2==i):
                return hq[ih]
    return None


def d_within_bounds(d, xopt, sl, su, xbdi):
    # Used in TRSBOX, force d to be within bounds
    # In Fortran code, is at label 190
    xnew = np.maximum(np.minimum(xopt + d, su), sl)
    xnew[xbdi == -1] = sl[xbdi == -1]
    xnew[xbdi == 1] = su[xbdi == 1]
    d = xnew - xopt
    return d
