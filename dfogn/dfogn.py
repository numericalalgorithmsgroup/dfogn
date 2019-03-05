"""
DFO-GN
====================
A derivative-free solver for least squares minimisation with bound constraints

Call structure is:
    x, f, nf, exit_flag, exit_str = dfogn(objfun, x0, lower, upper,
                                          maxfun, init_tr_radius, rhoend=1e-8)

Required inputs:
    objfun          Objective function, callable as: residual_vector = objfun(x)
    x0              Initial starting point, NumPy ndarray
Optional inputs:
    lower, upper    Lower and upper bound constraints (lower <= x <= upper),
                    must be NumPy ndarrays of same size as x0 (default +/-1e20)
    maxfun          Maximum number of allowable function evalutions (default 1000)
    init_tr_radius  Initial trust region radius (default 0.1*max(1, ||x0||_infty)
    rhoend          Termination condition on trust region radius (default 1e-8)

Outputs:
    x               Estimate of minimiser
    f               Value of least squares objective at x (f = ||objfun(x)||^2)
    nf              Number of objective evaluations used to find x
    exit_flag       Integer flag indicating termination criterion (see list below imports)
    exit_str        String with more detailed termination message


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

import logging
from math import sqrt
import numpy as np
import scipy.linalg as sp_linalg
import warnings

from .util import *
from .trust_region import *
from .alternative_move import *
from .linear_altmov import *

__all__ = ['solve', 'EXIT_SUCCESS', 'EXIT_INPUT_ERROR', 'EXIT_MAXFUN_WARNING', 'EXIT_TR_INCREASE_ERROR',
           'EXIT_LINALG_ERROR', 'EXIT_ALTMOV_MEMORY_ERROR']

#######################
# Exit codes
EXIT_SUCCESS = 0  # successful finish (rho=rhoend or sufficient objective reduction)
EXIT_INPUT_ERROR = 1  # error, bad inputs
EXIT_MAXFUN_WARNING = 2  # warning, reached max function evals
EXIT_TR_INCREASE_ERROR = 3  # error, trust region step increased model value
EXIT_LINALG_ERROR = 4  # error, linalg error (singular matrix encountered)
EXIT_ALTMOV_MEMORY_ERROR = 5  # error, stpsav issue in ALTMOV
#######################


class OptimResults:
    def __init__(self, xmin, rmin, fmin, jacmin, nf, exit_flag, exit_msg):
        self.x = xmin
        self.resid = rmin
        self.f = fmin
        self.jacobian = jacmin
        self.nf = nf
        self.flag = exit_flag
        self.msg = exit_msg
        # Set standard names for exit flags
        self.EXIT_MAXFUN_WARNING = EXIT_MAXFUN_WARNING
        self.EXIT_SUCCESS = EXIT_SUCCESS
        self.EXIT_INPUT_ERROR = EXIT_INPUT_ERROR
        self.EXIT_TR_INCREASE_ERROR = EXIT_TR_INCREASE_ERROR
        self.EXIT_LINALG_ERROR = EXIT_LINALG_ERROR
        self.EXIT_ALTMOV_MEMORY_ERROR = EXIT_ALTMOV_MEMORY_ERROR


class Model:
    def __init__(self, n, m, npt, x0, xl, xu):
        assert npt==n+1, "Require strictly linear model"
        # Problem sizes
        self.n = n
        self.m = m
        self.npt = npt

        # Actual model info
        # Here, the model for each residual is centred around xbase
        #    m(x) = model_const_term + gqv*(x-xbase)

        self.kbase = 0  # index of base point
        self.xbase = x0  # base point
        self.xl = xl  # lower bounds (absolute terms)
        self.xu = xu  # upper bounds (absolute terms)
        self.sl = xl - x0  # lower bounds (adjusted for xbase), should be -ve (actually < -rhobeg)
        self.su = xu - x0  # upper bounds (adjusted for xbase), should be +ve (actually > rhobeg)
        self.xpt = np.zeros((npt, n))  # interpolation points
        self.fval_v = np.zeros((npt, m))  # residual vectors at each xpt(+xbase)
        self.fval = np.zeros((npt, ))  # total sum of squares at each xpt(+xbase)
        self.model_const_term_v = np.zeros((m,))  # constant term of each mini-model
        self.gqv = np.zeros((n, m))  # interpolated gradients for each mini-model

        self.kopt = None  # index of current best x

        self.fbeg = None  # initial sum of squares at x0

        self.xsave = None  # possible final return value (abs coords)
        self.rsave = None  # residuals for possible final return value
        self.fsave = None  # sum of squares for final return value
        self.jacsave = None  # approximate Jacobian at possible final return value

        self.lu = None  # LU decomp of interp matrix
        self.piv = None  # pivots for LU decomposition of interp matrix
        self.lu_current = False  # whether current LU factorisation of interp matrix is up-to-date or not

        self.EXACT_CONST_TERM = True  # use exact c=r(xopt) for interpolation (improve conditioning)
        # Affects mini-model interpolation / interpolation matrix, but also geometry updating

    def x_within_bounds(self, k=None, x=None):
        # Get x value for k-th point or x vector (in absolute terms, force within bounds)
        if k is not None:
            return np.minimum(np.maximum(self.xl, self.xbase + self.xpt[k, :]), self.xu)
        elif x is not None:
            return np.minimum(np.maximum(self.xl, self.xbase + x), self.xu)
        else:
            return None

    def xopt(self):
        # Current best x (relative to xbase)
        return self.xpt[self.kopt, :].copy()

    def fval_v_opt(self):
        return self.fval_v[self.kopt,:]

    def fval_opt(self):
        return self.fval[self.kopt]

    def update_point(self, knew, xnew, v_err, f):
        # Add point xnew with objective vector v_err (full objective f) at the knew-th index
        self.xpt[knew,:] = xnew
        self.fval_v[knew, :] = v_err
        self.fval[knew] = f

        # Update XOPT, GOPT and KOPT if the new calculated F is less than FOPT.
        if f < self.fval_opt():
            self.kopt = knew

        self.lu_current = False
        return

    def gqv_at_xopt(self):
        return self.gqv

    def shift_base(self, xbase_shift):
        for m1 in range(self.m):
            self.model_const_term_v[m1] += np.dot(self.gqv[:, m1], xbase_shift)

        # The main updates
        for k in range(self.npt):
            self.xpt[k, :] = self.xpt[k, :] - xbase_shift
        self.xbase += xbase_shift
        self.sl = self.sl - xbase_shift
        self.su = self.su - xbase_shift

        self.lu_current = False
        self.factorise_LU()

        return

    def interpolate_mini_models(self):
        # Build interpolation matrix and factorise (in self.lu, self.piv)
        try:
            self.factorise_LU()
            if self.EXACT_CONST_TERM:
                idx_to_use = [k for k in range(self.npt) if k != self.kopt]
                for m1 in range(self.m):
                    rhs = np.zeros((self.n,))
                    for i in range(self.n):
                        k = idx_to_use[i]
                        rhs[i] = self.fval_v[k, m1] - self.fval_v[self.kopt, m1] - \
                                 np.dot(self.gqv[:, m1], self.xpt[k, :] - self.xopt())
                    soln = sp_linalg.lu_solve((self.lu, self.piv), rhs)
                    self.gqv[:, m1] += soln  # whole solution is gradient

                # shift constant term back
                self.model_const_term_v = self.fval_v[self.kopt, :] - np.dot(self.gqv.T, self.xopt())
                return True  # flag ok
            else:
                model_values_v = np.zeros((self.npt, self.m))
                for k in range(self.npt):
                    model_values_v[k, :] = self.predicted_values(self.xpt[k, :], d_based_at_xopt=False,
                                                                 with_const_term=True)

                # Sometimes when things get too close to a solution, we can get NaNs in model_values - flag error & quit
                if np.any(np.isnan(model_values_v)):
                    self.gqv = None
                    return False  # flag error

                for m1 in range(self.m):
                    rhs = self.fval_v[:, m1] - model_values_v[:, m1]
                    soln = sp_linalg.lu_solve((self.lu, self.piv), rhs)
                    self.model_const_term_v[m1] += soln[0]
                    self.gqv[:, m1] += soln[1:]  # first term is constant, rest is gradient term

                return True  # flag ok
        except np.linalg.LinAlgError:
            self.gqv = None
            return False  # flag error
        except ValueError:  # happens when LU decomposition has Inf or NaN
            self.gqv = None
            return False  # flag error

    def factorise_LU(self):
        if not self.lu_current:
            Wmat = self.build_interp_matrix()
            self.lu, self.piv = sp_linalg.lu_factor(Wmat)  # LU has L and U parts, piv indicates row swaps for pivoting
            self.lu_current = True
        return

    def solve_LU(self, rhs):
        # If lu_current, use that, otherwise revert to generic solver
        if self.lu_current:
            if self.EXACT_CONST_TERM:
                return sp_linalg.lu_solve((self.lu, self.piv), rhs)  # only get gradient (no const term)
            else:
                return sp_linalg.lu_solve((self.lu, self.piv), rhs)[1:]  # only return gradient (1st term is constant)
        else:
            logging.warning("model.solve_LU not using factorisation")
            Wmat = self.build_interp_matrix()
            if self.EXACT_CONST_TERM:
                return np.linalg.solve(Wmat, rhs)  # only get gradient (no const term)
            else:
                return np.linalg.solve(Wmat, rhs)[1:]  # only return gradient (1st term is constant)

    def get_final_results(self):
        # Called when about to exit BOBYQB
        # Return x and fval for optimal point (either from xsave+fsave or kopt)
        if self.fval_opt() <= self.fsave:  # optimal has changed since xsave+fsave were last set
            x = self.x_within_bounds(k=self.kopt)
            rvec = self.fval_v_opt()
            f = self.fval_opt()
            jacmin = self.gqv_at_xopt().T
        else:
            x = self.xsave
            rvec = self.rsave
            f = self.fsave
            jacmin = self.jacsave

        return x, rvec, f, jacmin

    def build_full_model(self):
        # Build full least squares objective model from mini-models
        # Centred around xopt = xpt[kopt, :]
        v_temp = self.fval_v_opt()  # m-vector
        gqv_xopt = self.gqv_at_xopt()  # J^T (transpose of Jacobian) at xopt, rather than xbase

        # Use the gradient at xopt to formulate \sum_i (2*f_i \nabla f_i) = 2 J^t m(x_opt)
        gopt = np.dot(gqv_xopt, v_temp)  # n-vector (gqv = J^T)

        # Gauss-Newton part of Hessian
        hq = to_upper_triangular_vector(np.dot(gqv_xopt, gqv_xopt.T))

        # Apply scaling based on convention for objective - this code uses sumsq(r_i) not 0.5*sumsq(r_i)
        gopt = 2.0 * gopt
        hq = 2.0 * hq

        return gopt, hq

    def build_interp_matrix(self):
        if self.EXACT_CONST_TERM:
            Wmat = np.zeros((self.n, self.n))
            idx_to_use = [k for k in range(self.npt) if k != self.kopt]
            for i in range(self.n):
                Wmat[i,:] = self.xpt[idx_to_use[i], :] - self.xopt()
        else:
            Wmat = np.zeros((self.n + 1, self.n + 1))
            Wmat[:, 0] = 1.0
            Wmat[:, 1:] = self.xpt  # size npt * n
        return Wmat

    def predicted_values(self, d, d_based_at_xopt=True, with_const_term=False):
        if d_based_at_xopt:
            Jd = np.dot(self.gqv.T, d + self.xopt())  # J^T * d (where Jacobian J = self.gqv^T)
        else: # d based at xbase
            Jd = np.dot(self.gqv.T, d)  # J^T * d (where Jacobian J = self.gqv^T)
        return Jd + (self.model_const_term_v if with_const_term else 0.0)

    def square_distances_to_xopt(self):
        sq_distances = np.zeros((self.npt,))
        for k in range(self.npt):
            sq_distances[k] = sumsq(self.xpt[k, :] - self.xopt())
        return sq_distances

    def min_objective_value(self, abs_tol=1.0e-12, rel_tol=1.0e-20):
        # Set a minimum value so that if the full objective falls below it, we immediately finish
        return max(abs_tol, rel_tol*self.fbeg)


def build_initial_set(objfun, x0, xl, xu, rhobeg, maxfun):
    # Evaluate at initial point (also gets us m)
    v_err0, f0 = eval_least_squares_objective(objfun, x0, eval_num=1)

    # Get dimension of problem and number of sample points from x0 and v_err0 information
    n = np.size(x0)
    npt = n + 1
    m = np.size(v_err0)

    # Initialise model (sets x0 as base point and xpt = zeros, so xpt[0,:] = x0)
    model = Model(n, m, npt, x0, xl, xu)

    # Build initial sample set
    at_upper_boundary = (model.su < 0.01 * rhobeg)  # su = xu - x0, should be +ve, actually > rhobeg
    for k in range(n):
        step_size = (rhobeg if not at_upper_boundary[k] else -rhobeg)
        model.xpt[k+1, k] = step_size

    # Add results of objective evaluation at x0
    model.fval_v[0, :] = v_err0
    model.fval[0] = f0
    model.kopt = 0
    model.fbeg = f0
    model.xsave = x0.copy()
    model.rsave = v_err0.copy()
    model.fsave = f0
    model.jacmin = np.zeros((m, n))

    # Evaluate objective at each point in the initial sample set
    for nf in range(1, min(npt, maxfun)):
        x = model.x_within_bounds(k=nf)
        v_err, f = eval_least_squares_objective(objfun, x, eval_num=nf+1)  # nf is one behind because of f(x0)

        model.fval[nf] = f
        model.fval_v[nf, :] = v_err

        if f < model.fval_opt():  # update optimal point
            model.kopt = nf

    return model


def altmov_wrapper(model, knew, adelt):
    model.factorise_LU()
    # First need to get knew-th column of H matrix
    if model.EXACT_CONST_TERM:
        if knew == model.kopt:
            ek = -np.ones((model.n, ))  # matrix based on (y-xk), so different geom structure for kopt
        else:
            ek = np.zeros((model.n, ))
            if knew < model.kopt:
                ek[knew] = 1.0
            else:
                ek[knew - 1] = 1.0
        H_knew = model.solve_LU(ek)
    else:
        ek = np.zeros((model.n + 1,))
        ek[knew] = 1.0
        H_knew = model.solve_LU(ek)
    xnew, xalt, cauchy, abs_denom = altmov(model.xpt, model.sl, model.su, model.kopt,
                                           model.xopt(), knew, adelt, H_knew)
    # abs_denom is Lagrange_knew evaluated at xnew
    return xnew, xalt, cauchy, abs_denom


def altmov_wrapper_v2(model, knew, adelt):
    model.factorise_LU()
    # First need to get knew-th column of H matrix
    if model.EXACT_CONST_TERM:
        if knew == model.kopt:
            ek = -np.ones((model.n, ))  # matrix based on (y-xk), so different geom structure for kopt
        else:
            ek = np.zeros((model.n, ))
            if knew < model.kopt:
                ek[knew] = 1.0
            else:
                ek[knew - 1] = 1.0
        g = model.solve_LU(ek)  # H_knew
    else:
        ek = np.zeros((model.n + 1,))
        ek[knew] = 1.0
        g = model.solve_LU(ek)  # H_knew

    c = 1 if knew == model.kopt else 0  # c, g are for knew-th Lagrange polynomial, based at xopt (c + g*(x-xopt))
    xnew = max_step_in_box_and_ball(model.xopt(), c, g, model.sl, model.su, adelt)
    return xnew


def choose_knew(model, delta, xnew, skip_kopt=True):
    # in model, uses: n, npt, xpt, kopt/xopt, build_interp_matrix()
    # model unchanged by this method

    # Criteria is to maximise: max(1, ||yt-xk||^4/Delta^4) * abs(Lagrange_t(xnew))
    # skip_kopt determines whether to check t=kopt as a possible candidate or not

    model.factorise_LU()  # Prep for linear solves

    delsq = delta ** 2
    scaden = -1.0
    knew = None  # may knew never be set here?

    try:
        for k in range(model.npt):
            if skip_kopt and k == model.kopt:
                continue  # next k in this inner loop
            if model.EXACT_CONST_TERM:
                if k == model.kopt:
                    ek = -np.ones((model.n,))  # matrix based on (y-xk), so different geom structure for kopt
                else:
                    ek = np.zeros((model.n, ))
                    if k < model.kopt:
                        ek[k] = 1.0
                    else:
                        ek[k-1] = 1.0
                Hk = model.solve_LU(ek)
            else:
                ek = np.zeros((model.n + 1,))
                ek[k] = 1.0
                Hk = model.solve_LU(ek)  # k-th column of H, except 1st entry (i.e. Lagrange polynomial gradient)
            lagrange_k_at_d = 1.0 + np.dot(xnew-model.xpt[k, :], Hk)
            distsq = sumsq(model.xpt[k, :] - model.xopt())
            temp = max(1.0, (distsq / delsq) ** 2)
            if temp * abs(lagrange_k_at_d) > scaden:
                scaden = temp * abs(lagrange_k_at_d)
                knew = k

        linalg_error = False
    except np.linalg.LinAlgError:
        linalg_error = True

    return knew, linalg_error


def trust_region_subproblem_least_squares(model, delta):
    # in model, uses: n, npt, xpt, kopt/xopt, sl, su, build_full_model()
    # model unchanged by this method

    # Build model for full least squares objectives
    gopt, hq = model.build_full_model()
    # Call original BOBYQA trsbox function
    d, gnew, crvmin = trsbox(model.xopt(), gopt, hq, model.sl, model.su, delta)
    return d, gopt, hq, gnew, crvmin


def done_with_current_rho(model, nf, nfsav, rho, diffs, xnew, gnew, hq, crvmin):
    # in model, uses: n, sl, su
    # model unchanged by this method

    if nf <= nfsav + 2:
        return False

    errbig = max(diffs)
    frhosq = 0.125 * rho ** 2
    if crvmin > 0.0 and errbig > frhosq * crvmin:
        return False

    bdtol = errbig / rho
    for j in range(model.n):
        bdtest = bdtol
        if xnew[j] == model.sl[j]:
            bdtest = gnew[j]
        if xnew[j] == model.su[j]:
            bdtest = -gnew[j]
        if bdtest < bdtol:
            curv = get_hessian_element(model.n, hq, j, j)  # curv = Hessian(j, j)
            bdtest += 0.5 * curv * rho
            if bdtest < bdtol:
                return False

    return True


def reduce_rho(old_rho, rhoend):
    ratio = old_rho/rhoend
    if ratio <= 16.0:
        new_rho = rhoend
    elif ratio <= 250.0:
        new_rho = sqrt(ratio)*rhoend
    else:
        new_rho = 0.1*old_rho
    delta = max(0.5*old_rho, new_rho)
    return delta, new_rho


def check_and_fix_geometry(model, objfun, distsq, delta, rho, dnorm, diffs, nf, nfsav, maxfun, rounding_error_const,
                           update_delta=True):
    # [Fortran label 650]
    # If any xpt more than distsq away from xopt, fix geometry
    knew_tmp, distsq_tmp = get_vector_max(all_square_distances(model.xpt, model.xopt()))
    if distsq_tmp > distsq:  # fix geometry and quit
        knew = knew_tmp
        distsq = distsq_tmp

        dist = sqrt(distsq)
        if update_delta:  # optional
            delta = max(min(0.1 * delta, 0.5 * dist), 1.5 * rho)  # use 0.5*dist, within range [0.1*delta, 1.5*rho]

        adelt = max(min(0.1 * dist, delta), rho)
        if adelt ** 2 <= rounding_error_const * sumsq(model.xopt()):
            model.shift_base(model.xopt())

        model, nf, nfsav, diffs, return_to_new_tr_iteration, exit_flag, exit_str \
            = fix_geometry(model, objfun, knew, adelt, rho, dnorm, diffs, nf, nfsav, maxfun)

        return model, delta, nf, nfsav, diffs, return_to_new_tr_iteration, exit_flag, exit_str
    else:
        # Do nothing, just quit
        # return_to_new_tr_iteration = None when didn't fix geometry
        return model, delta, nf, nfsav, diffs, None, None, None


def fix_geometry(model, objfun, knew, adelt, rho, dnorm, diffs, nf, nfsav, maxfun):
    # in model, uses: n, npt, xpt, sl, su, kopt/xopt, build_interp_metrix, and others
    # model is changed by this function: gqv from interp_mini_models, and others

    USE_OLD_ALTMOV = False
    try:
        if USE_OLD_ALTMOV:
            xnew, xalt, cauchy, denom = altmov_wrapper(model, knew, adelt)
        else:
            xnew = altmov_wrapper_v2(model, knew, adelt)
            xalt = None
            cauchy = None
            denom = None
    except np.linalg.LinAlgError:
        exit_flag = EXIT_LINALG_ERROR
        exit_str = "Singular matrix encountered in ALTMOV"
        return_to_new_tr_iteration = False  # return and quit
        return model, nf, nfsav, diffs, return_to_new_tr_iteration, exit_flag, exit_str

    if xnew is None:  # issue with stpsav occurred, quit DFOGN
        exit_flag = EXIT_ALTMOV_MEMORY_ERROR
        exit_str = "Error in ALTMOV - stpsav undefined"
        return_to_new_tr_iteration = False  # return and quit
        return model, nf, nfsav, diffs, return_to_new_tr_iteration, exit_flag, exit_str

    if USE_OLD_ALTMOV and denom < cauchy and cauchy > 0.0:
        xnew = xalt.copy()

    d = xnew - model.xopt()

    # [Fortran label 360]
    x = model.x_within_bounds(x=xnew)
    if nf >= maxfun:
        exit_flag = EXIT_MAXFUN_WARNING
        exit_str = "Objective has been called MAXFUN times"
        return_to_new_tr_iteration = False  # return and quit
        return model, nf, nfsav, diffs, return_to_new_tr_iteration, exit_flag, exit_str

    nf += 1
    v_err, f = eval_least_squares_objective(objfun, x, eval_num=nf)

    if f <= model.min_objective_value():
        # Force model.get_final_results() to return this new point if it's better than xopt, then quit
        model.xsave = x
        model.rsave = v_err.copy()
        model.fsave = f
        model.jacsave = model.gqv_at_xopt().T
        exit_flag = EXIT_SUCCESS
        exit_str = "Sufficient reduction in objective value"
        return_to_new_tr_iteration = False  # return and quit
        return model, nf, nfsav, diffs, return_to_new_tr_iteration, exit_flag, exit_str

    # Use the quadratic model to predict the change in F due to the step D,
    # and set DIFF to the error of this prediction.
    gopt, hq = model.build_full_model()
    if gopt is None:  # Use this to indicate linalg error
        if f < model.fval_opt():
            # Force model.get_final_results() to return this new point if it's better than xopt, then quit
            model.xsave = x
            model.rsave = v_err.copy()
            model.fsave = f
            model.jacsave = model.gqv_at_xopt().T
        exit_flag = EXIT_LINALG_ERROR
        exit_str = "Singular matrix encountered in FIX_GEOMETRY (full model interpolation step)"
        return_to_new_tr_iteration = False  # return and quit
        return model, nf, nfsav, diffs, return_to_new_tr_iteration, exit_flag, exit_str

    pred_reduction = - calculate_model_value(gopt, hq, d)
    actual_reduction = model.fval_opt() - f
    diffs = [abs(pred_reduction - actual_reduction), diffs[0], diffs[1]]

    if dnorm > rho:
        nfsav = nf

    # Update bmat, zmat, gopt, etc. (up to label ~560)
    model.update_point(knew, xnew, v_err, f)

    exit_flag = None
    exit_str = None
    return_to_new_tr_iteration = True  # return and start new trust region iteration (label 60)
    return model, nf, nfsav, diffs, return_to_new_tr_iteration, exit_flag, exit_str


def dfogn_main(objfun, x0, xl, xu, rhobeg, rhoend, maxfun):
    exit_flag = None
    exit_str = None

    # One variable in BOBYQB depends on which code form we are using
    if zhang_code_structure:
        rounding_error_const = 0.1  # Zhang code
    else:
        rounding_error_const = 1.0e-3  # BOBYQA

    ###########################################################
    # Set up initial interpolation set
    ###########################################################
    model = build_initial_set(objfun, x0, xl, xu, rhobeg, maxfun)

    if maxfun < model.npt:
        exit_flag = EXIT_MAXFUN_WARNING
        exit_str = "Objective has been called MAXFUN times"
        x, rvec, f, jacmin = model.get_final_results()
        return x, rvec, f, jacmin, maxfun, exit_flag, exit_str
        # return x, f, maxfun, exit_flag, exit_str

    ###########################################################
    # Set other variables before begin iterations
    ###########################################################
    finished_main_loop = False

    (rho, delta) = (rhobeg, rhobeg)
    nf = min(maxfun, model.npt)  # number of function evaluations so far
    nfsav = nf
    diffs = [0.0, 0.0, 0.0]  # (diffa, diffb, diffc) in Fortran code, used in done_with_current_rho()

    ###########################################################
    # Start of main loop [Fortran label 60]
    ###########################################################
    while not finished_main_loop:
        # Interpolate each mini-model
        interp_ok = model.interpolate_mini_models()
        if not interp_ok:
            exit_flag = EXIT_LINALG_ERROR
            exit_str = "Singular matrix in mini-model interpolation (main loop)"
            finished_main_loop = True
            break  # quit

        # Solve trust region subproblem to get tentative step d
        # Model for full least squares objective is given by (gopt, hq)
        d, gopt, hq, gnew, crvmin = trust_region_subproblem_least_squares(model, delta)
        logging.debug("Trust region step is d = " + str(d))
        xnew = model.xopt() + d
        dsq = sumsq(d)
        dnorm = min(delta, sqrt(dsq))

        if dnorm < 0.5 * rho:
            ###################
            # Start failed TR step
            ###################
            logging.debug("Failed trust region step")

            if not done_with_current_rho(model, nf, nfsav, rho, diffs, xnew, gnew, hq, crvmin):
                # [Fortran label 650]
                distsq = (10.0 * rho) ** 2
                model, delta, nf, nfsav, diffs, return_to_new_tr_iteration, geom_exit_flag, geom_exit_str = \
                    check_and_fix_geometry(model, objfun, distsq, delta, rho, dnorm, diffs, nf, nfsav, maxfun,
                                           rounding_error_const, update_delta=True)

                if return_to_new_tr_iteration is not None:  # i.e. if we did actually fix geometry
                    if return_to_new_tr_iteration:
                        finished_main_loop = False
                        continue  # next trust region step
                    else:  # quit
                        exit_flag = geom_exit_flag
                        exit_str = geom_exit_str
                        finished_main_loop = True
                        break  # quit
                # If we didn't fix geometry, reduce rho as below
            # otherwise, if we are done with current rho, reduce rho as below

            # Reduce rho and continue [Fortran label 680]
            if rho > rhoend:
                delta, rho = reduce_rho(rho, rhoend)
                logging.info("New rho = %g after %i function evaluations" % (rho, nf))
                logging.debug("Best so far: f = %.15g at x = " % (model.fval_opt()) + str(model.xbase + model.xopt()))
                nfsav = nf
                finished_main_loop = False
                continue  # next trust region step
            else:
                # Cannot reduce rho, so check xnew and quit
                x = model.x_within_bounds(x=xnew)
                if nf >= maxfun:  # quit
                    exit_flag = EXIT_MAXFUN_WARNING
                    exit_str = "Objective has been called MAXFUN times"
                    finished_main_loop = True
                    break  # quit

                nf += 1
                v_err, f = eval_least_squares_objective(objfun, x, eval_num=nf)  # v_err not used here

                # Force model.get_final_results() to return this new point if it's better than xopt, then quit
                model.xsave = x
                model.rsave = v_err.copy()
                model.fsave = f
                model.jacsave = model.gqv_at_xopt().T
                exit_flag = EXIT_SUCCESS
                exit_str = "rho has reached rhoend"
                finished_main_loop = True
                break  # quit
            ###################
            # End failed TR step
            ###################
        else:
            ###################
            # Start successful TR step
            ###################
            logging.debug("Successful trust region step")

            # Severe cancellation is likely to occur if XOPT is too far from XBASE. [Fortran label 90]
            if dsq <= rounding_error_const * sumsq(model.xopt()):
                base_shift = model.xopt()
                xnew = xnew - base_shift  # before xopt is updated
                model.shift_base(base_shift)  # includes a re-factorisation of the interpolation matrix

            # Set KNEW to the index of the next interpolation point to be deleted to make room for a trust
            # region step. Again RESCUE may be called if rounding errors have damaged
            # the chosen denominator, which is the reason for attempting to select
            # KNEW before calculating the next value of the objective function.
            knew, linalg_error = choose_knew(model, delta, xnew, skip_kopt=True)

            if linalg_error:
                exit_flag = EXIT_LINALG_ERROR
                exit_str = "Singular matrix when finding knew (in main loop)"
                finished_main_loop = True
                break  # quit

            # Calculate the value of the objective function at XBASE+XNEW, unless
            # the limit on the number of calculations of F has been reached.
            # [Fortran label 360, with ntrits > 0]
            x = model.x_within_bounds(x=xnew)

            if nf >= maxfun:
                exit_flag = EXIT_MAXFUN_WARNING
                exit_str = "Objective has been called MAXFUN times"
                finished_main_loop = True
                break  # quit

            nf += 1
            v_err, f = eval_least_squares_objective(objfun, x, eval_num=nf)

            if f <= model.min_objective_value():
                # Force model.get_final_results() to return this new point if it's better than xopt, then quit
                model.xsave = x
                model.rsave = v_err.copy()
                model.fsave = f
                model.jacsave = model.gqv_at_xopt().T
                exit_flag = EXIT_SUCCESS
                exit_str = "Objective is sufficiently small"
                finished_main_loop = True
                break  # quit

            # Use the quadratic model to predict the change in F due to the step D,
            # and set DIFF to the error of this prediction.
            pred_reduction = - calculate_model_value(gopt, hq, d)
            actual_reduction = model.fval_opt() - f
            diffs = [abs(pred_reduction - actual_reduction), diffs[0], diffs[1]]

            if dnorm > rho:
                nfsav = nf

            if pred_reduction < 0.0:
                exit_flag = EXIT_TR_INCREASE_ERROR
                exit_str = "Trust region step gave model increase"
                finished_main_loop = True
                break  # quit

            # Pick the next value of DELTA after a trust region step.
            # Update trust region radius
            ratio = actual_reduction / pred_reduction
            if ratio <= 0.1:
                delta = min(0.5 * delta, dnorm)
            elif ratio <= 0.7:
                delta = max(0.5 * delta, dnorm)
            else:  # (ratio > 0.7) Different updates depending on which code version we're using
                if zhang_code_structure:
                    delta = min(max(2.0 * delta, 4.0 * dnorm), 1.0e10)  # DFBOLS code version
                elif bbqtr:
                    delta = max(0.5 * delta, 2.0 * dnorm)  # BOBYQA version
                else:
                    delta = max(delta, 2.0 * dnorm)  # Zhang paper version
            if delta <= 1.5 * rho:  # cap trust region radius at rho
                delta = rho
            logging.debug("New delta = %g (rho = %g) from ratio %g" % (delta, rho, ratio))

            # Recalculate KNEW and DENOM if the new F is less than FOPT.
            if actual_reduction > 0.0:  # f < model.fval_opt()
                knew, linalg_error = choose_knew(model, delta, xnew, skip_kopt=False)

                if linalg_error:
                    exit_flag = EXIT_LINALG_ERROR
                    exit_str = "Singular matrix when finding knew (in main loop, second time)"
                    finished_main_loop = True
                    break  # quit

            # Updating...
            logging.debug("Updating with knew = %i" % knew)
            model.update_point(knew, xnew, v_err, f)

            # If a trust region step has provided a sufficient decrease in F, then
            # branch for another trust region calculation.
            if ratio >= 0.1:
                finished_main_loop = False
                continue  # next trust region step

            # Alternatively, find out if the interpolation points are close enough
            # to the best point so far.
            # [Fortran label 650]
            distsq = max((2.0 * delta) ** 2, (10.0 * rho) ** 2)
            model, delta, nf, nfsav, diffs, return_to_new_tr_iteration, geom_exit_flag, geom_exit_str = \
                check_and_fix_geometry(model, objfun, distsq, delta, rho, dnorm, diffs, nf, nfsav, maxfun,
                                       rounding_error_const, update_delta=False)  # don't update delta when ntrits > 0

            if return_to_new_tr_iteration is not None:  # i.e. if we did actually fix geometry
                if return_to_new_tr_iteration:
                    finished_main_loop = False
                    continue  # next trust region step
                else:  # quit
                    exit_flag = geom_exit_flag
                    exit_str = geom_exit_str
                    finished_main_loop = True
                    break  # quit
            # If we didn't fix geometry, reduce rho [Fortran label 680]

            if ratio > 0.0:
                finished_main_loop = False
                continue  # next trust region step

            if max(delta, dnorm) > rho:
                finished_main_loop = False
                continue  # next trust region step

            # Reduce rho and continue [Fortran label 680]
            if rho > rhoend:
                delta, rho = reduce_rho(rho, rhoend)
                logging.info("New rho = %g after %i function evaluations" % (rho, nf))
                logging.debug("Best so far: f = %.15g at x = " % (model.fval_opt()) + str(model.xbase + model.xopt()))
                nfsav = nf
                finished_main_loop = False
                continue  # next trust region step
            else:
                # Cannot reduce rho further
                exit_flag = EXIT_SUCCESS
                exit_str = "rho has reached rhoend"
                finished_main_loop = True
                break  # quit
            ###################
            # End successful TR step
            ###################
        #############################
        # End this iteration of main loop - take next TR step
        #############################
    ###########################################################
    # End of main loop [Fortran label 720]
    ###########################################################

    x, rvec, f, jacmin = model.get_final_results()
    logging.debug("At return from DFOGN, number of function evals = %i" % nf)
    logging.debug("Smallest objective value = %.15g at x = " % f + str(x))
    return x, rvec, f, jacmin, nf, exit_flag, exit_str
    # return x, f, nf, exit_flag, exit_str


def solve(objfun, x0, lower=None, upper=None, maxfun=1000, rhobeg=None, rhoend=1e-8):
    # If bounds not provided, set to something large
    xl = (lower if lower is not None else -1.0e20 * np.ones(x0.shape))
    xu = (upper if upper is not None else 1.0e20 * np.ones(x0.shape))

    # Set default value of rhobeg to something sensible
    rhobeg = (rhobeg if rhobeg is not None else 0.1 * max(np.max(np.abs(x0)), 1.0))

    n = np.size(x0)

    # Input & parameter checks
    input_error_msg = None
    if rhobeg < 0.0:
        input_error_msg = "Input error: rhobeg must be strictly positive"

    if rhoend < 0.0:
        input_error_msg = "Input error: rhoend must be strictly positive"

    if rhobeg <= rhoend:
        input_error_msg = "Input error: rhobeg must be > rhoend"

    if maxfun <= 0:
        input_error_msg = "Input error: maxfun must be strictly positive"

    if np.shape(x0) != (n,):
        input_error_msg = "Input error: x0 must be a vector"

    if np.shape(x0) != np.shape(xl):
        input_error_msg = "Input error: lower bounds must have same shape as x0"

    if np.shape(x0) != np.shape(xu):
        input_error_msg = "Input error: upper bounds must have same shape as x0"

    if np.min(xu - xl) < 2.0 * rhobeg:
        input_error_msg = "Input error: gap between lower and upper must be at least 2*rhobeg"

    # Process input errors
    if input_error_msg is not None:
        results = OptimResults(x0, None, None, None, 0, EXIT_INPUT_ERROR, "Input error: " + input_error_msg)
        return results

    if maxfun <= n + 1:
        warnings.warn("maxfun <= npt: Are you sure your budget is large enough?", RuntimeWarning)

    # Enforce lower bounds on x0 (ideally with gap of at least rhobeg)
    idx = (xl < x0) & (x0 <= xl+rhobeg)
    if np.any(idx):
        warnings.warn("Some entries of x0 too close to lower bound, adjusting", RuntimeWarning)
    x0[idx] = xl[idx] + rhobeg

    idx = (x0 <= xl)
    if np.any(idx):
        warnings.warn("Some entries of x0 below lower bound, adjusting", RuntimeWarning)
    x0[idx] = xl[idx]

    # Enforce upper bounds on x0 (ideally with gap of at least rhobeg)
    idx = (xu-rhobeg <= x0) & (x0 < xu)
    if np.any(idx):
        warnings.warn("Some entries of x0 too close to upper bound, adjusting", RuntimeWarning)
    x0[idx] = xu[idx] - rhobeg

    idx = (x0 >= xu)
    if np.any(idx):
        warnings.warn("Some entries of x0 above upper bound, adjusting", RuntimeWarning)
    x0[idx] = xu[idx]

    x, rvec, f, jacmin, nf, exit_flag, exit_str = dfogn_main(objfun, x0.copy(), xl, xu, rhobeg, rhoend, maxfun)

    # Clean up exit_str to have better information:
    if exit_flag == EXIT_SUCCESS:
        exit_str = "Success: " + exit_str
    elif exit_flag == EXIT_MAXFUN_WARNING:
        exit_str = "Warning: " + exit_str
    elif exit_flag == EXIT_INPUT_ERROR:
        exit_str = "Input error: " + exit_str
    elif exit_flag == EXIT_TR_INCREASE_ERROR:
        exit_str = "Trust region subproblem error: " + exit_str
    elif exit_flag == EXIT_LINALG_ERROR:
        exit_str = "Linear algebra error: " + exit_str
    elif exit_flag == EXIT_ALTMOV_MEMORY_ERROR:
        exit_str = "ALTMOV memory error: " + exit_str
    else:
        exit_str = "Unknown exit flag " + str(exit_flag) + " with message " + exit_str

    # Build solution object
    results = OptimResults(x, rvec, f, jacmin, nf, exit_flag, exit_str)
    # return x, f, nf, exit_flag, exit_str
    return results

