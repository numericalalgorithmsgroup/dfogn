"""
DFO-GN
====================
A derivative-free solver for least squares minimisation with bound constraints.
This version has resampling (not part of main package).

This file is a modified version of DFOGN which allows resampling and restarts,
to better cope with noisy problems.

Lindon Roberts, 2017

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

import numpy as np
import scipy.linalg as sp_linalg
from math import sqrt
import logging
from .util import *
from .trust_region import *
from .alternative_move import *

#######################
# Exit codes
EXIT_SUCCESS = 0  # successful finish (rho=rhoend or sufficient objective reduction)
EXIT_INPUT_ERROR = 1  # error, bad inputs
EXIT_MAXFUN_WARNING = 2  # warning, reached max function evals
EXIT_TR_INCREASE_ERROR = 3  # error, trust region step increased model value
EXIT_LINALG_ERROR = 4  # error, linalg error (singular matrix encountered)
EXIT_ALTMOV_MEMORY_ERROR = 5  # error, stpsav issue in ALTMOV

# Errors for which we can do a restart (not including rho=rhoend in EXIT_SUCCESS)
DO_RESTART_ERRORS = [EXIT_TR_INCREASE_ERROR, EXIT_LINALG_ERROR, EXIT_ALTMOV_MEMORY_ERROR]
#######################

#######################
# Sampling scenarios
SCEN_PRELIM = 1  # during prelim
SCEN_GROWING_NEW_DIRECTION = 2  # adding new direction while growing
SCEN_TR_STEP = 3  # sampling xk+sk from successful trust region step
SCEN_GEOM_UPDATE = 4  # calling altmov for geometry fixing
SCEN_RHOEND_REACHED = 5  # reached rhoend in unsuccessful TR step
#######################


class Model:
    def __init__(self, n, m, npt, x0, xl, xu):
        assert npt==n+1, "Require strictly linear model"
        # Problem sizes
        self.n = n
        self.m = m
        self.npt = npt
        self.npt_so_far = 0  # how many points have we added so far (for growing initial set)

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
        self.fsave = None  # sum of squares for final return value

        self.lu = None  # LU decomp of interp matrix
        self.piv = None  # pivots for LU decomposition of interp matrix
        self.lu_current = False  # whether current LU factorisation of interp matrix is up-to-date or not

        self.EXACT_CONST_TERM = True  # use exact c=r(xopt) for interpolation (improve conditioning)
        # Affects mini-model interpolation / interpolation matrix, but also geometry updating

        self.nsamples = np.zeros((npt,), dtype=int)  # how many samples we have averaged to get fval_v, where fval = sumsq(avg fval_v)

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
        if knew >= self.npt_so_far and self.npt_so_far < self.npt:
            # when still growing, need to append in correct order
            assert knew == self.npt_so_far, "Updating new index too far along (%g when should be %g)" % (knew, self.npt_so_far)
            self.npt_so_far += 1

        # Add point xnew with objective vector v_err (full objective f) at the knew-th index
        self.xpt[knew,:] = xnew
        self.fval_v[knew, :] = v_err
        self.fval[knew] = f
        self.nsamples[knew] = 1

        # Update XOPT, GOPT and KOPT if the new calculated F is less than FOPT.
        if f < self.fval_opt():
            self.kopt = knew

        self.lu_current = False
        return

    def add_point_resample(self, knew, v_err_new):
        # We have resampled point knew and got a new fval_v = v_err_new
        # Update our estimates of fval_v
        assert knew < self.npt_to_use(), "Invalid knew"
        t = float(self.nsamples[knew]) / float(self.nsamples[knew] + 1)
        self.fval_v[knew, :] = t * self.fval_v[knew, :] + (1 - t) * v_err_new
        self.fval[knew] = sumsq(self.fval_v[knew, :])
        self.nsamples[knew] += 1

        if self.fval[knew] < self.fval_opt():
            self.kopt = knew

        return

    def npt_to_use(self):
        # Number of points to use when building interpolation system
        return min(self.npt_so_far, self.npt)  # depends on whether we have a full set yet (or not)

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
        self.factorise_LU()
        try:
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
        if self.fsave is None or self.fval_opt() <= self.fsave:  # optimal has changed since xsave+fsave were last set
            x = self.x_within_bounds(k=self.kopt)
            f = self.fval_opt()
        else:
            x = self.xsave
            f = self.fsave

        return x, f

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
        if self.fbeg is not None:
            return max(abs_tol, rel_tol * self.fbeg)
        else:
            return abs_tol


def sample_objective(m, objfun, x, nf, nx, maxfun, min_obj_value, nsamples=1):
    # Sample from objective function several times, keeping track of maxfun and min_obj_value throughout
    if m is None:
        # Don't initialise v_err_list yet
        v_err_list = None
    else:
        v_err_list = np.zeros((nsamples, m))
    f_list = np.zeros((nsamples,))
    exit_flag = None
    exit_str = None
    nsamples_run = 0

    for i in range(nsamples):
        if nf >= maxfun:
            exit_flag = EXIT_MAXFUN_WARNING
            exit_str = "Objective has been called MAXFUN times"
            break  # quit

        nf += 1
        this_v_err, f_list[i] = eval_least_squares_objective_v2(objfun, x, eval_num=nf, pt_num=nx+1, full_x_thresh=6)
        if m is None:
            m = len(this_v_err)
            v_err_list = np.zeros((nsamples, m))
        v_err_list[i, :] = this_v_err

        nsamples_run += 1

        if f_list[i] <= min_obj_value:
            # Force model.get_final_results() to return this new point if it's better than xopt, then quit
            exit_flag = EXIT_SUCCESS
            exit_str = "Objective is sufficiently small"
            break  # quit

    return v_err_list, f_list, nf, nx+1, nsamples_run, exit_flag, exit_str


def build_initial_set(objfun, x0, xl, xu, rhobeg, maxfun, nsamples, nf_so_far, nx_so_far, ndirs_initial, nruns_so_far,
                      m=None, random_initial_directions=False):
    n = np.size(x0)
    npt = n + 1
    if m is not None:
        # Initialise model (sets x0 as base point and xpt = zeros, so xpt[0,:] = x0)
        model = Model(n, m, npt, x0, xl, xu)
        model.kopt = 0
        minval = model.min_objective_value()
    else:
        # If we don't yet have m, wait until we have done a function evaluation before initialising model
        model = None
        minval = -1.0

    assert 1 <= ndirs_initial < np.size(x0)+1, "build_inital_set: must have 1 <= ndirs_initial < n+1"
    nx = nx_so_far
    nf = nf_so_far

    # For calling nsamples:
    delta = rhobeg
    rho = rhobeg
    current_iter = 0

    # Evaluate at initial point (also gets us m in the first run through)
    nsamples_to_use = nsamples(delta, rho, current_iter, nruns_so_far, SCEN_PRELIM)
    v_err_list, f_list, nf, nx, nsamples_run, exit_flag, exit_str = sample_objective(m, objfun, x0,
                                                                                     nf, nx, maxfun,
                                                                                     minval,
                                                                                     nsamples=nsamples_to_use)

    # If we have just learned m, initialise model (sets x0 as base point and xpt = zeros, so xpt[0,:] = x0)
    if model is None:
        # Now we know m = v_err_list.shape[1]
        model = Model(n, v_err_list.shape[1], npt, x0, xl, xu)
        model.kopt = 0

    f0 = sumsq(np.mean(v_err_list[:nsamples_run, :], axis=0))  # estimate actual objective value
    # Handle exit conditions (f < min obj value or maxfun reached)
    if exit_flag is not None:  # then exit_str is also set
        if nsamples_run > 0:
            fmin = np.min(f_list[:nsamples_run])
            if model.fsave is None or fmin < model.fsave:
                model.xsave = x0
                model.fsave = fmin
        return_to_new_tr_iteration = False  # return and quit
        return model, nf, nx, return_to_new_tr_iteration, exit_flag, exit_str

    # Otherwise, add new results (increments model.npt_so_far)
    model.update_point(0, model.xpt[0, :], v_err_list[0, :], f_list[0])
    for i in range(1, nsamples_run):
        model.add_point_resample(0, v_err_list[i, :])  # add new info

    # Add results of objective evaluation at x0
    model.fbeg = f0
    model.xsave = x0.copy()
    model.fsave = f0

    # Build initial sample set either using random orthogonal directions, or coordinate directions
    if random_initial_directions:
        # Get ndirs_initial random orthogonal directions
        A = np.random.randn(n, ndirs_initial)  # Standard Gaussian n*ndirs_initial
        Q = np.linalg.qr(A)[0]  # Q is n*ndirs_initial with orthonormal columns

        # Now add the random directions
        for ndirns in range(ndirs_initial):
            dirn = Q[:, ndirns]
            # Scale direction to ensure the new point lies within initial trust region, satisfies constraints
            scale_factor = rhobeg / np.linalg.norm(dirn)
            for j in range(n):
                if dirn[j] < 0.0:
                    scale_factor = min(scale_factor, model.sl[j] / dirn[j])
                elif dirn[j] > 0.0:
                    scale_factor = min(scale_factor, model.su[j] / dirn[j])
            model.xpt[1 + ndirns, :] = scale_factor * dirn

    else:
        at_upper_boundary = (model.su < 0.01 * rhobeg)  # su = xu - x0, should be +ve, actually > rhobeg
        for k in range(ndirs_initial):
            step_size = (rhobeg if not at_upper_boundary[k] else -rhobeg)
            model.xpt[k+1, k] = step_size

    # Evaluate objective at each point in the initial sample set
    for k in range(1, ndirs_initial):
        x = model.x_within_bounds(k=k)
        nsamples_to_use = nsamples(delta, rho, current_iter, nruns_so_far, SCEN_PRELIM)
        v_err_list, f_list, nf, nx, nsamples_run, exit_flag, exit_str = sample_objective(model.m, objfun, x,
                                                                                         nf, nx, maxfun,
                                                                                         model.min_objective_value(),
                                                                                         nsamples=nsamples_to_use)

        # f = sumsq(np.mean(v_err_list[:nsamples_run, :], axis=0))  # estimate actual objective value
        # Handle exit conditions (f < min obj value or maxfun reached)
        if exit_flag is not None:  # then exit_str is also set
            if nsamples_run > 0:
                fmin = np.min(f_list[:nsamples_run])
                if model.fsave is None or fmin < model.fsave:
                    model.xsave = x
                    model.fsave = fmin
            return_to_new_tr_iteration = False  # return and quit
            return model, nf, nx, return_to_new_tr_iteration, exit_flag, exit_str

        # Otherwise, add new results (increments model.npt_so_far)
        model.update_point(k, model.xpt[k, :], v_err_list[0, :], f_list[0])
        for i in range(1, nsamples_run):
            model.add_point_resample(k, v_err_list[i, :])  # add new info

    return_to_new_tr_iteration = True  # return and continue
    exit_flag = None
    exit_str = None
    return model, nf, nx, return_to_new_tr_iteration, exit_flag, exit_str


def get_new_orthogonal_directions(model, adelt, num_steps=1):
    # Step from xopt along a random direction orthogonal to other yt (or multiple mutually orthogonal steps)
    for i in range(20):  # allow several tries, in case we choosing a point in the subspace of (yt-xk) [very unlucky]
        A = np.random.randn(model.n, num_steps)
        # (modified) Gram-Schmidt to orthogonalise
        for k in range(min(model.npt_so_far, model.npt)):
            if k == model.kopt:
                continue
            yk = model.xpt[k,:] - model.xopt()
            for j in range(num_steps):
                A[:,j] = A[:,j] - (np.dot(A[:,j], yk) / np.dot(yk, yk)) * yk
        # continue if every column sufficiently large
        all_cols_ok = True
        for j in range(num_steps):
            if np.linalg.norm(A[:,j]) < 1e-8:
                all_cols_ok = False
        if all_cols_ok:
            break
    # Scale appropriately so within bounds and ||d|| <= adelt
    Q = np.linalg.qr(A)[0]  # Q is n*ndirs with orthonormal columns
    for j in range(num_steps):
        scale_factor = adelt / np.linalg.norm(Q[:,j])
        for i in range(model.n):
            if Q[i,j] < 0.0:
                scale_factor = min(scale_factor, (model.sl[i] - model.xopt()[i]) / Q[i,j])
            elif Q[i,j] > 0.0:
                scale_factor = min(scale_factor, (model.su[i] - model.xopt()[i]) / Q[i,j])
        Q[:,j] = Q[:,j] * scale_factor
    # Finished!
    return Q


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


def done_with_current_rho(model, current_iter, last_successful_iter, rho, diffs, xnew, gnew, hq, crvmin):
    # in model, uses: n, sl, su
    # model unchanged by this method

    if current_iter <= last_successful_iter + 2:
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


def check_and_fix_geometry(model, objfun, distsq, delta, rho, dnorm, diffs, nf, nx, current_iter, last_successful_iter,
                           maxfun, nsamples, rounding_error_const, nruns_so_far, update_delta=True):
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

        model, nf, nx, last_successful_iter, diffs, return_to_new_tr_iteration, exit_flag, exit_str \
            = fix_geometry(model, objfun, knew, delta, adelt, rho, dnorm, diffs, nf, nx, current_iter,
                           last_successful_iter, maxfun, nsamples, nruns_so_far)

        return model, delta, nf, nx, last_successful_iter, diffs, return_to_new_tr_iteration, exit_flag, exit_str
    else:
        # Do nothing, just quit
        # return_to_new_tr_iteration = None when didn't fix geometry
        return model, delta, nf, nx, last_successful_iter, diffs, None, None, None


def fix_geometry(model, objfun, knew, delta, adelt, rho, dnorm, diffs, nf, nx, current_iter, last_successful_iter,
                 maxfun, nsamples, nruns_so_far):
    # in model, uses: n, npt, xpt, sl, su, kopt/xopt, build_interp_metrix, and others
    # model is changed by this function: gqv from interp_mini_models, and others

    try:
        xnew, xalt, cauchy, denom = altmov_wrapper(model, knew, adelt)
    except np.linalg.LinAlgError:
        exit_flag = EXIT_LINALG_ERROR
        exit_str = "Singular matrix encountered in ALTMOV"
        return_to_new_tr_iteration = False  # return and quit
        return model, nf, nx, last_successful_iter, diffs, return_to_new_tr_iteration, exit_flag, exit_str

    if xnew is None:  # issue with stpsav occurred, quit DFOGN
        exit_flag = EXIT_ALTMOV_MEMORY_ERROR
        exit_str = "Error in ALTMOV - stpsav undefined"
        return_to_new_tr_iteration = False  # return and quit
        return model, nf, nx, last_successful_iter, diffs, return_to_new_tr_iteration, exit_flag, exit_str

    if denom < cauchy and cauchy > 0.0:
        xnew = xalt.copy()

    d = xnew - model.xopt()

    # [Fortran label 360]
    x = model.x_within_bounds(x=xnew)
    nsamples_to_use = nsamples(delta, rho, current_iter, nruns_so_far, SCEN_GEOM_UPDATE)
    v_err_list, f_list, nf, nx, nsamples_run, exit_flag, exit_str = sample_objective(model.m, objfun, x, nf, nx, maxfun,
                                                                                     model.min_objective_value(),
                                                                                     nsamples=nsamples_to_use)

    # Handle exit conditions (f < min obj value or maxfun reached)
    if exit_flag is not None:  # then exit_str is also set
        if nsamples_run > 0:
            fmin = np.min(f_list[:nsamples_run])
            if fmin < model.fsave:
                model.xsave = x
                model.fsave = fmin
        return_to_new_tr_iteration = False  # return and quit
        return model, nf, nx, last_successful_iter, diffs, return_to_new_tr_iteration, exit_flag, exit_str

    # Otherwise, add new results
    model.update_point(knew, xnew, v_err_list[0, :], f_list[0])  # increments model.npt_so_far, if still growing
    for i in range(1, nsamples_run):
        model.add_point_resample(knew, v_err_list[i, :])  # add new info

    # Estimate actual reduction to add to diffs vector
    f = sumsq(np.mean(v_err_list[:nsamples_run, :], axis=0))  # estimate actual objective value

    # Use the quadratic model to predict the change in F due to the step D,
    # and set DIFF to the error of this prediction.
    gopt, hq = model.build_full_model()
    if gopt is None:  # Use this to indicate linalg error
        if f < model.fval_opt():
            # Force model.get_final_results() to return this new point if it's better than xopt, then quit
            model.xsave = x
            model.fsave = f
        exit_flag = EXIT_LINALG_ERROR
        exit_str = "Singular matrix encountered in FIX_GEOMETRY (full model interpolation step)"
        return_to_new_tr_iteration = False  # return and quit
        return model, nf, nx, last_successful_iter, diffs, return_to_new_tr_iteration, exit_flag, exit_str

    pred_reduction = - calculate_model_value(gopt, hq, d)
    actual_reduction = model.fval_opt() - f
    diffs = [abs(pred_reduction - actual_reduction), diffs[0], diffs[1]]

    if dnorm > rho:
        last_successful_iter = current_iter

    exit_flag = None
    exit_str = None
    return_to_new_tr_iteration = True  # return and start new trust region iteration (label 60)
    return model, nf, nx, last_successful_iter, diffs, return_to_new_tr_iteration, exit_flag, exit_str


def dfogn_main(objfun, x0, xl, xu, rhobeg, rhoend, maxfun, nsamples, m=None, delta_scale_for_new_dirns_when_growing=1.0,
               use_random_initial_directions=False, ndirs_initial=None, num_geom_steps_when_growing=1, nf_so_far=0,
               nx_so_far=0, nruns_so_far=0):
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
    # It shouldn't ever happen, but make sure ndirs_initial is not None
    if ndirs_initial is None:
        ndirs_initial = np.size(x0)

    model, nf, nx, return_to_new_tr_iteration, exit_flag, exit_str = \
        build_initial_set(objfun, x0, xl, xu, rhobeg, maxfun, nsamples, nf_so_far, nx_so_far, ndirs_initial,
                          nruns_so_far, m=m, random_initial_directions=use_random_initial_directions)

    if not return_to_new_tr_iteration:
        x, f = model.get_final_results()
        return x, f, nf, nx, exit_flag, exit_str, model.m

    ###########################################################
    # Set other variables before begin iterations
    ###########################################################
    finished_main_loop = False

    (rho, delta) = (rhobeg, rhobeg)
    diffs = [0.0, 0.0, 0.0]  # (diffa, diffb, diffc) in Fortran code, used in done_with_current_rho()

    ###########################################################
    # Start of main loop [Fortran label 60]
    ###########################################################
    current_iter = -1
    last_successful_iter = 0
    while not finished_main_loop:
        current_iter += 1
        logging.debug("Iter %g (last successful %g) with delta = %g and rho = %g" % (
                        current_iter, last_successful_iter, delta, rho))

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

        if dnorm < 0.5 * rho and model.npt_so_far < model.n + 1:
            # Failed TR step during initial phase - add a point and see if that helps
            logging.debug("Failed trust region step during growing phase - adding new directions")
            dnew_matrix = get_new_orthogonal_directions(model, delta_scale_for_new_dirns_when_growing * delta,
                                                            num_steps=num_geom_steps_when_growing)
            break_main_loop = False  # the internal breaks only quit this inner loop!
            for j in range(num_geom_steps_when_growing):
                xnew = model.xopt() + dnew_matrix[:, j]
                logging.debug("Growing: compulsory geometry improving step xnew = %s" % str(xnew))
                x = model.x_within_bounds(x=xnew)

                nsamples_to_use = nsamples(delta, rho, current_iter, nruns_so_far, SCEN_GROWING_NEW_DIRECTION)
                v_err_list, f_list, nf, nx, nsamples_run, exit_flag, exit_str = \
                    sample_objective(model.m, objfun, x, nf, nx, maxfun, model.min_objective_value(),
                                     nsamples=nsamples_to_use)

                # Handle exit conditions (f < min obj value or maxfun reached)
                if exit_flag is not None:  # then exit_str is also set
                    if nsamples_run > 0:
                        fmin = np.min(f_list[:nsamples_run])
                        if fmin < model.fsave:
                            model.xsave = x
                            model.fsave = fmin
                    break_main_loop = True
                    break  # quit inner loop over j, then quit main iteration

                if model.npt_so_far < model.npt:  # still growing
                    kmin = model.npt_so_far
                    logging.debug("Updating point kmin=%g, since still growing" % kmin)
                else:  # full set
                    kmin, linalg_error = choose_knew(model, delta, xnew, skip_kopt=True)

                    if linalg_error:
                        exit_flag = EXIT_LINALG_ERROR
                        exit_str = "Singular matrix when finding kmin (in main loop)"
                        break_main_loop = True
                        break  # quit inner loop over j, then quit main iteration
                    logging.debug("Updating point kmin=%g, chosen in usual way" % kmin)

                # Otherwise, add new results, incrementing model.npt_so_far (if still growing)
                model.update_point(kmin, xnew, v_err_list[0, :], f_list[0])
                for i in range(1, nsamples_run):
                    model.add_point_resample(kmin, v_err_list[i, :])  # add new info

            # Finished adding new directions - restart main trust region iteration (if no errors encountered)
            if break_main_loop:
                finished_main_loop = True
                break  # quit
            else:
                finished_main_loop = False
                continue  # next trust region step

        elif dnorm < 0.5 * rho:
            ###################
            # Start failed TR step
            ###################
            logging.debug("Failed trust region step (main phase)")

            if not done_with_current_rho(model, current_iter, last_successful_iter, rho, diffs, xnew, gnew, hq,
                                         crvmin):
                # [Fortran label 650]
                distsq = (10.0 * rho) ** 2
                model, delta, nf, nx, last_successful_iter, diffs, return_to_new_tr_iteration, geom_exit_flag, geom_exit_str = \
                    check_and_fix_geometry(model, objfun, distsq, delta, rho, dnorm, diffs, nf, nx, current_iter,
                                           last_successful_iter, maxfun, nsamples, rounding_error_const, nruns_so_far, update_delta=True)

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
                last_successful_iter = current_iter
                finished_main_loop = False
                continue  # next trust region step
            else:
                # Cannot reduce rho, so check xnew and quit
                x = model.x_within_bounds(x=xnew)
                nsamples_to_use = nsamples(delta, rho, current_iter, nruns_so_far, SCEN_RHOEND_REACHED)
                v_err_list, f_list, nf, nx, nsamples_run, exit_flag, exit_str = \
                    sample_objective(model.m, objfun, x, nf, nx, maxfun, model.min_objective_value(),
                                     nsamples=nsamples_to_use)

                # Handle exit conditions (f < min obj value or maxfun reached)
                if exit_flag is not None:  # then exit_str is also set
                    if nsamples_run > 0:
                        fmin = np.min(f_list[:nsamples_run])
                        if fmin < model.fsave:
                            model.xsave = x
                            model.fsave = fmin
                    finished_main_loop = True
                    break  # quit

                # Force model.get_final_results() to return this new point if it's better than xopt, then quit
                model.xsave = x
                model.fsave = np.min(f_list[:nsamples_run])
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
                model.shift_base(model.xopt())  # includes a re-factorisation of the interpolation matrix
                xnew = xnew - model.xopt()

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

            nsamples_to_use = nsamples(delta, rho, current_iter, nruns_so_far, SCEN_TR_STEP)
            v_err_list, f_list, nf, nx, nsamples_run, exit_flag, exit_str = \
                sample_objective(model.m, objfun, x, nf, nx, maxfun, model.min_objective_value(),
                                 nsamples=nsamples_to_use)
            # Estimate f in order to compute 'actual reduction'
            f = sumsq(np.mean(v_err_list[:nsamples_run, :], axis=0))  # estimate actual objective value

            # Handle exit conditions (f < min obj value or maxfun reached)
            if exit_flag is not None:  # then exit_str is also set
                if nsamples_run > 0:
                    fmin = np.min(f_list[:nsamples_run])
                    if fmin < model.fsave:
                        model.xsave = x
                        model.fsave = fmin
                finished_main_loop = True
                break  # quit

            # Use the quadratic model to predict the change in F due to the step D,
            # and set DIFF to the error of this prediction.
            pred_reduction = - calculate_model_value(gopt, hq, d)
            actual_reduction = model.fval_opt() - f
            diffs = [abs(pred_reduction - actual_reduction), diffs[0], diffs[1]]

            if dnorm > rho:
                last_successful_iter = current_iter

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
            model.update_point(knew, xnew, v_err_list[0, :], f_list[0])  # increments model.npt_so_far, if still growing
            for i in range(1, nsamples_run):
                model.add_point_resample(knew, v_err_list[i, :])  # add new info

            # When growing and don't yet have a full set of directions, we always need a geometry improving step
            if model.npt_so_far <= model.n + 1:  # even after npt function evaluations, still one direction short
                dnew_matrix = get_new_orthogonal_directions(model, delta_scale_for_new_dirns_when_growing * delta,
                                                            num_steps=num_geom_steps_when_growing)
                # breaks below only stop num_geom_steps_when_growing loop, check if need to quit main loop too
                break_main_loop = False
                for j in range(num_geom_steps_when_growing):
                    xnew = model.xopt() + dnew_matrix[:, j]
                    logging.debug("Growing: compulsory geometry improving step xnew = %s" % str(xnew))
                    x = model.x_within_bounds(x=xnew)

                    nsamples_to_use = nsamples(delta, rho, current_iter, nruns_so_far, SCEN_GROWING_NEW_DIRECTION)
                    v_err_list, f_list, nf, nx, nsamples_run, exit_flag, exit_str = \
                        sample_objective(model.m, objfun, x, nf, nx, maxfun, model.min_objective_value(),
                                         nsamples=nsamples_to_use)

                    # Handle exit conditions (f < min obj value or maxfun reached)
                    if exit_flag is not None:  # then exit_str is also set
                        if nsamples_run > 0:
                            fmin = np.min(f_list[:nsamples_run])
                            if fmin < model.fsave:
                                model.xsave = x
                                model.fsave = fmin
                        finished_main_loop = True
                        break_main_loop = True
                        break  # quit

                    if model.npt_so_far < model.npt:  # still growing
                        kmin = model.npt_so_far
                        logging.debug("Updating point kmin=%g, since still growing" % kmin)
                    else:  # full set
                        kmin, linalg_error = choose_knew(model, delta, xnew, skip_kopt=True)

                        if linalg_error:
                            exit_flag = EXIT_LINALG_ERROR
                            exit_str = "Singular matrix when finding kmin (in main loop)"
                            finished_main_loop = True
                            break_main_loop = True
                            break  # quit
                        logging.debug("Updating point kmin=%g, chosen in usual way" % kmin)

                    # Otherwise, add new results, incrementing model.npt_so_far (if still growing)
                    model.update_point(kmin, xnew, v_err_list[0, :], f_list[0])
                    for i in range(1, nsamples_run):
                        model.add_point_resample(kmin, v_err_list[i, :])  # add new info

                # Finished adding new directions - restart main trust region iteration (if no errors encountered)
                if break_main_loop:
                    finished_main_loop = True
                    break  # quit

            # If a trust region step has provided a sufficient decrease in F, then
            # branch for another trust region calculation.
            if ratio >= 0.1:
                finished_main_loop = False
                continue  # next trust region step

            # Alternatively, find out if the interpolation points are close enough
            # to the best point so far.
            # [Fortran label 650]
            distsq = max((2.0 * delta) ** 2, (10.0 * rho) ** 2)
            model, delta, nf, nx, last_successful_iter, diffs, return_to_new_tr_iteration, geom_exit_flag, geom_exit_str = \
                check_and_fix_geometry(model, objfun, distsq, delta, rho, dnorm, diffs, nf, nx, current_iter,
                                       last_successful_iter, maxfun, nsamples, rounding_error_const, nruns_so_far,
                                       update_delta=False)  # don't update delta when ntrits > 0

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
                last_successful_iter = current_iter
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

    x, f = model.get_final_results()
    logging.debug("At return from DFOGN, number of function evals = %i" % nf)
    logging.debug("Smallest objective value = %.15g at x = " % f + str(x))
    return x, f, nf, nx, exit_flag, exit_str, model.m


def dfogn_resampling(objfun, x0, lower=None, upper=None, maxfun=1000, nsamples=None, init_tr_radius=None, rhoend=1e-8,
          delta_scale_for_new_dirns_when_growing=1.0, use_random_initial_directions=False,
          ndirs_initial='n', num_geom_steps_when_growing=1, use_restarts=True,
          max_unsuccessful_restarts=10):
    # If bounds not provided, set to something large
    xl = (lower if lower is not None else -1.0e20 * np.ones(x0.shape))
    xu = (upper if upper is not None else 1.0e20 * np.ones(x0.shape))

    # Set default value of rhobeg to something sensible
    rhobeg = (init_tr_radius if init_tr_radius is not None else 0.1 * max(np.max(np.abs(x0)), 1.0))

    # Set default number of samples to be 1 for every evaluation
    if nsamples is None:
        nsamples_to_use = lambda delta, rho, iter, nruns_so_far, scenario : 1
    else:
        nsamples_to_use = nsamples

    n = np.size(x0)
    assert (rhobeg > 0.0), "rhobeg must be strictly positive"
    assert (rhoend > 0.0), "rhoend must be strictly positive"
    assert (rhoend < rhobeg), "rhoend must be less than rhobeg"
    assert (maxfun > 0), "maxfun must be strictly positive"
    assert (np.shape(x0) == (n,)), "x0 must be a vector"
    assert (np.shape(x0) == np.shape(xl)), "xl must have same shape as x0"
    assert (np.shape(x0) == np.shape(x0)), "xu must have same shape as x0"
    assert (np.all(xu-xl >= 2.0*rhobeg)), "gap between xl and xu must be at least 2*rhobeg"

    if maxfun <= n+1:
        logging.warning("Warning (maxfun <= n+1): Are you sure your budget is large enough?")

    # Parse string arguments: number of geometry steps to take at each growing iteration of main TR loop
    n_extra_steps_to_use = None
    if type(num_geom_steps_when_growing) == int:
        n_extra_steps_to_use = num_geom_steps_when_growing
    elif type(num_geom_steps_when_growing) == str:
        if num_geom_steps_when_growing == 'tenthn':
            n_extra_steps_to_use = int(x0.size // 10)
        elif num_geom_steps_when_growing == 'fifthn':
            n_extra_steps_to_use = int(x0.size // 5)
        elif num_geom_steps_when_growing == 'qtrn':
            n_extra_steps_to_use = int(x0.size // 4)

    assert n_extra_steps_to_use is not None, "Unknown num_geom_steps_when_growing: " + str(
        num_geom_steps_when_growing)
    n_extra_steps_to_use = max(n_extra_steps_to_use, 1)  # floor at 1

    # Parse string arguments: number of initial directions to add before beginning main TR loop
    ndirs_initial_val = None
    if type(ndirs_initial) == int:
        ndirs_initial_val = ndirs_initial
    elif type(ndirs_initial) == str:
        if ndirs_initial == 'tenthn':
            ndirs_initial_val = int(n // 10)
        elif ndirs_initial == 'fifthn':
            ndirs_initial_val = int(n // 5)
        elif ndirs_initial == 'qtrn':
            ndirs_initial_val = int(n // 4)
        elif ndirs_initial == 'halfn':
            ndirs_initial_val = int(n // 2)
        elif ndirs_initial == 'n':
            ndirs_initial_val = n
        elif ndirs_initial == '2n':
            ndirs_initial_val = 2 * n
        elif ndirs_initial == 'nsq':
            ndirs_initial_val = (n + 1) * (n + 2) // 2 - 1

    assert ndirs_initial_val is not None, "Unknown ndirs_initial: " + str(ndirs_initial)
    assert ndirs_initial_val == n, "Must have n initial directions (build_interp_matrix assumes this)"
    ndirs_initial_val = max(ndirs_initial_val, 1)  # floor at 1

    # Enforce lower bounds on x0 (ideally with gap of at least rhobeg)
    idx = (xl < x0) & (x0 <= xl+rhobeg)
    x0[idx] = xl[idx] + rhobeg

    idx = (x0 <= xl)
    x0[idx] = xl[idx]

    # Enforce upper bounds on x0 (ideally with gap of at least rhobeg)
    idx = (xu-rhobeg <= x0) & (x0 < xu)
    x0[idx] = xu[idx] - rhobeg

    idx = (x0 >= xu)
    x0[idx] = xu[idx]

    # First run
    x, f, nf, nx, exit_flag, exit_str, m = \
        dfogn_main(objfun, x0, xl, xu, rhobeg, rhoend, maxfun, nsamples_to_use, m=None,
                   delta_scale_for_new_dirns_when_growing=delta_scale_for_new_dirns_when_growing,
                   use_random_initial_directions=use_random_initial_directions, ndirs_initial=ndirs_initial_val,
                   num_geom_steps_when_growing=n_extra_steps_to_use,
                   nf_so_far=0, nx_so_far=0, nruns_so_far=0)

    # Now do repeats
    nruns_so_far = 1
    reduction_last_run = True  # did the last run give us a reduction?
    rhobeg_to_use = rhobeg
    rhoend_to_use = rhoend
    last_successful_run = 1

    while use_restarts and nf < maxfun and nruns_so_far - last_successful_run < max_unsuccessful_restarts and \
            ((exit_flag == EXIT_SUCCESS and 'rho' in exit_str) or exit_flag in DO_RESTART_ERRORS):
        if reduction_last_run:
            rhobeg_to_use = max(0.1 * max(np.max(np.abs(x)), 1.0), 10 * rhoend_to_use)
            rhoend_to_use = 1.0 * rhoend_to_use
        else:
            # Reduce initial TR radius when things have been going badly
            rhobeg_to_use = max(0.5 * rhobeg_to_use, 10 * rhoend_to_use)
        logging.info(
            "Restarting from finish point (f = %g) after %g function evals; new rhobeg = %g and rhoend = %g" % (
            f, nf, rhobeg_to_use, rhoend_to_use))
        x2, f2, nf, nx, exit_flag, exit_str, m_tmp = \
            dfogn_main(objfun, x, xl, xu, rhobeg_to_use, rhoend_to_use, maxfun, nsamples_to_use, m=m,
                       delta_scale_for_new_dirns_when_growing=delta_scale_for_new_dirns_when_growing,
                       use_random_initial_directions=use_random_initial_directions, ndirs_initial=ndirs_initial_val,
                       num_geom_steps_when_growing=n_extra_steps_to_use,
                       nf_so_far=nf, nx_so_far=nx, nruns_so_far=nruns_so_far)

        nruns_so_far += 1
        if f2 < f or np.isnan(f):
            logging.info("Successful run with new f = %s compared to old f = %s" % (f2, f))
            last_successful_run = nruns_so_far
            x = x2
            f = f2
            reduction_last_run = True
        else:
            logging.info("Unsuccessful run with new f = %s compared to old f = %s" % (f2, f))
            reduction_last_run = False

    logging.info("Finished after a total of %g runs" % nruns_so_far)

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

    return x, f, nf, exit_flag, exit_str
