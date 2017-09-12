"""
alternative_move
==================
A geometry-improving step calculation.
Based on the routine ALTMOV from BOBYQA (Powell, 2009).


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

from math import sqrt
import numpy as np

__all__ = ['altmov']


# Exact interpolation for n+1 points (i.e. linear models only)
# Also works for inexact interpolation and n+1 points (when still linear models) - see above function definition
def altmov(xpt, sl, su, kopt, xopt, knew, adelt, H_knew):
    npt, n = xpt.shape
    assert sl.size == n, "sl and xpt have incompatible sizes"
    assert su.size == n, "su and xpt have incompatible sizes"
    assert xopt.size == n, "xopt and xpt have incompatible sizes"
    assert 0 <= kopt <= npt-1, "kopt must be in range 0, ..., npt-1"
    assert adelt > 0.0, "adelt must be strictly positive"
    assert H_knew.size == n, "H_knew and xpt have incompatible sizes"
    # H_knew = knew-th column of H (used to construct knew-th Lagrange polynomial)
    # Outputs:
    #xnew = np.zeros((n,))
    xalt = np.zeros((n,))
    cauchy = 0.0

    # Calculate the gradient of the KNEW-th Lagrange function at XOPT.
    glag = H_knew  # 0th entry is constant term, rest is gradient
    yt = xpt[knew,:]  # interpolation point to replace

    # Searching along lines {xk + alpha*(yj-xk)} to get xnew = xopt + stpsav * (xpt[jsav, :] - xopt)
    jsav = None
    stpsav = None
    best_abs_phi = -1.0  # way to compare different values of j

    for j in range(npt):
        if j==kopt:
            continue
        yj = xpt[j, :]

        # Get bounds on alpha_j from sl, su and adelt
        alpha_lower = -adelt / np.linalg.norm(yj - xopt)
        alpha_upper = adelt / np.linalg.norm(yj - xopt)
        for i in range(n):
            if yj[i]-xopt[i] > 0.0:
                alpha_lower = max(alpha_lower, (sl[i] - xopt[i]) / (yj[i] - xopt[i]))
                alpha_upper = min(alpha_upper, (su[i] - xopt[i]) / (yj[i] - xopt[i]))
            elif yj[i]-xopt[i] < 0.0:
                alpha_lower = max(alpha_lower, (su[i] - xopt[i]) / (yj[i] - xopt[i]))
                alpha_upper = min(alpha_upper, (sl[i] - xopt[i]) / (yj[i] - xopt[i]))

        # Since the Lagrange poly is linear, optimal alpha is either lower or upper value
        phi_lower = 1.0 + np.dot(xopt - yt, glag) + alpha_lower * np.dot(yj - xopt, glag)
        phi_upper = 1.0 + np.dot(xopt - yt, glag) + alpha_upper * np.dot(yj - xopt, glag)
        if abs(phi_lower) > abs(phi_upper):
            alpha_j = alpha_lower
            abs_phi_j = abs(phi_lower)
        else:
            alpha_j = alpha_lower
            abs_phi_j = abs(phi_lower)

        # Compare to best so far
        if stpsav is None or abs_phi_j > best_abs_phi:
            jsav = j
            stpsav = alpha_j
            best_abs_phi = abs_phi_j

    # Finish up this search
    xnew = xopt + stpsav * (xpt[jsav, :] - xopt)
    xnew = np.maximum(sl, np.minimum(xnew, su))

    # Alternative (xalt) - try a standard Cauchy step for
    # Output 'cauchy' is set to |Lambda_t(xalt)| for later comparison against |Lambda_t(xnew)|
    # In BOBYQA, there were two steps: generate sk from linear model, then scale to get ck (using curvature)
    # Here, only first step needed, and the code is mostly reused

    # Prepare for the iterative method that assembles the constrained Cauchy
    # step in W. The sum of squares of the fixed components of W is formed in
    # WFIXSQ, and the free components of W are set to BIGSTP.
    bigstp = 2.0 * adelt
    xalt_backup = np.zeros((n,))  # w(n+i) for i in range(n)
    iflag = False
    csave = -1.0

    while True:  # loop label 100
        wfixsq = 0.0
        ggfree = 0.0
        w_vec = np.zeros((n,))  # first n components of w
        for i in range(n):
            w_vec[i] = 0.0
            tempa = min(xopt[i] - sl[i], glag[i])
            tempb = max(xopt[i] - su[i], glag[i])
            if tempa > 0.0 or tempb < 0.0:
                w_vec[i] = bigstp
                ggfree += glag[i] ** 2
        if ggfree == 0.0:
            cauchy = 0.0
            return xnew, xalt, cauchy, best_abs_phi

        # Investigate whether more components of W can be fixed.
        while True:  # loop label 120
            temp = adelt ** 2 - wfixsq
            if temp > 0.0:
                wsqsav = wfixsq
                step = sqrt(temp / ggfree)
                ggfree = 0.0
                for i in range(n):
                    if w_vec[i] == bigstp:
                        temp = xopt[i] - step * glag[i]
                        if temp <= sl[i]:
                            w_vec[i] = sl[i] - xopt[i]
                            wfixsq += w_vec[i] ** 2
                        elif temp >= su[i]:
                            w_vec[i] = su[i] - xopt[i]
                            wfixsq += w_vec[i] ** 2
                        else:
                            ggfree += glag[i] ** 2
                if wfixsq > wsqsav and ggfree > 0.0:
                    continue  # next iteration loop label 120
                else:
                    break  # quit loop label 120
            else:
                break  # quit loop label 120

        # Set the remaining free components of W and all components of XALT,
        # except that W may be scaled later.
        for i in range(n):
            if w_vec[i] == bigstp:
                w_vec[i] = -step * glag[i]
                xalt[i] = max(sl[i], min(xopt[i] + w_vec[i], su[i]))
            elif w_vec[i] == 0.0:
                xalt[i] = xopt[i]
            elif glag[i] > 0.0:
                xalt[i] = sl[i]
            else:
                xalt[i] = su[i]
        #gw = np.dot(glag, w_vec)

        # Skip the curvature scaling step, 'cauchy' is just abs(linear objective)
        #cauchy = abs(gw)
        cauchy = abs(1.0 + np.dot(xalt-yt, glag))

        # If IFLAG is zero, then XALT is calculated as before after reversing
        # the sign of GLAG. Thus two XALT vectors become available. The one that
        # is chosen is the one that gives the larger value of CAUCHY.
        # xalt_backup = np.zeros((n,)) # w(n+i) for i in range(n)
        if not iflag:
            glag = -glag
            xalt_backup = xalt.copy()
            csave = cauchy
            iflag = True
            continue  # another iteration of loop label 100
        else:
            break  # quit loop label 100
    # end loop label 100

    # Choose either xalt or xalt_backup depending on cauchy v csave
    if csave > cauchy:
        xalt = xalt_backup.copy()
        cauchy = csave
    return xnew, xalt, cauchy, best_abs_phi
