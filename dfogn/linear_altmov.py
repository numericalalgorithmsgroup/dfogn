"""
linear_altmov
==================
A geometry-improving step calculation (assuming linear models).

Here, ALTMOV needs to maximise |l(y)| for linear Lagrange polynomial l(y)
in a box intersect ball. We can solve this exactly via an active set approach.

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


__all__ = ["max_step_in_box_and_ball"]

ZERO_THRESH = 1e-14


def ball_step(x0, g, Delta):
    # Given initial point x0, take largest step in direction g allowed by ||x|| <= Delta
    # That is, solve
    #   ||x0 + alpha*g||^2 = Delta^2, alpha >= 0
    # Using this method, solution exists whenever ||x0|| <= Delta^2 [take alpha=0 if g=0]
    gdotx0 = np.dot(g, x0)
    gsqnorm = np.dot(g, g)
    x0sqnorm = np.dot(x0, x0)
    if sqrt(gsqnorm) < ZERO_THRESH:  # Error catching: if g=0, make no step
        return 0.0
    return (sqrt(gdotx0**2 + gsqnorm*(Delta**2 - x0sqnorm)) - gdotx0) / gsqnorm


def min_linear_in_box_and_ball(g, a, b, Delta):
    # Solve the convex program:
    #   min_x   g' * x
    #   s.t.   a <= x <= b
    #           ||x||^2 <= Delta^2
    # using an active-set type approach

    n = g.size
    x = np.zeros((n,))
    dirn = -g
    cons_dirns = []

    # If g[i] = 0, never step along this direction
    constant_directions = np.where(np.abs(dirn) < ZERO_THRESH)[0]
    dirn[constant_directions] = 0.0
    cons_dirns += list(constant_directions)

    for i in range(n):
        if np.linalg.norm(dirn) < ZERO_THRESH:
            return x
        alpha_unc = ball_step(x, dirn, Delta)
        xnew = x + alpha_unc * dirn
        # Check if hit box bounds
        on_box_bdry = False
        hit_upper = None
        idx_hit = None
        for j in range(n):
            if j in cons_dirns:
                continue  # only looking at unconstrained directions
            if xnew[j] <= a[j]:
                on_box_bdry = True
                hit_upper = False
                idx_hit = j
                break
            elif xnew[j] >= b[j]:
                on_box_bdry = True
                hit_upper = True
                idx_hit = j
                break

        if not on_box_bdry:
            return xnew  # unconstrained solution
        else:
            # Go as far as possible until hit box, then remove that direction from 'dirn'
            cons_dirns.append(idx_hit)  # new constrained direction
            alpha_con = ((b[idx_hit] if hit_upper else a[idx_hit]) - x[idx_hit]) / dirn[idx_hit]
            x = x + alpha_con * dirn
            x[idx_hit] = b[idx_hit] if hit_upper else a[idx_hit]  # force boundary exactly
            dirn[idx_hit] = 0.0  # no more searching this direction
    return x


def max_linear_in_box_and_ball(g, a, b, Delta):
    # Solve the convex program:
    #   max_x   g' * x
    #   s.t.   a <= x <= b
    #           ||x||^2 <= Delta^2
    return min_linear_in_box_and_ball(-g, a, b, Delta)


def max_abs_linear_in_box_and_ball(g, a, b, Delta, c=0):
    # Solve the program:
    #   max_x   abs(c + g' * x)
    #   s.t.   a <= x <= b
    #           ||x||^2 <= Delta^2
    # by maximising and minimising (g' * x) separately
    xmin = min_linear_in_box_and_ball(g, a, b, Delta)
    xmax = max_linear_in_box_and_ball(g, a, b, Delta)

    if abs(c + np.dot(g, xmin)) >= abs(c + np.dot(g, xmax)):
        return xmin
    else:
        return xmax


def max_step_in_box_and_ball(xbase, c, g, lower, upper, Delta):
    # Consider the problem of maximising the step. That is, solve
    #   min_x  abs(c + g' * (x - xbase))
    #    s.t.  lower <= x <= upper
    #          ||x-xbase|| <= Delta
    # Setting s = x-xbase (or x = xbase + s), this is equivalent to:
    #   min_s  abs(c + g' * s)
    #   s.t.   lower - xbase <= s <= upper - xbase
    #          ||s|| <= Delta
    s = max_abs_linear_in_box_and_ball(g, lower-xbase, upper-xbase, Delta, c=c)
    return xbase + s

if __name__ == '__main__':
    g = np.array([1.0, -1.0])
    a = np.array([-2.0, -2.0])
    b = np.array([1.0, 2.0])
    delta = 2.0
    c = -1.0
    xmin = min_linear_in_box_and_ball(g, a, b, delta)
    xmax = max_linear_in_box_and_ball(g, a, b, delta)
    xabs = max_abs_linear_in_box_and_ball(g, a, b, delta, c=c)
    print("Min: x = %s, c + g'*x = %g" % (str(xmin), c + np.dot(g, xmin)))
    print("Max: x = %s, c + g'*x = %g" % (str(xmax), c + np.dot(g, xmax)))
    print("Max abs: take x = %s, abs(g'*x) = %g" % (str(xabs), abs(c + np.dot(g, xabs))))
    print("Done")
