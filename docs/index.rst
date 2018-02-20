.. DFO-GN documentation master file, created by
   sphinx-quickstart on Thu Aug  3 16:29:21 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

DFO-GN: A Derivative-Free Gauss-Newton Solver
=============================================

**Release:** |version|

**Date:** |today|

**Author:** `Lindon Roberts <lindon.roberts@maths.ox.ac.uk>`_ (Mathematical Institute, University of Oxford)

DFO-GN is a Python package for finding local solutions to **nonlinear least-squares minimization problems (with optional bound constraints)**, without requiring any derivatives of the objective. DFO-GN stands for Derivative-Free Optimization using Gauss-Newton, and is applicable to problems such as

* Parameter estimation/data fitting;
* Solving systems of nonlinear equations (including under- and over-determined systems); and
* Inverse problems, including data assimilation.

DFO-GN is a *derivative-free* algorithm, meaning it does not require any information about the derivative of the objective, nor does it attempt to estimate such information (e.g. by using finite differencing). This means that it is **particularly useful for solving noisy problems**; i.e. where evaluating the objective function several times for the same input may give different results.

Mathematically, DFO-GN solves

.. math::

   \min_{x\in\mathbb{R}^n}  &\quad  f(x) := \sum_{i=1}^{m}r_{i}(x)^2 \\
   \text{s.t.} &\quad  a \leq x \leq b

where the functions :math:`r_i(x)` may be nonlinear and even nonconvex. Full details of the DFO-GN algorithm are given in our paper: `A Derivative-Free Gauss-Newton Method <https://arxiv.org/abs/1710.11005>`_, C. Cartis and L. Roberts, submitted (2017).

DFO-GN is released under the open source GNU General Public License, a copy of which can be found in LICENSE.txt. Please `contact NAG <http://www.nag.com/content/worldwide-contact-information>`_ for alternative licensing. It is compatible with both Python 2 and Python 3.

Note: we have released a newer package, called DFO-LS, which is an upgrade of DFO-GN to improve its flexibility and robustness to noisy problems. See `here <https://github.com/numericalalgorithmsgroup/dfols>`_ for details.

If you have any questions or suggestsions about the code, or have used DFO-GN for an interesting application, we would very much like to hear from you: please contact `Lindon Roberts <lindon.roberts@maths.ox.ac.uk>`_ (`alternative email <lindon.roberts@gmail.com>`_).

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   info
   install
   userguide
   history

Acknowledgements
----------------
This software was developed under the supervision of `Coralia Cartis <https://www.maths.ox.ac.uk/people/coralia.cartis>`_ (Mathematical Institute, University of Oxford), and was supported by the EPSRC Centre For Doctoral Training in `Industrially Focused Mathematical Modelling <https://www.maths.ox.ac.uk/study-here/postgraduate-study/industrially-focused-mathematical-modelling-epsrc-cdt>`_ (EP/L015803/1) at the University of Oxford's Mathematical Institute, in collaboration with the `Numerical Algorithms Group <http://www.nag.com/>`_.

DFO-GN was developed using techniques from DFBOLS (`Zhang, Conn & Scheinberg, 2010 <https://doi.org/10.1137/09075531X>`_) and BOBYQA (`Powell, 2009 <http://mat.uc.pt/~zhang/software.html>`_).
The structure of this documentation is from `oBB <http://pythonhosted.org/oBB/>`_ by Jari Fowkes.

