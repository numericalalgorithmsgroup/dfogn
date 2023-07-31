======================================================
DFO-GN: Derivative-Free Nonlinear Least-Squares Solver
======================================================

.. image::  https://travis-ci.org/numericalalgorithmsgroup/dfogn.svg?branch=master
   :target: https://travis-ci.org/numericalalgorithmsgroup/dfogn
   :alt: Build Status

.. image::  https://img.shields.io/badge/License-GPL%20v3-blue.svg
   :target: https://www.gnu.org/licenses/gpl-3.0
   :alt: GNU GPL v3 License

.. image:: https://img.shields.io/pypi/v/DFOGN.svg
   :target: https://pypi.python.org/pypi/DFOGN
   :alt: Latest PyPI version

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.2629875.svg
   :target: https://doi.org/10.5281/zenodo.2629875
   :alt: DOI:10.5281/zenodo.2629875

DFO-GN is a package for solving nonlinear least-squares minimisation, without requiring derivatives of the objective.

This is an implementation of the algorithm from our paper:
`A Derivative-Free Gauss-Newton Method <https://doi.org/10.1007/s12532-019-00161-7>`_, C. Cartis and L. Roberts, Mathematical Programming Computation (2019). For reproducibility of all figures in this paper, please feel free to contact the authors. A preprint of the paper is available `here <https://arxiv.org/abs/1710.11005>`_.

Note: we have released a newer package, called DFO-LS, which is an upgrade of DFO-GN to improve its flexibility and robustness to noisy problems. See `here <https://github.com/numericalalgorithmsgroup/dfols>`_ for details.

**Citation** To cite DFO-GN, please use
::
   @Article{DFOGN,
     Title    = {A derivative-free {G}auss-{N}ewton method},
     Author   = {Cartis, Coralia and Roberts, Lindon},
     Journal  = {Mathematical Programming Computation},
     Year     = {2019},
     Doi      = {10.1007/s12532-019-00161-7},
     Url      = {https://doi.org/10.1007/s12532-019-00161-7}
   }

Documentation
-------------
See manual.pdf or `here <https://numericalalgorithmsgroup.github.io/dfogn/>`_.

Requirements
------------
DFO-GN requires the following software to be installed:

* `Python 2.7 or Python 3 <http://www.python.org/>`_

Additionally, the following python packages should be installed (these will be installed automatically if using `pip <http://www.pip-installer.org/>`_, see `Installation using pip`_):

* `NumPy 1.11 or higher <http://www.numpy.org/>`_ 
* `SciPy 0.18 or higher <http://www.scipy.org/>`_


Installation using pip
----------------------
For easy installation, use `pip <http://www.pip-installer.org/>`_ as root:

 .. code-block:: bash

    $ [sudo] pip install --pre dfogn

If you do not have root privileges or you want to install DFO-GN for your private use, you can use:

 .. code-block:: bash

    $ pip install --pre --user dfogn
      
which will install DFO-GN in your home directory.

Note that if an older install of DFO-GN is present on your system you can use:

 .. code-block:: bash

    $ [sudo] pip install --pre --upgrade dfogn
      
to upgrade DFO-GN to the latest version.

Manual installation
-------------------
The source code for DFO-GN is `available on Github <https://https://github.com/numericalalgorithmsgroup/dfogn>`_:

 .. code-block:: bash
 
    $ git clone https://github.com/numericalalgorithmsgroup/dfogn
    $ cd dfogn

DFO-GN is written in pure Python and requires no compilation. It can be installed using:

 .. code-block:: bash

    $ [sudo] pip install --pre .

If you do not have root privileges or you want to install DFO-GN for your private use, you can use:

 .. code-block:: bash

    $ pip install --pre --user .
    
instead.    

Testing
-------
If you installed DFO-GN manually, you can test your installation by running:

 .. code-block:: bash

    $ pytest

for Python 3.7+ (need `pytest <http://pytest.org>`_)

 .. code-block:: bash

    $ python setup.py test

for Python 2.7

Alternatively, the `documentation <https://numericalalgorithmsgroup.github.io/dfogn/>`_ provides some simple examples of how to run DFO-GN, which are also available in the examples directory.

