=====================================================================
DFO-GN: Derivative-Free Nonlinear Least-Squares Solver |PyPI Version|
=====================================================================
DFO-GN is a package for solving nonlinear least-squares minimisation, without requiring derivatives of the objective.

This is an implementation of the algorithm from our paper:
`A Derivative-Free Gauss-Newton Method <https://arxiv.org/abs/1710.11005>`_, C. Cartis and L. Roberts, submitted (2017). For reproducibility of all figures in this paper, please feel free to contact the authors.

Note: we have released a newer package, called DFO-LS, which is an upgrade of DFO-GN to improve its flexibility and robustness to noisy problems. See `here <https://github.com/numericalalgorithmsgroup/dfols>`_ for details.

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

    $ python setup.py test

Alternatively, the `documentation <https://numericalalgorithmsgroup.github.io/dfogn/>`_ provides some simple examples of how to run DFO-GN, which are also available in the examples directory.

.. |PyPI Version| image:: https://img.shields.io/pypi/v/DFOGN.svg
                  :target: https://pypi.python.org/pypi/DFOGN
