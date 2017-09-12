====================================================================================
DFO-GN: Derivative-Free Nonlinear Least-Squares Solver |Build Status| |PyPI Version|
====================================================================================
DFO-GN is a package for solving nonlinear least-squares minimisation, without requiring derivatives of the objective.

This is an implementation of the algorithm from our paper:
A Derivative-Free Gauss-Newton  Method, C. Cartis and L. Roberts, submitted (2017).

Documentation
-------------
Documentation for DFO-GN is available at [URL].

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

or through the `Python Package Index <https://pypi.python.org/pypi/dfogn>`_:

 .. code-block:: bash

    $ wget http://pypi.python.org/packages/source/d/dfogn/dfogn-X.X.tar.gz
    $ tar -xzvf dfogn-X.X.tar.gz
    $ cd dfogn-X.X

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

Alternatively, the HTML documentation provides some simple examples of how to run DFO-GN.

.. |Build Status| image::  https://travis-ci.org/numericalalgorithmsgroup/DFOGN.svg?branch=master
                  :target: https://travis-ci.org/numericalalgorithmsgroup/DFOGN
.. |PyPI Version| image:: https://img.shields.io/pypi/v/DFOGN.svg
                  :target: https://pypi.python.org/pypi/DFOGN