Installing DFO-GN
=================

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

Uninstallation
--------------
If DFO-GN was installed using `pip <http://www.pip-installer.org/>`_ you can uninstall as follows:

 .. code-block:: bash

    $ [sudo] pip uninstall dfogn

If DFO-GN was installed manually you have to remove the installed files by hand (located in your python site-packages directory).


