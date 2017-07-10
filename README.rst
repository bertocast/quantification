.. -*- mode: rst -*-

|Codecov|_

.. |Codecov| image:: https://codecov.io/github/albertorepo/quantification/badge.svg?branch=master&service=github
.. _Codecov: https://codecov.io/github/albertorepo/quantification?branch=master

PyQuan
======

PyQuan is a Python module for quantification learning.

The project was started in 2016 by Alberto Castaño and Juan José del Coz.


Installation
------------

Dependencies
~~~~~~~~~~~~

PyQuan requires:

- Python (>= 2.7 or >= 3.3)
- NumPy (>= 1.8.2)
- SciPy (>= 0.13.3)
- Sklearn

User installation
~~~~~~~~~~~~~~~~~

If you already have a working installation of numpy and scipy,
the easiest way to install PyQuan is using ``pip`` ::

    pip install -U pip install https://github.com/albertorepo/quantification/archive/master.zip


Source code
~~~~~~~~~~~

You can check the latest sources with the command::

    git clone https://github.com/albertorepo/quantification.git


Testing
~~~~~~~

After installation, you can launch the test suite from outside the
source directory (you will need to have the ``nose`` package installed)::

    nosetests -v quantification
