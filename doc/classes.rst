=============
API Reference
=============

This is the class and function reference of quantification.


.. _cc_ref:

:mod:`quantification.cc`. Classify and Count methods
======================================================

.. automodule:: quantification.cc
   :no-members:
   :no-inherited-members:

Classes
-------
.. currentmodule:: quantification

.. autosummary::
   :toctree: generated/
   :template: class.rst

   cc.BaseBinaryCC
   cc.BinaryCC
   cc.BinaryAC
   cc.BinaryPCC
   cc.BinaryPAC
   cc.BaseMulticlassCC
   cc.MulticlassCC
   cc.MulticlassAC
   cc.MulticlassPCC
   cc.MulticlassPAC
   cc.EnsembleBinaryCC
   cc.EnsembleMulticlassCC


.. _dm_ref:

:mod:`quantification.dm`. Distribution Matching methods
======================================================

.. automodule:: quantification.dm
   :no-members:
   :no-inherited-members:

Classes
-------
.. currentmodule:: quantification

.. autosummary::
   :toctree: generated/
   :template: class.rst

   dm.BinaryHDy
   dm.BinaryEM
   dm.BinaryCDEIter
   dm.MulticlassHDy
   dm.MulticlassEM
   dm.MulticlassCDEIter
   dm.BinaryEnsembleHDy
   dm.BinaryEnsembleEM
   dm.MulticlassEnsembleHDy
   dm.MulticlassEnsembleEM


.. _metrics_ref:

:mod:`quantification.metrics`. Quantification metrics
======================================================

.. automodule:: quantification.metrics
   :no-members:
   :no-inherited-members:

Functions
---------
.. currentmodule:: quantification

.. autosummary::
   :toctree: generated/
   :template: function.rst
   
   metrics.bias
   metrics.square_error
   metrics.kl_divergence
   metrics.absolute_error
   metrics.normalized_absolute_error
   metrics.relative_absolute_error
   metrics.symmetric_absolute_error
   metrics.normalized_square_score
   metrics.normalized_relative_absolute_error
   metrics.bray_curtis
