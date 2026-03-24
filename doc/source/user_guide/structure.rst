Structure of pflm
=================

pflm is organised into five subpackages.  Each one is independently importable so you can pick only the parts you need.

Subpackage overview
-------------------

- ``pflm.fpca`` — Functional Principal Component Analysis

  - Classes: ``FunctionalPCA``, ``FunctionalDataGenerator``, ``FpcaModelParams``, ``SmoothedModelResult``,
    ``FunctionalPCAMuCovParams``, ``FunctionalPCAUserDefinedParams``
  - Functions: ``get_covariance_matrix``, ``get_eigen_analysis_results``, ``estimate_rho``, ``get_fpca_ce_score``,
    ``get_fpca_in_score``, …

- ``pflm.smooth`` — Kernel smoothing / local polynomial regression

  - Classes: ``Polyfit1DModel``, ``Polyfit2DModel``
  - Enum: ``KernelType`` (Gaussian, Epanechnikov, Rectangular, …)

- ``pflm.interp`` — Fast 1D / 2D interpolation (linear & spline)

  - Functions: ``interp1d``, ``interp2d``

- ``pflm.pflm`` — Partial Functional Linear Models

  - Classes: ``PartialFunctionalLinearModel``, ``FPCAConfig``

- ``pflm.pflm.utils`` — ADMM-based elastic-net solver

  - Classes: ``ElasticNet``
  - Enum: ``LinearModelFamily``

- ``pflm.utils`` — Shared utilities

  - ``FlattenFunctionalData``, ``flatten_and_sort_data_matrices``, ``trapz``

Typical imports
---------------

.. code-block:: python

   from pflm.fpca import FunctionalPCA, FunctionalDataGenerator
   from pflm.smooth import Polyfit1DModel, Polyfit2DModel, KernelType
   from pflm.interp import interp1d, interp2d
   from pflm.pflm.partial_flm import PartialFunctionalLinearModel, FPCAConfig
   from pflm.pflm.utils import ElasticNet, LinearModelFamily
   from pflm.utils import FlattenFunctionalData, flatten_and_sort_data_matrices, trapz

Estimator API
-------------

All main models (``FunctionalPCA``, ``Polyfit1DModel``, ``Polyfit2DModel``, ``ElasticNet``,
``PartialFunctionalLinearModel``) follow the
`scikit-learn estimator API <https://scikit-learn.org/stable/developers/develop.html>`_:

- ``model.fit(...)`` trains the model and returns ``self``.
- ``model.predict(...)`` produces predictions on new data.
- Fitted attributes use the trailing underscore convention (``coef_``, ``bandwidth_``, ``xi_``, etc.).
