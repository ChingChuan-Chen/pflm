pflm
****

|license| |python| |docs|

.. |license| image:: https://img.shields.io/badge/license-MIT-blue.svg
   :target: https://github.com/ChingChuan-Chen/pflm/blob/main/LICENSE

.. |python| image:: https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue
   :target: https://www.python.org

.. |docs| image:: https://img.shields.io/badge/docs-GitHub%20Pages-blue
   :target: https://chingchuan-chen.github.io/pflm/

**pflm** is a Python package for functional data analysis and partial
functional linear modelling with L1/L2 regularisation via ADMM
(Alternating Direction Method of Multipliers).

It is inspired by the MATLAB **PACE** toolkit and the R package
**fdapace**, and aims to bring equivalent capabilities to Python users
with a familiar `scikit-learn <https://scikit-learn.org/stable/>`_
estimator interface (``fit`` / ``predict``).

Key Features
============

- **Functional PCA** — mean / covariance smoothing, eigen decomposition,
  score estimation (conditional-expectation or numerical integration),
  with automatic PC-selection via FVE, AIC, or BIC.
- **Partial Functional Linear Model** — combine scalar and functional
  predictors (via FPCA scores) in a single regularised regression.
- **ElasticNet via ADMM** — elastic-net solver supporting Gaussian,
  Binomial, Poisson, Gamma, Tweedie, and Multinomial families.
- **Local polynomial smoothing** — 1D and 2D kernel regression with
  GCV / CV bandwidth selection (``Polyfit1DModel``, ``Polyfit2DModel``).
- **Fast interpolation** — Cython-backed 1D / 2D linear and cubic-spline
  interpolation (``interp1d``, ``interp2d``).
- **Synthetic data generation** —
  ``FunctionalDataGenerator`` for reproducible simulation studies.

Package Structure
=================

.. code-block:: text

   pflm.fpca          FPCA, data generator, result containers
   pflm.smooth        Local polynomial regression (1D / 2D)
   pflm.interp        Fast 1D / 2D interpolation
   pflm.pflm          PartialFunctionalLinearModel, FPCAConfig
   pflm.pflm.utils    ElasticNet, LinearModelFamily
   pflm.utils         FlattenFunctionalData, trapz, helpers

Documentation
=============

Full documentation (API reference, user guide, examples):
https://chingchuan-chen.github.io/pflm/

Installation
============

From GitHub (recommended)
-------------------------

.. code-block:: shell

   pip install git+https://github.com/ChingChuan-Chen/pflm.git

pip will build the Cython extensions automatically via meson-python.

Requirements: Python ≥ 3.10, a C/C++ compiler, ``numpy ≥ 1.22``,
``scipy ≥ 1.14``, ``scikit-learn ≥ 1.5``.

Development install
-------------------

.. code-block:: shell

   git clone https://github.com/ChingChuan-Chen/pflm.git
   cd pflm
   pip install "meson-python>=0.17.1" "meson>=1.3" "ninja" "Cython>=3.0.10"
   pip install -e .

Quick Start
===========

Functional PCA
--------------

.. code-block:: python

   import numpy as np
   from scipy.special import j0
   from pflm.fpca import FunctionalDataGenerator, FunctionalPCA

   # Regular time grid
   t = np.linspace(0.0, 10.0, 201)

   # Generate 50 synthetic functional curves
   gen = FunctionalDataGenerator(
       t,
       lambda tt: np.sin(tt) * 0.5,
       lambda tt: 1.0 + 0.2 * np.cos(tt),
       j0,
       variation_prop_thresh=0.99,
   )
   y_list, t_list = gen.generate(n=50, seed=42)

   # Fit FPCA
   fpca = FunctionalPCA(num_points_reg_grid=101)
   fpca.fit(t_list, y_list)

   print("Selected #PCs:", fpca.num_pcs_)
   print("Eigenvalues:", fpca.fpca_model_params_.fpca_lambda)
   print("Score matrix shape:", fpca.xi_.shape)

Partial Functional Linear Model
---------------------------------

.. code-block:: python

   from pflm.pflm.partial_flm import PartialFunctionalLinearModel

   rng = np.random.default_rng(0)
   scalar = rng.standard_normal((50, 3))
   response = scalar @ np.array([1.0, -0.5, 0.3]) + rng.normal(0, 0.2, 50)

   model = PartialFunctionalLinearModel()
   model.fit([t_list], [y_list], scalar, response)
   print("Coefficients:", model.linear_model_.coef_)

ElasticNet
----------

.. code-block:: python

   from pflm.pflm.utils import ElasticNet

   X = rng.standard_normal((200, 5))
   y = X @ np.array([1.5, -2.0, 0.0, 0.0, 3.0]) + rng.normal(0, 0.5, 200)

   reg = ElasticNet(alpha=0.1, l1_ratio=0.5).fit(X, y)
   print("Coefficients:", reg.coef_)

1D Smoothing
------------

.. code-block:: python

   from pflm.smooth import Polyfit1DModel

   X = np.linspace(0, 1, 200)
   y = np.sin(2 * np.pi * X) + rng.normal(0, 0.1, 200)

   model = Polyfit1DModel().fit(X, y)
   print("Bandwidth:", model.bandwidth_)
   yq = model.predict(np.linspace(0, 1, 50))

Contributing
============

pflm is under active development. Contributions, bug reports, and
feature requests are welcome — please open an issue or pull request on
`GitHub <https://github.com/ChingChuan-Chen/pflm>`_.

License
=======

MIT — see `LICENSE <https://github.com/ChingChuan-Chen/pflm/blob/main/LICENSE>`_.

Reference
=========

1. fdapace: Functional Data Analysis and Empirical Dynamics,
   https://cran.r-project.org/web/packages/fdapace/index.html.
2. PACE: Principal Analysis by Conditional Expectation,
   https://www.stat.ucdavis.edu/PACE/
