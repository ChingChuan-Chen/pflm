pflm
****

**pflm** is a Python package to help users fit partial function linear models with L1/L2 regularization via Alternating Direction Method of Multipliers (ADMM).
**PACE** is a popular package written in MATLAB for functional data analysis, and **fdapace** is an R package which is based on the PACE methodology.
This package is designed to be a Python alternative to these packages, providing similar functionality for users familiar with Python.
It supports various types of functional data analysis, including functional linear models, functional principal component analysis (FPCA), and functional partial least squares (FPLS).
Also, it provides user to use the similar interface as `scikit-learn <https://scikit-learn.org/stable/>`_ for fitting models and making predictions.
Note that this package is not a complete replacement for PACE or fdapace, but rather a complementary tool for users who prefer Python.
And it is under active development, so please feel free to contribute if you have any suggestions or issues.

Installation
============

You can install **pflm** via `pip` command:

.. code-block:: shell

  pip install git+https://github.com:ChingChuan-Chen/pflm.git

Quick Start (Python)
====================

Workload Overview
-----------------

- Data generator: ``FunctionalDataGenerator`` (pflm)
- Replications: 1,000 (repeat runs to stabilize timing)
- Sample size (per replication): ``n = 300``
- Observation grid: ``t = np.linspace(0, 1, 100)``
- Mean function: ``mean(t) = sin(2πt)``
- Variance function: ``var(t) = 0.1``
- Correlation function: ``corr(s, t) = j0(|s - t|)`` (Bessel J0)
- FPCA retained variation: ``variation_prop_thresh = 0.95``
- Observation noise variance: ``error_var = 1``

Code Example
------------

.. code-block:: python

   import numpy as np
   from scipy.special import j0
   from pflm.fpca import FunctionalDataGenerator, FunctionalPCA

   # 1) Prepare a regular time grid
   t = np.linspace(0.0, 10.0, 201)

   # 2) Define mean/variance and correlation kernel
   mean_func = lambda tt: np.sin(tt) * 0.5
   var_func = lambda tt: 1.0 + 0.2 * np.cos(tt)
   corr_func = j0  # default in FunctionalDataGenerator

   # 3) Generate synthetic functional data
   gen = FunctionalDataGenerator(t, mean_func, var_func, corr_func, variation_prop_thresh=0.99)
   y_list, t_list = gen.generate(n=50, seed=42)

   # 4) Fit FPCA
   fpca = FunctionalPCA(assume_measurement_error=True, num_points_reg_grid=101)
   fpca.fit(t_list, y_list)

   # 5) Access results
   print("Selected #PCs:", fpca.fpca_model_params_.num_pcs)
   print("Eigenvalues:", fpca.fpca_model_params_.fpca_lambda)


Reference
=========

1. fdapace: Functional Data Analysis and Empirical Dynamics, https://cran.r-project.org/web/packages/fdapace/index.html.
2. PACE: Principal Analysis by Conditional Expectation, https://www.stat.ucdavis.edu/PACE/
