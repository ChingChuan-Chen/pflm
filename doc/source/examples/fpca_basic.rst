FPCA: basic workflow
====================

This example demonstrates a minimal FPCA workflow: generate synthetic functional data, run Functional PCA, and
inspect the results.

Generate synthetic functional data
-----------------------------------

``FunctionalDataGenerator`` builds a stationary covariance surface from a mean function, a marginal variance function,
and a correlation kernel, then samples low-rank functional signals with Gaussian noise.

.. code-block:: python

   import numpy as np
   from scipy.special import j0
   from pflm.fpca import FunctionalDataGenerator, FunctionalPCA

   # 1) Prepare a regular time grid
   t = np.linspace(0.0, 10.0, 201)

   # 2) Define mean/variance and correlation kernel
   mean_func = lambda tt: np.sin(tt) * 0.5
   var_func  = lambda tt: 1.0 + 0.2 * np.cos(tt)
   corr_func = j0  # Bessel J0 — the default

   # 3) Generate 50 synthetic curves
   gen = FunctionalDataGenerator(
       t, mean_func, var_func, corr_func,
       variation_prop_thresh=0.99,
   )
   y_list, t_list = gen.generate(n=50, seed=42)
   print(f"Generated {len(y_list)} curves, each with {len(y_list[0])} points")

Fit FPCA
--------

``FunctionalPCA`` follows the scikit-learn estimator API.  Calling ``fit()`` performs mean/covariance smoothing,
eigen decomposition, and PC-score estimation in one go.

.. code-block:: python

   fpca = FunctionalPCA(
       assume_measurement_error=True,
       num_points_reg_grid=101,
   )
   fpca.fit(t_list, y_list)

   # 5) Access results
   print("Selected #PCs:", fpca.num_pcs_)
   print("Eigenvalues:", fpca.fpca_model_params_.fpca_lambda)
   print("Score matrix shape:", fpca.xi_.shape)         # (50, k)
   print("Fitted curves:", len(fpca.fitted_y_))          # 50

Inspect the smoothed mean and covariance
-----------------------------------------

.. code-block:: python

   # Smoothed mean on the regular grid
   reg_result = fpca.smoothed_model_result_reg_
   print("Regular grid size:", reg_result.grid.shape)
   print("Mean (first 5):", reg_result.mu[:5])
   print("Covariance shape:", reg_result.cov.shape)

   # Measurement error variance
   print("σ²:", fpca.fpca_model_params_.measurement_error_variance)

Choose a different selection method
------------------------------------

By default ``method_select_num_pcs='FVE'`` is used.  You can switch to AIC or BIC, or fix the number of PCs directly.

.. code-block:: python

   # Use AIC to choose the number of PCs
   fpca_aic = FunctionalPCA(num_points_reg_grid=101)
   fpca_aic.fit(t_list, y_list, method_select_num_pcs="AIC")
   print("AIC selected #PCs:", fpca_aic.num_pcs_)

   # Fix the number of PCs to 3
   fpca_fixed = FunctionalPCA(num_points_reg_grid=101)
   fpca_fixed.fit(t_list, y_list, method_select_num_pcs=3)
   print("Fixed #PCs:", fpca_fixed.num_pcs_)

Predict scores for new curves
-------------------------------

.. code-block:: python

   # Generate 10 new curves
   y_new, t_new = gen.generate(n=10, seed=99)

   xi_new, xi_var_new, fitted_y_mat_new, fitted_y_new = fpca.predict(y_new, t_new)
   print("New scores shape:", xi_new.shape)        # (10, k)
   print("New fitted curves:", len(fitted_y_new))   # 10

Simulate sparse / irregular data
----------------------------------

``FunctionalDataGenerator.make_missing`` randomly drops observations from each curve to simulate irregular designs.

.. code-block:: python

   y_sparse, t_sparse = FunctionalDataGenerator.make_missing(
       y_list, t_list, missing_number=150, seed=0,
   )
   print(f"Curve 0: {len(t_sparse[0])} points (was {len(t_list[0])})")

   fpca_sparse = FunctionalPCA(num_points_reg_grid=101)
   fpca_sparse.fit(t_sparse, y_sparse, method_pcs="CE")
   print("Sparse #PCs:", fpca_sparse.num_pcs_)
