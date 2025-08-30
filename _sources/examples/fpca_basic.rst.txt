FPCA: basic workflow
====================

This example demonstrates a minimal FPCA workflow.

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