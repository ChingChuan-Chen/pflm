Quick Start
===========

A minimal FPCA workflow.

.. code-block:: python

   import numpy as np
   from scipy.special import j0
   from pflm.fpca import FunctionalDataGenerator, FunctionalPCA

   # Regular time grid
   t = np.linspace(0.0, 10.0, 201)
   mean_func = lambda tt: np.sin(tt) * 0.5
   var_func = lambda tt: 1.0 + 0.2 * np.cos(tt)

   # Generate synthetic functional data
   gen = FunctionalDataGenerator(t, mean_func, var_func, j0, variation_prop_thresh=0.99)
   y_list, t_list = gen.generate(n=20, seed=0)

   # Fit FPCA
   fpca = FunctionalPCA(num_points_reg_grid=101)
   fpca.fit(t_list, y_list)
   print("Selected #PCs:", fpca.fpca_model_params_.num_pcs)