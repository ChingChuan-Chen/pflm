Quick Start
===========

This page walks through the core pflm modules with minimal examples.

Functional PCA in 30 seconds
-----------------------------

Generate synthetic functional data and run FPCA:

.. code-block:: python

   import numpy as np
   from scipy.special import j0
   from pflm.fpca import FunctionalDataGenerator, FunctionalPCA

   # Regular time grid
   t = np.linspace(0.0, 10.0, 201)
   mean_func = lambda tt: np.sin(tt) * 0.5
   var_func  = lambda tt: 1.0 + 0.2 * np.cos(tt)

   # Generate synthetic functional data
   gen = FunctionalDataGenerator(t, mean_func, var_func, j0, variation_prop_thresh=0.99)
   y_list, t_list = gen.generate(n=20, seed=0)

   # Fit FPCA
   fpca = FunctionalPCA(num_points_reg_grid=101)
   fpca.fit(t_list, y_list)
   print("Selected #PCs:", fpca.num_pcs_)
   print("Score matrix shape:", fpca.xi_.shape)

1D smoothing in 30 seconds
--------------------------

.. code-block:: python

   from pflm.smooth import Polyfit1DModel

   rng = np.random.default_rng(0)
   X = np.linspace(0, 1, 200)
   y = np.sin(2 * np.pi * X) + rng.normal(0, 0.1, 200)

   model = Polyfit1DModel().fit(X, y)
   print("Bandwidth:", model.bandwidth_)
   yq = model.predict(np.linspace(0, 1, 50))

ElasticNet in 30 seconds
------------------------

.. code-block:: python

   from pflm.pflm.utils import ElasticNet

   rng = np.random.default_rng(0)
   X = rng.standard_normal((100, 5))
   y = X @ np.array([1.0, -2.0, 0.0, 0.0, 3.0]) + rng.normal(0, 0.5, 100)

   reg = ElasticNet(alpha=0.1, l1_ratio=0.5).fit(X, y)
   print("Coefficients:", reg.coef_)

Next steps
----------

- See :doc:`../examples/index` for detailed worked examples.
- Consult the :doc:`../reference/index` for the full API.