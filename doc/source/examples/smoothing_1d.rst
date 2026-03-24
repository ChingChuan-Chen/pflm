1D smoothing with Polyfit1DModel
================================

``Polyfit1DModel`` performs local polynomial regression with a selectable kernel.  Bandwidth can be chosen
automatically via GCV or CV.  After fitting, predictions are computed efficiently through internal interpolation on
a regular grid.

Basic usage
-----------

.. code-block:: python

   import numpy as np
   from pflm.smooth import Polyfit1DModel, KernelType

   rng = np.random.default_rng(0)
   X = np.linspace(0, 1, 200)
   y_true = np.sin(2 * np.pi * X)
   y = y_true + rng.normal(0, 0.1, size=X.shape)

   model = Polyfit1DModel(
       kernel_type=KernelType.GAUSSIAN,
       degree=1,
       deriv=0,
       interp_kind="spline",
   )
   model.fit(X, y, bandwidth_selection_method="gcv", num_points_reg_grid=150)
   print("Selected bandwidth:", model.bandwidth_)

   Xq = np.linspace(0, 1, 100)
   yq = model.predict(Xq)  # uses internal interpolation
   print("Max error:", np.max(np.abs(yq - np.sin(2 * np.pi * Xq))))

Kernel and bandwidth options
----------------------------

The ``kernel_type`` parameter supports several kernels.  You can also provide a fixed bandwidth or use CV with a
custom candidate grid.

.. code-block:: python

   # Epanechnikov kernel with cross-validation
   model_cv = Polyfit1DModel(
       kernel_type=KernelType.EPANECHNIKOV,
       degree=1,
       interp_kind="linear",
       random_seed=42,
   )
   model_cv.fit(
       X, y,
       bandwidth_selection_method="cv",
       cv_folds=5,
       num_bw_candidates=30,
   )
   print("CV bandwidth:", model_cv.bandwidth_)

   # Or supply a known bandwidth
   model_fixed = Polyfit1DModel(kernel_type=KernelType.GAUSSIAN)
   model_fixed.fit(X, y, bandwidth=0.08)
   print("Fixed bandwidth:", model_fixed.bandwidth_)

Weighted observations
---------------------

.. code-block:: python

   weights = np.ones_like(X)
   weights[:50] = 2.0  # up-weight the first 50 points
   model.fit(X, y, sample_weight=weights, bandwidth_selection_method="gcv")