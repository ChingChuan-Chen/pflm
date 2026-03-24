2D smoothing with Polyfit2DModel
================================

``Polyfit2DModel`` extends local polynomial regression to two-dimensional
inputs.  Bandwidths for each dimension are selected independently (or
forced identical with ``same_bandwidth_for_2dim=True``).

Basic usage
-----------

.. code-block:: python

   import numpy as np
   from pflm.smooth import Polyfit2DModel, KernelType

   rng = np.random.default_rng(0)
   x1 = np.linspace(-1, 1, 40)
   x2 = np.linspace(-1, 1, 40)
   X1, X2 = np.meshgrid(x1, x2, indexing="ij")
   Z = np.sin(np.pi * X1) * np.cos(np.pi * X2) + 0.1 * rng.standard_normal(X1.shape)

   X = np.column_stack([X1.ravel(), X2.ravel()])
   y = Z.ravel()

   model = Polyfit2DModel(
       kernel_type=KernelType.GAUSSIAN,
       degree=1,
       interp_kind="linear",
   )
   model.fit(X, y, bandwidth_selection_method="gcv", num_points_reg_grid=80)
   print("Bandwidths:", model.bandwidth1_, model.bandwidth2_)

   # Predict on a finer grid
   X1q = np.linspace(-1, 1, 100)
   X2q = np.linspace(-1, 1, 100)
   Zq = model.predict(X1q, X2q)
   print("Output shape:", Zq.shape)

Shared bandwidth across dimensions
-----------------------------------

When you expect similar smoothness along both axes, force the solver to
pick a single bandwidth.

.. code-block:: python

   model_shared = Polyfit2DModel(kernel_type=KernelType.GAUSSIAN, degree=1)
   model_shared.fit(
       X, y,
       bandwidth_selection_method="gcv",
       same_bandwidth_for_2dim=True,
       num_points_reg_grid=80,
   )
   print("Shared bandwidth:", model_shared.bandwidth1_, "==", model_shared.bandwidth2_)

Fixed bandwidths
----------------

.. code-block:: python

   model_fixed = Polyfit2DModel(kernel_type=KernelType.EPANECHNIKOV, degree=1)
   model_fixed.fit(X, y, bandwidth1=0.2, bandwidth2=0.3, num_points_reg_grid=80)
   print("Fixed bw:", model_fixed.bandwidth1_, model_fixed.bandwidth2_)