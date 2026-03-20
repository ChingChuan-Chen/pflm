2D smoothing with Polyfit2DModel
================================

Fit a 2D local polynomial model with kernel smoothing.

.. code-block:: python

   import numpy as np
   from pflm.smooth import Polyfit2DModel, KernelType

   rng = np.random.default_rng(0)
   x1 = np.linspace(-1, 1, 40)
   x2 = np.linspace(-1, 1, 40)
   X1, X2 = np.meshgrid(x1, x2, indexing="ij")
   Z = np.sin(np.pi*X1) * np.cos(np.pi*X2) + 0.1*rng.standard_normal(X1.shape)

   X = np.column_stack([X1.ravel(), X2.ravel()])
   y = Z.ravel()

   model = Polyfit2DModel(kernel_type=KernelType.GAUSSIAN, degree=1, interp_kind="linear")
   model.fit(X, y, bandwidth_selection_method="gcv", num_points_reg_grid=80)

   # Predict on a finer grid
   X1q = np.linspace(-1, 1, 100)
   X2q = np.linspace(-1, 1, 100)
   Zq = model.predict(X1q, X2q)