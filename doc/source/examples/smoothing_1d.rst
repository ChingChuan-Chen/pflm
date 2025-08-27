1D smoothing with Polyfit1DModel
================================

Fit a 1D local polynomial model with kernel smoothing and interpolate.

.. code-block:: python

   import numpy as np
   from pflm.smooth import Polyfit1DModel, KernelType

   rng = np.random.default_rng(0)
   X = np.linspace(0, 1, 200)
   y_true = np.sin(2*np.pi*X)
   y = y_true + rng.normal(0, 0.1, size=X.shape)

   model = Polyfit1DModel(kernel_type=KernelType.GAUSSIAN, degree=1, deriv=0, interp_kind="spline")
   model.fit(X, y, bandwidth_selection_method="gcv", num_points_reg_grid=150)

   Xq = np.linspace(0, 1, 100)
   yq = model.predict(Xq)  # uses internal interpolation