Access pflm C-API
=================

Available extensions
-------------------------------

- ``pflm.smooth._polyfit``: low-level kernels and polynomial fitters
  - ``polyfit1d_f32``, ``polyfit1d_f64``
  - ``polyfit2d_f32``, ``polyfit2d_f64``
  - ``calculate_kernel_value_f32``, ``calculate_kernel_value_f64``
- ``pflm.interp._interp``: internal interpolation routines
- ``pflm.utils._trapz``, ``pflm.utils._lapack_helper``: numeric helpers

Example: call a 1D fitter directly
----------------------------------

.. code-block:: python

   import numpy as np
   from pflm.smooth._polyfit import polyfit1d_f64

   x = np.linspace(0, 1, 21, dtype=np.float64)
   y = np.sin(2*np.pi*x).astype(np.float64)
   w = np.ones_like(y)
   reg_grid = np.linspace(0, 1, 51, dtype=np.float64)

   kernel_type = 0  # GAUSSIAN in this project
   degree = 1
   deriv = 0
   bandwidth = 0.1

   y_fit = polyfit1d_f64(x, y, w, reg_grid, bandwidth, kernel_type, degree, deriv)

Notes
-----

- Inputs must be contiguous and use the exact dtype (``float32`` for ``*_f32`` and ``float64`` for ``*_f64``).
- ABI can vary by platform; prefer the high-level estimators for stability.