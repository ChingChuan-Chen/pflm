Interpolation 2D
================

``interp2d`` interpolates 2D gridded data.  Input is a pair of 1D coordinate vectors and a 2D value matrix; the output
is the interpolated surface on the query grid.

Bilinear interpolation
----------------------

.. code-block:: python

   import numpy as np
   from pflm.interp import interp2d

   x1 = np.linspace(-1, 1, 21)
   x2 = np.linspace(-1, 1, 21)
   X1, X2 = np.meshgrid(x1, x2, indexing="ij")
   Z = np.sin(np.pi * X1) * np.cos(np.pi * X2)

   x1q = np.linspace(-1, 1, 51)
   x2q = np.linspace(-1, 1, 51)
   Zq = interp2d(x1, x2, Z, x1q, x2q, method="linear")
   print("Output shape:", Zq.shape)  # (51, 51)

Bicubic spline interpolation
----------------------------

For smoother surfaces, switch to spline.

.. code-block:: python

   Zq_spline = interp2d(x1, x2, Z, x1q, x2q, method="spline")

   # Ground truth on query grid
   X1q, X2q = np.meshgrid(x1q, x2q, indexing="ij")
   Z_true = np.sin(np.pi * X1q) * np.cos(np.pi * X2q)
   print("Linear  max-err:", np.max(np.abs(Zq - Z_true)))
   print("Spline  max-err:", np.max(np.abs(Zq_spline - Z_true)))