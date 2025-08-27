Interpolation 2D
=====================

Interpolation 2D:

.. code-block:: python

   import numpy as np
   from pflm.interp import interp2d

   x1 = np.linspace(-1, 1, 21)
   x2 = np.linspace(-1, 1, 21)
   X1, X2 = np.meshgrid(x1, x2, indexing="ij")
   Z = np.sin(np.pi*X1)*np.cos(np.pi*X2)
   x1q = np.linspace(-1, 1, 51)
   x2q = np.linspace(-1, 1, 51)
   Zq = interp2d(x1, x2, Z, x1q, x2q, method="spline")