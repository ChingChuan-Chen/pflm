Interpolation 1D
================

Interpolation 1D:

.. code-block:: python

   import numpy as np
   from pflm.interp import interp1d

   x = np.linspace(0, 1, 11)
   y = np.sin(2*np.pi*x)
   xq = np.linspace(0, 1, 51)
   yq = interp1d(x, y, xq, method="spline")
