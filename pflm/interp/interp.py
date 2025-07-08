import numpy as np
from pflm.interp._interp import interp1d_f32, interp1d_f64, interp2d_f32, interp2d_f64

"""
pflm.interp.interp
==================

This module provides functions for 1D and 2D interpolation of data using linear and spline methods.

Functions
---------
- `interp1d`: Interpolate 1D data using linear or spline interpolation.
- `interp2d`: Interpolate 2D data using linear or spline interpolation.
"""

def interp1d(x, y, x_new, method='linear'):
    """Interpolate 1D data using linear or spline interpolation.

    Parameters
    ----------
    x (np.ndarray): 1D array of x-coordinates of the data points.
    y (np.ndarray): 1D array of y-coordinates of the data points.
    x_new (np.ndarray): 1D array of x-coordinates where the interpolation is evaluated.
    method (str, optional): The interpolation method.
        The interpolation method:
        'linear' for linear interpolation,
        'spline' for spline interpolation.

    Returns
    -------
       np.ndarray: The interpolated values at the new x-coordinates.
    """
    if x.ndim != 1 or y.ndim != 1 or x_new.ndim != 1:
        raise ValueError("x, y, and x_new must be 1-dimensional arrays.")
    if x.size == 0 or y.size == 0 or x_new.size == 0:
        raise ValueError("x, y, and x_new must not be empty.")
    if x.size != y.size:
        raise ValueError("x must have the same size as y.")
    if method not in ['linear', 'spline']:
        raise ValueError("Invalid method. Use 'linear' or 'spline'.")
    method_mapping = {'linear': 0, 'spline': 1}
    interp_func = interp1d_f32 if x.dtype == np.float32 else interp1d_f64
    input_type = x.dtype
    x_unique, idx = np.unique(x, return_index=True)
    y_unique = y[idx].astype(input_type, copy=False)
    x_new_typed = x_new.astype(input_type, copy=False)
    return interp_func(x_unique, y_unique, x_new_typed, method_mapping[method])

def interp2d(x, y, v, x_new, y_new, method='linear'):
    """
    Interpolate 2D data using linear or spline interpolation.

    Parameters
    ----------
    x (np.ndarray): 1D array of x-coordinates of the data points.
    y (np.ndarray): 1D array of y-coordinates of the data points.
    v (np.ndarray): 2D array of values at the grid points defined by x and y.
    x_new (np.ndarray): 1D array of x-coordinates where the interpolation is evaluated.
    y_new (np.ndarray): 1D array of y-coordinates where the interpolation is evaluated.
    method (str, optional): The interpolation method.
        The interpolation method:
        'linear' for linear interpolation,
        'spline' for spline interpolation.

    Returns
    -------
    np.ndarray: The interpolated values at the grid points defined by x_new and y_new.
    """
    if x.ndim != 1 or y.ndim != 1 or v.ndim != 2 or x_new.ndim != 1 or y_new.ndim != 1:
        raise ValueError("x, y, and v must be 1D and 2D arrays respectively.")
    if x.size == 0 or y.size == 0 or v.size == 0 or x_new.size == 0 or y_new.size == 0:
        raise ValueError("x, y, v, x_new, and y_new must not be empty.")
    if x.size != v.shape[1]:
        raise ValueError("x must have the same length as the first dimension of v")
    if y.size != v.shape[0]:
        raise ValueError("y must have the same length as the second dimension of v")
    if method not in ['linear', 'spline']:
        raise ValueError("Invalid method. Use 'linear' or 'spline'.")
    method_mapping = {'linear': 0, 'spline': 1}
    interp_func = interp2d_f32 if x.dtype == np.float32 else interp2d_f64
    input_type = x.dtype
    x_unique, idx_x = np.unique(x, return_index=True)
    y_unique, idx_y = np.unique(y, return_index=True)
    y_unique_typed = y_unique.astype(input_type, copy=False)
    v_unique = v[np.ix_(idx_y, idx_x)].astype(input_type, copy=False)
    x_new_typed = x_new.astype(input_type, copy=False)
    y_new_typed = y_new.astype(input_type, copy=False)
    return interp_func(x_unique, y_unique_typed, v_unique, x_new_typed, y_new_typed, method_mapping[method])
