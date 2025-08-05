"""Interpolation on 1D and 2D data."""

# Authors: Ching-Chuan Chen
# SPDX-License-Identifier: MIT

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


def interp1d(x: np.ndarray, y: np.ndarray, x_new: np.ndarray, method: str = "linear") -> np.ndarray:
    """Interpolate 1D data using linear or spline interpolation.

    This function is aligned with MATLAB's `interp1` function and Python's `scipy.interpolate.interp1d`.
    It returns a 1D array of interpolated values at the new x-coordinates `x_new`.
    The input arrays `x` and `y` must be 1D arrays of the same length, where `x` contains the x-coordinates
    and `y` contains the corresponding y-coordinates. The interpolation method can be either 'linear' or 'spline'.

    Parameters
    ----------
    x : np.ndarray
        1D array of x-coordinates of the data points.
    y : np.ndarray
        1D array of y-coordinates of the data points.
    x_new : np.ndarray
        1D array of x-coordinates where the interpolation is evaluated.
    method : str, optional
        The interpolation method. Default is 'linear'.
        The interpolation method:
        'linear' for linear interpolation,
        'spline' for spline interpolation.
    Returns
    -------
    y_new : np.ndarray
        1D array of interpolated values at the new x-coordinates.

    See Also
    --------
    interp2d: Interpolate 2D data using linear or spline interpolation.
    """
    if x.ndim != 1 or y.ndim != 1 or x_new.ndim != 1:
        raise ValueError("x, y, and x_new must be 1-dimensional arrays.")
    if x.size == 0 or y.size == 0 or x_new.size == 0:
        raise ValueError("x, y, and x_new must not be empty.")
    if x.size != y.size:
        raise ValueError("x must have the same size as y.")
    # NaN check
    if np.isnan(x).any():
        raise ValueError("Input array x contains NaN values.")
    if np.isnan(y).any():
        raise ValueError("Input array y contains NaN values.")
    if np.isnan(x_new).any():
        raise ValueError("Input array x_new contains NaN values.")
    if method not in ["linear", "spline"]:
        raise ValueError("Invalid method. Use 'linear' or 'spline'.")
    method_mapping = {"linear": 0, "spline": 1}
    interp_func = interp1d_f32 if x.dtype == np.float32 else interp1d_f64
    x_unique, idx = np.unique(x, return_index=True)
    y_unique = y[idx].astype(x.dtype, copy=False)
    return interp_func(x_unique, y_unique, x_new.astype(x.dtype, copy=False), method_mapping[method])


def interp2d(x: np.ndarray, y: np.ndarray, v: np.ndarray, x_new: np.ndarray, y_new: np.ndarray, method: str = "linear") -> np.ndarray:
    """Interpolate 2D data using linear or spline interpolation.

    MATLAB's `interp2` function returns a 2D array of interpolated values at the grid points defined by `x_new` and `y_new`.
    The output array `v_new` has the shape (n_new_x, n_new_y), where n_new_x is the length of `x_new` and n_new_y is the length of `y_new`.
    This function is aligned with the Python `scipy.interpolate.interp2d` function.
    The shape of `v` should be (n_x, n_y), where n_x is the length of `x` and n_y is the length of `y`.
    The shape of output `v_new` will be (n_new_x, n_new_y).

    Parameters
    ----------
    x : array_like
        1D array of x-coordinates of the data points with shape (n_x,).
    y : array_like
        1D array of y-coordinates of the data points with shape (n_y,).
    v : array_like
        2D array of values at the grid points defined by x and y with shape (n_x, n_y).
    x_new : array_like
        1D array of x-coordinates where the interpolation is evaluated with shape (n_new_x,).
    y_new : array_like
        1D array of y-coordinates where the interpolation is evaluated with shape (n_new_y,).
    method : str, optional
        The interpolation method. Default is 'linear'.
        The interpolation method:
        'linear' for linear interpolation,
        'spline' for spline interpolation.

    Returns
    -------
    v_new: array_like
        The shape (n_new_x, n_new_y) 2D array of interpolated values at the grid points defined by x_new and y_new.

    See Also
    --------
    interp1d: Interpolate 1D data using linear or spline interpolation.
    """
    if x.ndim != 1 or y.ndim != 1 or v.ndim != 2 or x_new.ndim != 1 or y_new.ndim != 1:
        raise ValueError("x, y, and v must be 1D and 2D arrays respectively.")
    if x.size == 0 or y.size == 0 or v.size == 0 or x_new.size == 0 or y_new.size == 0:
        raise ValueError("x, y, v, x_new, and y_new must not be empty.")
    if x.size != v.shape[0]:
        raise ValueError("x must have the same length as the first dimension of v")
    if y.size != v.shape[1]:
        raise ValueError("y must have the same length as the second dimension of v")
    # NaN check
    if np.isnan(x).any():
        raise ValueError("Input array x contains NaN values.")
    if np.isnan(y).any():
        raise ValueError("Input array y contains NaN values.")
    if np.isnan(v).any():
        raise ValueError("Input array v contains NaN values.")
    if np.isnan(x_new).any():
        raise ValueError("Input array x_new contains NaN values.")
    if np.isnan(y_new).any():
        raise ValueError("Input array y_new contains NaN values.")
    if method not in ["linear", "spline"]:
        raise ValueError("Invalid method. Use 'linear' or 'spline'.")
    method_mapping = {"linear": 0, "spline": 1}
    interp_func = interp2d_f32 if x.dtype == np.float32 else interp2d_f64
    x_unique, idx_x = np.unique(x, return_index=True)
    y_unique, idx_y = np.unique(y.astype(x.dtype, copy=False), return_index=True)
    # C++ Function takes x as the second dimension and y as the first dimension for v
    v_unique = np.ascontiguousarray(v[np.ix_(idx_x, idx_y)].astype(x.dtype, copy=False).T)
    v_new = interp_func(
        x_unique, y_unique, v_unique, x_new.astype(x.dtype, copy=False), y_new.astype(x.dtype, copy=False), method_mapping[method]
    ).T
    return v_new
