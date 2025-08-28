"""Interpolation on 1D and 2D data.

This module provides fast linear and spline interpolation for 1D/2D arrays.
"""

# Authors: Ching-Chuan Chen
# SPDX-License-Identifier: MIT

import numpy as np

from pflm.interp.interp import interp1d_f32, interp1d_f64, interp2d_f32, interp2d_f64


def interp1d(x: np.ndarray, y: np.ndarray, x_new: np.ndarray, method: str = "linear") -> np.ndarray:
    """Interpolate 1D data using linear or spline interpolation.

    Parameters
    ----------
    x : np.ndarray of shape (n,)
        Strictly 1D input coordinates. Duplicates are allowed but will be
        reduced to the first occurrence internally.
    y : np.ndarray of shape (n,)
        Values at `x`. Must match `x` in length.
    x_new : np.ndarray of shape (m,)
        Query points.
    method : {"linear", "spline"}, default="linear"
        Interpolation method.

    Returns
    -------
    y_new : np.ndarray of shape (m,)
        Interpolated values at `x_new`. The dtype follows `x`:
        float32 uses the f32 backend; otherwise f64.

    Raises
    ------
    ValueError
        If any input is not 1D, is empty, sizes mismatch, contains NaN, or
        `method` is invalid.

    Notes
    -----
    - Input duplicates in `x` are deduplicated using the first occurrence.
    - Backend is selected by dtype of `x` (float32 -> f32; otherwise f64).

    See Also
    --------
    interp2d : Interpolate 2D gridded data.
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


def interp2d(
    x: np.ndarray,
    y: np.ndarray,
    v: np.ndarray,
    x_new: np.ndarray,
    y_new: np.ndarray,
    method: str = "linear",
) -> np.ndarray:
    """Interpolate 2D gridded data using linear or spline interpolation.

    Parameters
    ----------
    x : np.ndarray of shape (n_x,)
        X-coordinates of the grid (first axis of `v`).
    y : np.ndarray of shape (n_y,)
        Y-coordinates of the grid (second axis of `v`).
    v : np.ndarray of shape (n_x, n_y)
        Values on the grid defined by (`x`, `y`).
    x_new : np.ndarray of shape (m_x,)
        Query x-coordinates.
    y_new : np.ndarray of shape (m_y,)
        Query y-coordinates.
    method : {"linear", "spline"}, default="linear"
        Interpolation method.

    Returns
    -------
    v_new : np.ndarray of shape (m_x, m_y)
        Interpolated values evaluated on the mesh defined by (`x_new`, `y_new`).
        The dtype follows `x` (float32 -> f32 backend; otherwise f64).

    Raises
    ------
    ValueError
        If input dimensionality is invalid, arrays are empty, shape of `v` does
        not match (`x`, `y`), any input contains NaN, or `method` is invalid.

    Notes
    -----
    - Duplicates in `x` and `y` are deduplicated using the first occurrence.
    - The underlying C++ implementation expects `v` in Fortran-like layout
      with axes swapped; this wrapper transposes/contiguates as needed.
    - Backend is selected by dtype of `x` (float32 -> f32; otherwise f64).

    See Also
    --------
    interp1d : Interpolate 1D data using linear or spline interpolation.
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
