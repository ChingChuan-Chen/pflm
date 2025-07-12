"""Polynomial fitting functions for pflm."""

# Authors: Ching-Chuan Chen
# SPDX-License-Identifier: MIT

import numpy as np

from pflm.smooth._polyfit import polyfit1d_f32, polyfit1d_f64, polyfit2d_f32, polyfit2d_f64
from pflm.smooth.kernel import KernelType


def polyfit1d(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    x_new: np.ndarray,
    bandwidth: float,
    kernel_type: KernelType = KernelType.GAUSSIAN,
    degree: int = 1,
    deriv: int = 0,
) -> np.ndarray:
    """Perform local polynomial regression on 1D data.
    Parameters
    ----------
    x : np.ndarray
        1D array of x-coordinates of the data points.
    y : np.ndarray
        1D array of y-coordinates of the data points.
    w : np.ndarray
        1D array of weights for the data points.
    x_new : np.ndarray
        1D array of x-coordinates where the polynomial should be evaluated.
    bandwidth : float
        The bandwidth for the local polynomial regression.
    kernel_type : KernelType, optional
        The kernel type to use for weighting the data points. Default is KernelType.GAUSSIAN.
    degree : int, optional
        The degree of the polynomial to fit. Default is 1.
    deriv : int, optional
        The order of the derivative to compute. Default is 0 (no derivative).

    Returns
    -------
    np.ndarray
        The value of the polynomial at the specified x_new coordinate.
    """
    if x.ndim != 1:
        raise ValueError("x must be a 1D array.")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array.")
    if w.ndim != 1:
        raise ValueError("w must be a 1D array.")
    if x.size != y.size:
        raise ValueError("y must have the same size as x.")
    if x.size != w.size:
        raise ValueError("w must have the same size as x.")
    if x_new.ndim != 1:
        raise ValueError("x_new must be a 1D array.")
    if x_new.size == 0:
        raise ValueError("x_new must not be empty.")
    if bandwidth is None or not isinstance(bandwidth, (float, int)):
        raise TypeError("Bandwidth, bandwidth, should be a float or an integer.")
    if np.isnan(bandwidth):
        raise ValueError("Bandwidth, bandwidth, should not be NaN.")
    if bandwidth <= 0:
        raise ValueError("Bandwidth, bandwidth, should be positive.")
    if kernel_type not in KernelType:
        raise ValueError(f"kernel must be one of {list(KernelType)}.")
    if degree is None or not isinstance(degree, int):
        raise TypeError("Degree of polynomial, degree, should be an integer.")
    if deriv is None or not isinstance(deriv, int):
        raise TypeError("Order of derivative, deriv, should be an integer.")
    if degree <= 0:
        raise ValueError("Degree of polynomial, degree, should be positive.")
    if deriv < 0:
        raise ValueError("Order of derivative, deriv, should be positive.")
    if degree < deriv:
        raise ValueError("Degree of polynomial, degree, should be greater than or equal to order of derivative, deriv.")
    if np.isnan(x).any():
        raise ValueError("Input array x contains NaN values.")
    if np.isnan(y).any():
        raise ValueError("Input array y contains NaN values.")
    if np.isnan(w).any():
        raise ValueError("Input array w contains NaN values.")
    if np.isnan(x_new).any():
        raise ValueError("Input array x_new contains NaN values.")
    if np.any(np.diff(x) < 0):
        raise ValueError("x must be sorted in ascending order.")
    if np.any(w < 0):
        raise ValueError("All weights in w must be greater than 0.")
    if np.any(np.diff(x_new) <= 0):
        raise ValueError("x_new must be strictly increasing.")

    polyfit1d_func = polyfit1d_f32 if x.dtype == np.float32 else polyfit1d_f64
    return polyfit1d_func(
        x,
        y.astype(x.dtype, copy=False),
        w.astype(x.dtype, copy=False),
        x_new.astype(x.dtype, copy=False),
        bandwidth,
        kernel_type.value,
        degree,
        deriv,
    )


def polyfit2d(
    x_grid: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    x_new1: np.ndarray,
    x_new2: np.ndarray,
    bandwidth1: float,
    bandwidth2: float,
    kernel_type: KernelType = KernelType.GAUSSIAN,
    degree: int = 1,
    deriv1: int = 0,
    deriv2: int = 0
):
    """Perform local polynomial regression on 2D data.

    Parameters
    ----------
    x_grid : np.ndarray
        2D array of x-coordinates of the data points.
    y : np.ndarray
        1D array of y-coordinates of the data points.
    w : np.ndarray
        1D array of weights for the data points.
    x_new1 : np.ndarray
        1D array of x-coordinates where the polynomial should be evaluated.
    x_new2 : np.ndarray
        1D array of y-coordinates where the polynomial should be evaluated.
    bandwidth1 : float
        The bandwidth for the local polynomial regression in the first dimension.
    bandwidth2 : float
        The bandwidth for the local polynomial regression in the second dimension.
    kernel_type : KernelType, optional
        The kernel type to use for weighting the data points. Default is KernelType.GAUSSIAN.
    degree : int, optional
        The degree of the polynomial to fit. Default is 1.
    deriv1 : int, optional
        The order of the derivative to compute in the first dimension. Default is 0 (no derivative).
    deriv2 : int, optional
        The order of the derivative to compute in the second dimension. Default is 0 (no derivative).

    Returns
    -------
    np.ndarray
        The value of the polynomial at the specified x_new0 and x_new1 coordinates.
    """
    if x_grid.ndim != 2:
        raise ValueError("x_grid must be a 2D array.")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array.")
    if w.ndim != 1:
        raise ValueError("w must be a 1D array.")
    if x_grid.shape[0] != y.size:
        raise ValueError("y must have the same size as the first dimension of x_grid.")
    if y.size != w.size:
        raise ValueError("w must have the same size as y.")
    if x_new1.ndim != 1 or x_new2.ndim != 1:
        raise ValueError("x_new1 and x_new2 must be 1D arrays.")
    if x_new1.size == 0 or x_new2.size == 0:
        raise ValueError("x_new1 and x_new2 must not be empty.")
    if bandwidth1 is None or not isinstance(bandwidth1, (float, int)):
        raise TypeError("Bandwidth, bandwidth1, should not be None and must be a float or int.")
    if bandwidth2 is None or not isinstance(bandwidth2, (float, int)):
        raise TypeError("Bandwidth, bandwidth2, should not be None and must be a float or int.")
    if np.isnan(bandwidth1) or np.isnan(bandwidth2):
        raise ValueError("Bandwidths, bandwidth1 and bandwidth2, should not be NaN.")
    if bandwidth1 <= 0 or bandwidth2 <= 0:
        raise ValueError("Bandwidths, bandwidth1 and bandwidth2, should be positive.")
    if kernel_type not in KernelType:
        raise ValueError(f"kernel must be one of {list(KernelType)}.")
    if degree is None or not isinstance(degree, int):
        raise TypeError("Degree of polynomial, degree, should not be None and must be an integer.")
    if deriv1 is None or not isinstance(deriv1, int):
        raise TypeError("Order of derivative, deriv1, should not be None and must be an integer.")
    if deriv2 is None or not isinstance(deriv2, int):
        raise TypeError("Order of derivative, deriv2, should not be None and must be an integer.")
    if degree <= 0:
        raise ValueError("Degree of polynomial, degree, should be positive.")
    if deriv1 < 0:
        raise ValueError("Order of derivative, deriv1, should be positive.")
    if deriv2 < 0:
        raise ValueError("Order of derivative, deriv2, should be positive.")
    if degree < deriv1 + deriv2:
        raise ValueError("Degree of polynomial, degree, should be greater than or equal to the sum of orders of derivatives, deriv1 and deriv2.")
    if np.isnan(x_grid).any():
        raise ValueError("Input array x_grid contains NaN values.")
    if np.isnan(y).any():
        raise ValueError("Input array y contains NaN values.")
    if np.isnan(w).any():
        raise ValueError("Input array w contains NaN values.")
    if np.isnan(x_new1).any():
        raise ValueError("Input array x_new1 contains NaN values.")
    if np.isnan(x_new2).any():
        raise ValueError("Input array x_new2 contains NaN values.")
    if np.any(w < 0):
        raise ValueError("All weights in w must be greater than 0.")

    polyfit2d_func = polyfit2d_f32 if x_grid.dtype == np.float32 else polyfit2d_f64
    if kernel_type.value >= 100:
        # For kernel that don't have support |u| <= 1, we need to ensure that x_grid is sorted in the first dimension.
        ord = np.lexsort((x_grid[:, 1], x_grid[:, 0]))
        # We need to convert x_grid to a contiguous array with shape (2, n) for Cython compatibility.
        x_grid_sorted = np.ascontiguousarray(x_grid[ord, :].T)
        y_sorted = y[ord].astype(x_grid_sorted.dtype, copy=False)
        w_sorted = w[ord].astype(x_grid_sorted.dtype, copy=False)
        return polyfit2d_func(
            x_grid_sorted,
            y_sorted,
            w_sorted,
            x_new1.astype(x_grid_sorted.dtype, copy=False),
            x_new2.astype(x_grid_sorted.dtype, copy=False),
            bandwidth1,
            bandwidth2,
            kernel_type.value,
            degree,
            deriv1,
            deriv2
        )
    else:
        # We need to convert x_grid to a contiguous array with shape (2, n) for Cython compatibility.
        x_grid = np.ascontiguousarray(x_grid.T)
        return polyfit2d_func(
            x_grid,
            y.astype(x_grid.dtype, copy=False),
            w.astype(x_grid.dtype, copy=False),
            x_new1.astype(x_grid.dtype, copy=False),
            x_new2.astype(x_grid.dtype, copy=False),
            bandwidth1,
            bandwidth2,
            kernel_type.value,
            degree,
            deriv1,
            deriv2
        )
