"""Polynomial fitting functions for pflm."""

# Authors: Ching-Chuan Chen
# SPDX-License-Identifier: MIT

import numpy as np

from pflm.smooth.kernel import KernelType
from pflm.smooth._polyfit import polyfit1d_f64, polyfit1d_f32


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
    if np.any(np.diff(x) < 0):
        raise ValueError("x must be sorted in ascending order.")
    if np.any(w < 0):
        raise ValueError("All weights in w must be greater than 0.")
    if x.size != w.size:
        raise ValueError("w must have the same size as x.")
    if x_new.ndim != 1:
        raise ValueError("x_new must be a 1D array.")
    if x_new.size == 0:
        raise ValueError("x_new must not be empty.")
    if np.any(np.diff(x_new) <= 0):
        raise ValueError("x_new must be strictly increasing.")
    if bandwidth <= 0:
        raise ValueError("Bandwidth, bandwidth, should be positive.")
    if kernel_type not in KernelType:
        raise ValueError(f"kernel must be one of {list(KernelType)}.")
    if degree <= 0:
        raise ValueError("Degree of polynomial, degree, should be positive.")
    if deriv < 0:
        raise ValueError("Order of derivative, deriv, should be positive.")
    if degree < deriv:
        raise ValueError("Degree of polynomial, degree, should be greater than or equal to order of derivative, deriv.")

    polyfit_func = polyfit1d_f32 if x.dtype == np.float32 else polyfit1d_f64
    return polyfit_func(
        x, y.astype(x.dtype, copy=False), w.astype(x.dtype, copy=False), x_new.astype(x.dtype, copy=False),
        bandwidth, kernel_type.value, degree, deriv
    )


def polyfit2d(
    x_grid: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    x_new0: float,
    x_new1: float,
    bandwidth1: float,
    bandwidth2: float,
    kernel_type: KernelType = KernelType.GAUSSIAN,
    order: int = 1,
    deriv: int = 0,
):
    pass
