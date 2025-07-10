"""Compute the integrated area value with trapezoidal rule."""

# Authors: Ching-Chuan Chen
# SPDX-License-Identifier: MIT
import numpy as np


def trapz(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Compute the integrated area value with trapezoidal rule.

    Parameters
    ----------
    y : array_like
        1D or 2D array of function values.
    x : array_like
        1D array of x-coordinates corresponding to the function values.

    Returns
    -------
    trapz_value : np.ndarray
        The integrated area value computed using the trapezoidal rule.
    """
    if len(y.shape) == 1:
        if y.shape[0] != x.shape[0]:
            raise ValueError("y and x must have the same length.")
        return np.dot((y[:-1] + y[1:]), np.diff(x)) * 0.5
    elif len(y.shape) == 2:
        if y.shape[1] != x.shape[0]:
            raise ValueError("The number of columns of y must match the size of x.")
        return np.matmul((y[:, :-1] + y[:, 1:]), np.diff(x)) * 0.5
    else:
        raise ValueError("y must be 1D or 2D.")
