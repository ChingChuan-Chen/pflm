"""Utility functions for covariance estimation"""

# Authors: Ching-Chuan Chen
# SPDX-License-Identifier: MIT

import numpy as np
from sklearn.utils.validation import check_array

from pflm.smooth import KernelType, Polyfit1DModel
from pflm.utils._raw_cov import get_raw_cov_f32, get_raw_cov_f64
from pflm.utils._rotate_polyfit2d import rotate_polyfit2d_f32, rotate_polyfit2d_f64
from pflm.utils.utility import trapz, FlattenFunctionalData


def get_raw_cov(flatten_func_data: FlattenFunctionalData, mu: np.ndarray) -> np.ndarray:
    """
    Get the dense covariance matrix from the flattened data matrices.

    This function computes the covariance between pairs of samples at different time points,
    using the flattened response values `yy`, corresponding time points `tt`, and weights `ww`.
    It returns a raw covariance matrix with columns representing (sid, t1, t2, w, cov),
    where `sid` is the sample index, `t1` and `t2` are the time points, `w` is the weight,
    and `cov` is the computed covariance value.

    Parameters
    ----------
    flatten_func_data : FlattenFunctionalData
        Flattened functional data containing response values, time points, and weights.
    mu : np.ndarray
        Mean function values at the observation grid.

    Returns
    -------
    raw_cov : np.ndarray
        The raw covariance matrix with columns representing (sid, t1, t2, w, cov).
        The shape is (num_pairs, 5), where num_pairs is the number of unique pairs of samples.
    """
    nt = flatten_func_data.unique_tid.size
    if mu.size != nt:
        raise ValueError("The length of mu must match the number of unique time indices.")

    input_dtype = flatten_func_data.y.dtype
    get_raw_cov_func = get_raw_cov_f64 if input_dtype == np.float64 else get_raw_cov_f32
    raw_cov = get_raw_cov_func(
        flatten_func_data.y, flatten_func_data.t, flatten_func_data.w,
        mu, flatten_func_data.tid, flatten_func_data.unique_sid, flatten_func_data.sid_cnt
    )
    return raw_cov


def get_covariance_matrix(raw_cov: np.ndarray, obs_grid: np.ndarray) -> np.ndarray:
    """Convert the raw covariance matrix to a dense covariance matrix.

    This function takes the raw covariance matrix obtained from `get_raw_cov` and maps it to a dense covariance matrix
    using the observation grid. The resulting matrix is symmetric and contains the covariance values for each pair of time points.
    This function would not check the validity of the input data, so it is assumed that the input is valid.

    Parameters
    ----------
    raw_cov : np.ndarray
        The raw covariance matrix with columns representing (sid, t1, t2, w, cov).
        This input should use `get_raw_cov` to obtain the raw covariance data.
        The shape is (num_pairs, 5), where num_pairs is the number of unique pairs of samples.
    obs_grid : np.ndarray
        The observation grid, which is a 1D array of sorted unique time points.

    Returns
    -------
    np.ndarray
        The dense covariance matrix.
    """
    # calculate the sum of weights and covariance for each unique pair of (t1, t2)
    t_pairs, idx = np.unique(raw_cov[:, [1, 2]], axis=0, return_inverse=True)
    ww_sum = np.bincount(idx, weights=raw_cov[:, 3])
    covariances = np.bincount(idx, weights=raw_cov[:, 3] * raw_cov[:, 4]) / np.array([w - 1.0 if w > 1.0 else 1.0 for w in ww_sum])
    covariances[ww_sum <= 1.0] = 0.0  # set cov to 0 if weight is less than or equal to 1

    # map the pairs to the observation grid
    upper_cov_matrix = np.zeros((obs_grid.size, obs_grid.size), dtype=raw_cov.dtype)
    t1 = np.digitize(t_pairs[:, 0], obs_grid, right=True)
    t2 = np.digitize(t_pairs[:, 1], obs_grid, right=True)

    for t1_idx, t2_idx, cov in zip(t1, t2, covariances):
        upper_cov_matrix[t1_idx, t2_idx] = cov

    # ensure symmetry
    cov_matrix = upper_cov_matrix + upper_cov_matrix.T
    np.fill_diagonal(cov_matrix, cov_matrix.diagonal() / 2.0)
    return cov_matrix


def rotate_polyfit2d(
    x_grid: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    new_grid: np.ndarray,
    bandwidth: float,
    kernel_type: KernelType = KernelType.GAUSSIAN,
) -> np.ndarray:
    """Rotate the 2D polynomial fit to a new grid.

    This function performs a 2D polynomial fit on the input data and rotates it to a new grid.
    It supports different kernel types for the fit.

    Parameters
    ----------
    x_grid : np.ndarray
        The input grid of shape (2, n), where n is the number of points.
    y : np.ndarray
        The response values corresponding to `x_grid`.
    w : np.ndarray
        The weights for each point in `x_grid`.
    new_grid : np.ndarray
        The new grid to which the polynomial fit will be rotated.
    bandwidth : float
        The bandwidth for the kernel.
    kernel_type : int, optional
        The type of kernel to use for the fit. Defaults to 0 (Gaussian kernel).
        Other values can be used for different kernels, such as Epanechnikov.

    Returns
    -------
    np.ndarray
        The rotated polynomial fit values at the new grid points.
    """
    if not isinstance(bandwidth, (int, float)):
        raise ValueError("bandwidth must be a numeric value.")
    if bandwidth <= 0:
        raise ValueError("bandwidth must be a positive number.")
    if np.isnan(bandwidth):
        raise ValueError("bandwidth must not be NaN.")
    bandwidth = float(bandwidth)

    if kernel_type not in KernelType:
        raise ValueError(f"kernel must be one of {list(KernelType)}.")

    input_dtype = x_grid.dtype
    x_grid = check_array(x_grid, ensure_2d=True, dtype=input_dtype)
    if x_grid.shape[1] != 2:
        raise ValueError("x_grid must be a 2D array with shape (n, 2), where n is the number of points.")
    y = check_array(y, ensure_2d=False, dtype=input_dtype)
    if y.ndim != 1:
        raise ValueError("y must be a 1D array.")
    w = check_array(w, ensure_2d=False, dtype=input_dtype)
    if w.ndim != 1:
        raise ValueError("w must be a 1D array.")
    new_grid = check_array(new_grid, ensure_2d=True, dtype=input_dtype)
    if new_grid.shape[1] != 2:
        raise ValueError("new_grid must be a 2D array with shape (m, 2), where m is the number of new points.")

    # rotate the grids
    rotation_matrix = np.array([[1, -1], [1, 1]]) / np.sqrt(2.0)
    x_grid_rotated = rotation_matrix @ x_grid.T
    new_grid_rotated = rotation_matrix @ new_grid.T

    # Sort the rotated grids
    sorted_idx = np.lexsort((x_grid_rotated[1, :], x_grid_rotated[0, :]))
    x_grid_rotated = np.ascontiguousarray(x_grid_rotated[:, sorted_idx]).astype(input_dtype)
    y_sorted = y[sorted_idx]
    w_sorted = w[sorted_idx]

    # Sort the new grid
    sorted_idx_new_grid = np.lexsort((new_grid_rotated[1, :], new_grid_rotated[0, :]))
    new_grid_rotated = np.ascontiguousarray(new_grid_rotated[:, sorted_idx_new_grid]).astype(input_dtype)

    rotate_polyfit2d_func = rotate_polyfit2d_f64 if input_dtype == np.float64 else rotate_polyfit2d_f32
    output = rotate_polyfit2d_func(x_grid_rotated, y_sorted, w_sorted, new_grid_rotated, bandwidth, kernel_type.value)
    return output


def get_measurement_error_variance(
    raw_cov: np.ndarray,
    reg_grid: np.ndarray,
    bandwidth: float,
    kernel_type: KernelType,
) -> float:
    """Estimate the measurement error variance from the raw covariance matrix.

    This function estimates the measurement error variance by fitting a polynomial to the raw covariance data
    and evaluating it at the regular grid points. It uses a kernel-based approach to smooth the covariance data.

    Parameters
    ----------
    raw_cov : np.ndarray
        The raw covariance matrix with columns representing (sid, t1, t2, w, cov).
        This input should use `get_raw_cov` to obtain the raw covariance data.
    obs_grid : np.ndarray
        The observation grid, which is a 1D array of sorted unique time points.
    reg_grid : np.ndarray
        The regular grid where the measurement error variance will be estimated.
    bandwidth : float
        The bandwidth for the kernel.
    kernel_type : KernelType
        The type of kernel to use for smoothing.

    Returns
    -------
    float
        The estimated measurement error variance.
    """

    input_dtype = raw_cov.dtype
    raw_cov = check_array(raw_cov, ensure_2d=True, dtype=input_dtype)
    if raw_cov.shape[1] != 5:
        raise ValueError("raw_cov must have 5 columns: (sid, t1, t2, w, cov).")
    reg_grid = check_array(reg_grid, ensure_2d=False, dtype=input_dtype)
    if reg_grid.ndim != 1:
        raise ValueError("reg_grid must be a 1D array.")
    if np.any(np.diff(reg_grid) <= 0):
        raise ValueError("reg_grid must be a 1D array with increasing values.")
    if not isinstance(bandwidth, (int, float)):
        raise ValueError("bandwidth must be a numeric value.")
    if bandwidth <= 0:
        raise ValueError("bandwidth must be a positive number.")
    if np.isnan(bandwidth):
        raise ValueError("bandwidth must not be NaN.")
    bandwidth = float(bandwidth)

    if kernel_type not in KernelType:
        raise ValueError(f"kernel must be one of {list(KernelType)}.")

    diag_mask = raw_cov[:, 1] == raw_cov[:, 2]
    diag_cov = raw_cov[diag_mask, :]
    smoothed_cov_diag_fit = Polyfit1DModel(kernel_type=kernel_type)
    smoothed_cov_diag_fit.fit(diag_cov[:, 1], diag_cov[:, 4], diag_cov[:, 3], bandwidth=bandwidth, reg_grid=reg_grid)
    smoothed_cov_diag = smoothed_cov_diag_fit.fitted_values()

    non_diag_mask = np.logical_not(diag_mask)
    diag_smoothed_cov_surface = rotate_polyfit2d(
        raw_cov[non_diag_mask, :][:, [1, 2]],
        raw_cov[non_diag_mask, 4],
        raw_cov[non_diag_mask, 3],
        new_grid=np.vstack((reg_grid, reg_grid)).T,
        bandwidth=bandwidth,
        kernel_type=kernel_type,
    )

    sigma2 = trapz(smoothed_cov_diag - diag_smoothed_cov_surface, reg_grid) / (reg_grid[-1] - reg_grid[0])
    return sigma2, smoothed_cov_diag, diag_smoothed_cov_surface
