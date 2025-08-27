"""Utility functions for covariance estimation on functional data."""

# Authors: Ching-Chuan Chen
# SPDX-License-Identifier: MIT

import numpy as np
from sklearn.utils.validation import check_array

from pflm.smooth import KernelType, Polyfit1DModel
from pflm.utils.utility import FlattenFunctionalData, trapz
from pflm.fpca._raw_cov import get_raw_cov_f32, get_raw_cov_f64
from pflm.fpca._rotate_polyfit2d import rotate_polyfit2d_f32, rotate_polyfit2d_f64


def get_raw_cov(flatten_func_data: FlattenFunctionalData, mu: np.ndarray) -> np.ndarray:
    """Compute per-subject raw covariance entries on the observation grid.

    For each subject, this function forms all within-subject pairs and computes
    raw covariance entries (y_i(t1) - mu(t1)) * (y_i(t2) - mu(t2)) with the
    associated subject id and per-subject weight, returning a compact table
    suitable for later binning/aggregation.

    Parameters
    ----------
    flatten_func_data : FlattenFunctionalData
        Flattened dataset with fields `y`, `t`, `w`, `tid`, `unique_sid`, and `sid_cnt`.
        The `unique_tid` field defines the observation grid ordering.
    mu : np.ndarray of shape (nt,)
        Mean evaluated on the observation grid `flatten_func_data.unique_tid`.

    Returns
    -------
    raw_cov : np.ndarray of shape (M, 5)
        Columns ordered as (sid, t1, t2, w, cov), where M is the total number
        of within-subject pairs across all subjects.

    Raises
    ------
    ValueError
        If the length of `mu` does not match the number of unique time points.

    Notes
    -----
    - The output is not yet symmetrized nor aggregated; use
      `get_covariance_matrix` to map these entries to a dense symmetric matrix.
    - The weight column `w` typically reflects subject-level weights expanded
      to pairwise entries.
    """
    nt = flatten_func_data.unique_tid.size
    if mu.size != nt:
        raise ValueError("The length of mu must match the number of unique time indices.")

    input_dtype = flatten_func_data.y.dtype
    get_raw_cov_func = get_raw_cov_f64 if input_dtype == np.float64 else get_raw_cov_f32
    raw_cov = get_raw_cov_func(
        flatten_func_data.y,
        flatten_func_data.t,
        flatten_func_data.w,
        mu,
        flatten_func_data.tid,
        flatten_func_data.unique_sid,
        flatten_func_data.sid_cnt,
    )
    return raw_cov


def get_covariance_matrix(raw_cov: np.ndarray, obs_grid: np.ndarray) -> np.ndarray:
    """Aggregate raw covariance entries onto a dense symmetric matrix.

    This maps the compact raw covariance table returned by `get_raw_cov`
    to a dense covariance matrix on the observation grid by summing weighted
    covariances per unique (t1, t2) pair and normalizing by (sum_w - 1).

    Parameters
    ----------
    raw_cov : np.ndarray of shape (M, 5)
        Columns are (sid, t1, t2, w, cov), typically from `get_raw_cov`.
    obs_grid : np.ndarray of shape (nt,)
        Sorted unique observation grid values.

    Returns
    -------
    cov_matrix : np.ndarray of shape (nt, nt)
        Symmetric covariance matrix aligned to `obs_grid`.

    Raises
    ------
    ValueError
        If `raw_cov` does not have at least 5 columns, or `obs_grid` is empty.

    Notes
    -----
    - Pairs with total weight <= 1 are set to zero to avoid division by zero.
    - The diagonal is adjusted to ensure symmetry after filling the upper-triangular part.
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
    """Evaluate a 2D local polynomial fit on a rotated/new grid.

    Performs weighted local polynomial regression on the input grid and
    evaluates the smoothed surface at `new_grid`. The underlying computation
    is delegated to optimized low-level routines.

    Parameters
    ----------
    x_grid : np.ndarray of shape (2, n)
        Original 2D coordinates where responses are observed: stacked as
        [x; y] with two rows and n columns.
    y : np.ndarray of shape (n,)
        Observed responses aligned with columns of `x_grid`.
    w : np.ndarray of shape (n,)
        Non-negative sample weights.
    new_grid : np.ndarray of shape (2, m)
        New 2D coordinates (stacked rows) at which to evaluate the fitted surface.
    bandwidth : float
        Positive bandwidth parameter for the kernel.
    kernel_type : KernelType, default=KernelType.GAUSSIAN
        Kernel used by the local polynomial smoother.

    Returns
    -------
    y_new : np.ndarray of shape (m,)
        Fitted values evaluated at `new_grid`.

    Raises
    ------
    TypeError
        If `bandwidth` is not a real number.
    ValueError
        If `bandwidth` is non-positive or NaN; if `kernel_type` is not a valid
        `KernelType`; or if input array dimensions are inconsistent.

    Notes
    -----
    - Inputs are validated and coerced to a consistent dtype before calling
      the low-level routine.
    - This function does not select bandwidth; it assumes `bandwidth` is given.
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
    """Estimate measurement error variance from raw covariance near the diagonal.

    Smooths the raw covariance surface along the diagonal and leverages that
    the theoretical covariance at zero-lag equals the process variance, while
    the observed raw covariance includes measurement error. The difference
    (or an extrapolation to zero lag) yields an estimate of the noise variance.

    Parameters
    ----------
    raw_cov : np.ndarray of shape (M, 5)
        Raw covariance entries (sid, t1, t2, w, cov).
    reg_grid : np.ndarray of shape (nt,)
        Regular grid used to evaluate the smoothed covariance.
    bandwidth : float
        Positive bandwidth for the smoothing kernel.
    kernel_type : KernelType
        Kernel used during smoothing.

    Returns
    -------
    sigma2 : float
        Estimated measurement error variance.

    Raises
    ------
    ValueError
        If inputs are inconsistent or `bandwidth` is invalid.

    See Also
    --------
    rotate_polyfit2d : 2D local polynomial smoothing on rotated grids.
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
