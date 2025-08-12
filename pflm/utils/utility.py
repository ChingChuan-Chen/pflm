"""Utility functions for flattening data matrices and computing eigenvalues and eigenvectors"""

# Authors: Ching-Chuan Chen
# SPDX-License-Identifier: MIT
from typing import List, Optional, Tuple, Union

import numpy as np
from sklearn.utils.validation import check_array

from pflm.smooth.kernel import KernelType
from pflm.utils import trapz
from pflm.utils._rotate_polyfit2d import rotate_polyfit2d_f32, rotate_polyfit2d_f64
from pflm.utils._raw_cov import get_raw_cov_f32, get_raw_cov_f64


def flatten_and_sort_data_matrices(
    y: List[np.ndarray],
    t: List[np.ndarray],
    input_dtype: Union[str, np.dtype] = np.float64,
    w: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Flatten and sort the data matrices.

    This function takes a list of response matrices `y` and corresponding time points `t`,
    and optionally weights `w`, and flattens them into 1D arrays. The function handles NaN values by excluding them
    from the output. If all values in `y` are NaN, it raises a ValueError.

    Parameters
    ----------
    y : list of array_like
        The response list, where each element is a 1D array of shape (nt_i,), i= 0, 1, ..., n-1.
    t : list of array_like
        The time points, where each element is a 1D array of shape (nt_i,), i= 0, 1, ..., n-1.
        It should be a 1D array where each element corresponds to a time point.
    input_dtype : str or np.dtype, optional
        The data type of the input arrays. Defaults to np.float64.
    w : array_like, optional
        The weights for each sample. If provided, it should have the same length as `y` and `t`.

    Returns
    -------
    yy : np.ndarray
        A 1D array of the flattened response values, sorted by sample index and time points.
    tt : np.ndarray
        A 1D array of the corresponding time points, sorted to match `yy`.
    ww : np.ndarray
        A 1D array of the weights corresponding to `yy`, sorted to match `yy`.
    sid : np.ndarray
        A 1D array of sample indices, where each index corresponds to the sample in `y`.
    """
    if not isinstance(y, list):
        raise ValueError("y must be a list of arrays.")
    if not isinstance(t, list):
        raise ValueError("t must be a list of arrays.")
    if len(y) != len(t):
        raise ValueError("The length of y and t must be the same.")
    for yi, ti in zip(y, t):
        if yi.ndim != 1 or ti.ndim != 1:
            raise ValueError("Each element of y and t must be a 1D array.")
        if yi.size != ti.size:
            raise ValueError("Each element of y and t must have the same length.")

    if w is None:
        w = np.ones((len(y),), dtype=input_dtype)
    else:
        if not isinstance(w, np.ndarray):
            raise ValueError("Weights w must be a 1D array.")
        if len(y) != w.size:
            raise ValueError("The length of y and w must be the same.")
        w = check_array(w, ensure_2d=False, dtype=input_dtype)
        if w.ndim != 1:
            raise ValueError("Each element of w must be a 1D array.")

    yy = np.concatenate(y).astype(input_dtype, copy=False)
    non_nan_mask = ~np.isnan(yy)
    if not non_nan_mask.any():
        raise ValueError("All values in y are NaN. Cannot flatten data matrices.")

    tt = np.concatenate(t).astype(input_dtype, copy=False)[non_nan_mask]
    ww = np.concatenate([np.full(yi.size, wi, dtype=input_dtype) for yi, wi in zip(y, w)])[non_nan_mask]
    sid = np.concatenate([np.full(yi.size, i, dtype=np.int64) for i, yi in enumerate(y)])[non_nan_mask]
    return yy[non_nan_mask], tt, ww, sid


def get_raw_cov(yy: np.ndarray, tt: np.ndarray, ww: np.ndarray, mu: np.ndarray, tid: np.ndarray, sid: np.ndarray) -> np.ndarray:
    """
    Get the dense covariance matrix from the flattened data matrices.

    This function computes the covariance between pairs of samples at different time points,
    using the flattened response values `yy`, corresponding time points `tt`, and weights `ww`.
    It returns a raw covariance matrix with columns representing (sid, t1, t2, w, cov),
    where `sid` is the sample index, `t1` and `t2` are the time points, `w` is the weight,
    and `cov` is the computed covariance value.

    Parameters
    ----------
    yy : np.ndarray
        Flattened response values without NaNs.
    tt : np.ndarray
        Corresponding time points to `yy`.
    ww : np.ndarray
        Weights corresponding to `yy`.
    mu : np.ndarray
        Mean function values at the observation grid.
    tid : np.ndarray
        Time indices corresponding to `tt`.
    sid : np.ndarray
        Sample indices corresponding to `yy`.

    Returns
    -------
    np.ndarray
        The raw covariance matrix with columns representing (sid, t1, t2, w, cov).
        The shape is (num_pairs, 5), where num_pairs is the number of unique pairs of samples.
    """
    if yy.ndim != 1 or tt.ndim != 1 or ww.ndim != 1:
        raise ValueError("yy, tt, and ww must be 1D arrays.")
    if yy.size != tt.size or yy.size != ww.size:
        raise ValueError("yy, tt, and ww must have the same length.")
    unique_tid = np.unique(tid)
    if unique_tid.size != mu.size:
        raise ValueError("The length of mu must match the number of unique time indices in tid.")
    if sid.ndim != 1 or sid.size != yy.size:
        raise ValueError("sid must be a 1D array with the same length as yy.")
    if np.any(np.diff(sid) < 0):
        raise ValueError("The sample indices, sid, must be sorted in ascending order.")
    unique_sid, sid_cnt = np.unique(sid, return_counts=True, sorted=True)
    if np.any(sid_cnt < 2):
        raise ValueError("Each sample must have at least two observations for covariance calculation.")

    get_raw_cov_func = get_raw_cov_f64 if yy.dtype == np.float64 else get_raw_cov_f32
    return get_raw_cov_func(yy, tt, ww, mu, tid, unique_sid, sid_cnt)


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
    bandwidth = float(bandwidth)

    if kernel_type not in KernelType:
        raise ValueError(f"kernel must be one of {list(KernelType)}.")

    input_dtype = x_grid.dtype
    x_grid = check_array(x_grid, ensure_2d=True, dtype=input_dtype)
    if x_grid.shape[1] != 2:
        raise ValueError("x_grid must be a 2D array with shape (2, n), where n is the number of points.")
    y = check_array(y, ensure_2d=False, dtype=input_dtype)
    if y.ndim != 1:
        raise ValueError("y must be a 1D array.")
    w = check_array(w, ensure_2d=False, dtype=input_dtype)
    if w.ndim != 1:
        raise ValueError("w must be a 1D array.")
    new_grid = check_array(new_grid, ensure_2d=True, dtype=input_dtype)
    if new_grid.shape[1] != 2:
        raise ValueError("new_grid must be a 2D array with shape (2, m), where m is the number of new points.")

    rotation_matrix = np.array([[1, -1], [1, 1]], dtype=input_dtype) / np.sqrt(2.0)
    x_grid_rotated = rotation_matrix @ x_grid
    new_grid_rotated = rotation_matrix @ new_grid

    # Sort the rotated grids
    sorted_idx = np.lexsort((x_grid_rotated[:, 1], x_grid_rotated[:, 0]))
    x_grid_rotated = np.ascontiguousarray(x_grid_rotated[:, sorted_idx])
    sorted_idx_new_grid = np.lexsort((new_grid_rotated[:, 1], new_grid_rotated[:, 0]))
    new_grid_rotated = np.ascontiguousarray(new_grid_rotated[:, sorted_idx_new_grid])

    rotate_polyfit2d_func = rotate_polyfit2d_f64 if input_dtype == np.float64 else rotate_polyfit2d_f32
    output = rotate_polyfit2d_func(
        np.ascontiguousarray(x_grid_rotated.T), y, w, np.ascontiguousarray(new_grid_rotated.T), bandwidth, kernel_type.value
    )
    return output


def get_eigen_results(
    t: np.ndarray, mean_func: np.ndarray, cov_func: np.ndarray, fve_thresh: float, max_principal: int = 20
) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Get eigenvalues and eigenvectors of the covariance function.

    Parameters
    ----------
    y : array_like
        The response matrix of shape (n, nt), where n is the number of samples and
        nt is the number of time points.
        Each row corresponds to a sample and each column corresponds to a time point.
        The values can be NaN, which will be ignored in the flattening process.
    t : array_like
        The time points of shape (nt,).
        It should be a 1D array where each element corresponds to a time point.
    mean_func : array_like
        The mean function values at the time points of shape (nt,).
        It should be a 1D array where each element corresponds to the mean value at the
        corresponding time point in `t`.
    cov_func : array_like
        The covariance function matrix of shape (nt, nt).
        It should be a square matrix where the (i, j)-th entry represents the covariance
        between the functional data at time points `t[i]` and `t[j]`.
        The covariance function must be positive semi-definite.
    fve_thresh : float
        The threshold for the proportion of variation explained by the functional principal
        components which must be between 0 and 1 (exclusive).
        If the threshold is not met, the number of principal components will be set to the maximum
        number of principal components specified by `max_principal`.
    max_principal : int, optional
        The maximum number of principal components to consider. Defaults to 20.

    Returns
    -------
    num_fpca : int
        The number of functional principal components that explain at least `fve_thresh` proportion of the
        functional variance.
    fpca_lambda : np.ndarray
        The eigenvalues corresponding to the functional principal components, scaled by the time step size.
    fpca_phi : np.ndarray
        The functional principal component basis functions (eigenvectors) of shape (p, num_fpca).
    cumu_fve : np.ndarray
        The cumulative functional variance explained by the functional principal components, normalized to sum to 1.
    """
    if mean_func.size == 0 or cov_func.size == 0 or t.size == 0:
        raise ValueError("mean_func, cov_func, and t must not be empty.")
    if mean_func.ndim != 1:
        raise ValueError("mean_func must be a 1D array.")
    if cov_func.ndim != 2 or cov_func.shape[0] != cov_func.shape[1]:
        raise ValueError("cov_func must be a square 2D array.")
    if t.ndim != 1:
        raise ValueError("t must be a 1D array.")
    if mean_func.shape[0] != t.shape[0]:
        raise ValueError("The length of mean_func must be equal to the length of t.")
    if cov_func.shape[0] != t.shape[0] or cov_func.shape[1] != t.shape[0]:
        raise ValueError("The shape of cov_func must match the length of t.")
    if fve_thresh <= 0 or fve_thresh >= 1:
        raise ValueError("fve_thresh must be between 0 and 1 (exclusive).")
    if max_principal <= 0:
        raise ValueError("max_principal must be a positive integer.")
    if np.isnan(mean_func).any() or np.isnan(cov_func).any() or np.isnan(t).any():
        raise ValueError("mean_func, cov_func, and t must not contain NaN values.")

    # get eigenvalues and eigenvectors of covariance function
    eigen_res = np.linalg.eigh(cov_func, "U")
    order_vec = np.argsort(eigen_res.eigenvalues, kind="stable")
    order_vec = order_vec[eigen_res.eigenvalues > 0]
    if len(order_vec) == 0:
        raise ValueError("No positive lambdas found.")
    cumu_fve = np.cumsum(eigen_res.eigenvalues[order_vec[::-1]])
    cumu_fve /= cumu_fve[-1]
    num_fpca = min(np.searchsorted(cumu_fve, fve_thresh) + 1, max_principal)

    # get fpca base functions (phi)
    h = (np.max(t) - np.min(t)) / (len(t) - 1)
    fpca_lambda = eigen_res.eigenvalues[order_vec[::-1][:num_fpca]] * h
    fpca_phi = eigen_res.eigenvectors[:, order_vec[::-1][:num_fpca]].reshape((-1, num_fpca)) / np.sqrt(h)
    phi_factor = np.sqrt(trapz(fpca_phi.T**2, t))
    fpca_phi /= phi_factor
    fpca_phi *= np.sign(np.sum(fpca_phi * mean_func.reshape((-1, 1)), axis=0))

    # get fitted covariance function
    return num_fpca, fpca_lambda, fpca_phi, cumu_fve
