"""Utility functions for flattening data matrices and computing eigenvalues and eigenvectors"""

# Authors: Ching-Chuan Chen
# SPDX-License-Identifier: MIT
from typing import Tuple, List, Union

import numpy as np
from sklearn.utils.validation import check_array

from pflm.utils import trapz


def flatten_and_sort_data_matrices(
    y: List[Union[np.ndarray, List[float]]],
    t: List[Union[np.ndarray, List[float]]],
    input_dtype: Union[str, np.dtype] = np.float64,
    w: List[Union[np.ndarray, List[float]]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Flatten and sort the data matrices.

    Parameters
    ----------
    y : list of array_like
        The response list, where each element is a 1D array of shape (nt_i,), i= 0, 1, ..., n-1.
    t : list of array_like
        The time points, where each element is a 1D array of shape (nt_i,), i= 0, 1, ..., n-1.
        It should be a 1D array where each element corresponds to a time point.
    input_dtype : str or np.dtype, optional
        The data type of the input arrays. Defaults to np.float64.
    w : list of array_like, optional
        The weights for each sample. If provided, it should have the same number of rows as `y`.

    Returns
    -------
    sid : np.ndarray
        A 1D array of sample indices, where each index corresponds to the sample in `y`.
        The indices are sorted by time.
    yy : np.ndarray
        A 1D array of the flattened response values, sorted by time.
    tt : np.ndarray
        A 1D array of the corresponding time points, sorted to match `yy`.
    ww : np.ndarray
        A 1D array of the weights corresponding to `yy`, sorted to match `yy`.
    """
    if not isinstance(y, list):
        raise ValueError("y must be a list of arrays.")
    if not isinstance(t, list):
        raise ValueError("t must be a list of arrays.")
    if len(y) != len(t):
        raise ValueError("The length of y and t must be the same.")

    if w is not None and len(y) != len(w):
        raise ValueError("The length of y and w must be the same.")
    if w is not None and not isinstance(w, list):
        raise ValueError("Weights w must be a list of 1D arrays.")

    if w is None:
        w = [np.ones_like(yi) for yi in y]
    else:
        w = [check_array(wi, ensure_2d=False, dtype=input_dtype) for wi in w]
        for ti, wi in zip(t, w):
            if ti.size != wi.size:
                raise ValueError("Each element of t and w must have the same length.")

    total_length = np.sum([yi.size for yi in y])
    sid = np.empty(total_length, dtype=input_dtype)
    yy = np.empty(total_length, dtype=input_dtype)
    tt = np.empty(total_length, dtype=input_dtype)
    ww = np.empty(total_length, dtype=input_dtype)

    i = 0
    idx = 0
    for yi, ti, wi in zip(y, t, w):
        if yi.size == 0 or ti.size == 0:
            continue
        if yi.ndim != 1 or ti.ndim != 1:
            raise ValueError("Each element of y and t must be a 1D array.")
        if yi.size != ti.size:
            raise ValueError("Each element of y and t must have the same length.")

        # Flatten and sort the data
        sid[idx:idx + yi.size] = i
        yy[idx:idx + yi.size] = yi
        tt[idx:idx + ti.size] = ti
        ww[idx:idx + wi.size] = wi
        idx += yi.size
        i += 1

    # sort the flattened arrays by time
    sort_idx = np.argsort(tt[~np.isnan(yy)])
    sid = sid[~np.isnan(yy)][sort_idx]
    tt = tt[~np.isnan(yy)][sort_idx]
    ww = ww[~np.isnan(yy)][sort_idx]
    yy = yy[~np.isnan(yy)][sort_idx]
    return sid, yy, tt, ww


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
