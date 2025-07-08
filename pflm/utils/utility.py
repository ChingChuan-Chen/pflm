# Authors: The pflm developers
# SPDX-License-Identifier: MIT
from typing import Tuple

import numpy as np

from pflm.utils import trapz


def flatten_data_matrix(y: np.ndarray, t: np.ndarray, w: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten and sort the data matrices.

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
    w : array_like, optional
        The weights of shape (n,). If provided, it should have the same number of rows as `y`.
        If not provided, a default weight of 1 will be used for all samples,
        and NaN values in `y` will result in NaN weights.

    Returns
    -------
    yy : array_like
        A 1D array of the flattened response values, sorted by time.
    tt : array_like
        A 1D array of the corresponding time points, sorted to match `yy`.
    ww : array_like
        A 1D array of the weights corresponding to `yy`, sorted to match `yy`.
    """
    if y.ndim != 2:
        raise ValueError("y must be a 2D array.")
    if t.ndim != 1:
        raise ValueError("t must be a 1D array.")
    if y.size == 0 or t.size == 0:
        raise ValueError("y and t must not be empty.")

    if y.shape[1] != t.shape[0]:
        raise ValueError("The number of columns of y must be equal to the length of t.")

    if w is not None and y.shape[0] != w.shape[0]:
        raise ValueError("The number of rows of y must be equal to the length of w.")

    if w is None:
        w = np.ones_like(y)
        w[np.isnan(y)] = np.nan
    else:
        w = np.repeat(w.reshape((-1, 1)), y.shape[1], axis=1)
        w[np.isnan(y)] = np.nan

    n = y.shape[0]
    yy_temp = y.reshape((-1,))
    tt_temp = np.repeat(t.reshape((1, -1)), n, 0).reshape((-1,))
    sort_idx = np.argsort(tt_temp[~np.isnan(yy_temp)])
    tt = tt_temp[~np.isnan(yy_temp)][sort_idx]
    yy = y.reshape((-1,))[~np.isnan(yy_temp)][sort_idx]
    ww = w.reshape((-1,))[~np.isnan(yy_temp)][sort_idx]
    return yy, tt, ww


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
