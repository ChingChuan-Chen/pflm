"""Utility functions used for FPCA"""

# Authors: Ching-Chuan Chen
# SPDX-License-Identifier: MIT
import warnings
from typing import List, Tuple

import numpy as np

from pflm.utils._fpca_score import fpca_ce_score_f32, fpca_ce_score_f64
from pflm.utils._lapack_helper import _syevd_memview_f32, _syevd_memview_f64
from pflm.utils.utility import trapz


def get_eigen_analysis_results(reg_cov: np.ndarray, is_upper_triangular: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get eigenvalues and eigenvectors from the covariance of functional data.

    Parameters
    ----------
    reg_cov : np.ndarray
        The regularized covariance matrix of shape (nt, nt).
    is_upper_triangular : bool, optional
        Whether the covariance matrix is upper triangular. Defaults to False.

    Returns
    -------
    eig_lambda : np.ndarray
        The eigenvalues corresponding to the functional principal components with shape (nt,)
    eig_vector : np.ndarray
        The functional principal component basis functions (eigenvectors) of shape (p, nt).
    """
    # initialize eigenvalues and eigenvectors
    nt = reg_cov.shape[0]
    eig_lambda = np.zeros(nt, dtype=reg_cov.dtype)
    eig_vector = np.ascontiguousarray(reg_cov.ravel(order="F"))

    # compute eigenvalues and eigenvectors
    eig_func = _syevd_memview_f64 if reg_cov.dtype == np.float64 else _syevd_memview_f32
    uplo = 117 if is_upper_triangular else 108
    info = eig_func(eig_vector, eig_lambda, uplo, nt, nt)
    if info != 0:
        warnings.warn(f"LAPACK syevd failed with info={info}")
        return None, None
    if np.any(np.isnan(eig_lambda)) or np.any(eig_lambda < 0):
        warnings.warn("Eigenvalues contain NaN or negative values. The covariance function may not be positive semi-definite.")

    # sort eigen values and corresponding eigen vectors
    mask = np.isfinite(eig_lambda) & (eig_lambda > 10.0 * np.finfo(eig_lambda.dtype).eps)  # only leave significant eigenvalues
    ord = np.argsort(eig_lambda[mask])[::-1]
    eig_lambda = eig_lambda[mask][ord]
    eig_vector = eig_vector.reshape(nt, -1).T[:, mask][:, ord]
    return eig_lambda, eig_vector


def select_num_pcs_fve(eig_lambda: np.ndarray, fve_threshold: float, max_components: int = 20):
    """
    Select the number of principal components based on the explained variance.

    Parameters
    ----------
    eig_lambda : np.ndarray
        The eigenvalues corresponding to the functional principal components.
    fve_threshold : float
        The threshold for the proportion of variance explained by the functional principal components.
    max_components : int
        The maximum number of principal components to consider.

    Returns
    -------
    cumulative_fve : np.ndarray
        The cumulative explained variance for each principal component.
    num_pcs : int
        The number of principal components selected based on the explained variance.
    """
    cumulative_fve = np.cumsum(eig_lambda) / np.sum(eig_lambda)
    num_pcs = min(np.searchsorted(cumulative_fve, fve_threshold) + 1, max_components)
    return cumulative_fve, num_pcs


def get_fpca_phi(num_pcs: int, reg_grid: np.ndarray, reg_mu: np.ndarray, eig_lambda: np.ndarray, eig_vector: np.ndarray):
    """
    Get the functional principal component basis functions (FPCA phi).

    Parameters
    ----------
    reg_grid : np.ndarray
        The grid points corresponding to the functional data with shape (nt,).
    reg_mu : np.ndarray
        The mean function values at the grid points with shape (nt,).
    eig_lambda : np.ndarray
        The eigenvalues corresponding to the functional principal components.

    Returns
    -------
    fpca_lambda : np.ndarray
        The functional principal component eigenvalues.
    fpca_phi : np.ndarray
        The functional principal component basis functions of shape (nt, num_pcs).
    """
    grid_size = reg_grid[1] - reg_grid[0]

    # Scale eigenvalues by grid spacing (Riemann approximation)
    fpca_lambda = eig_lambda[:num_pcs] * grid_size

    # Scale eigenvectors: columns correspond to eigenfunctions sampled on reg_grid
    eig_vector_temp = eig_vector[:, :num_pcs] / np.sqrt(grid_size)

    # Normalize each eigenfunction to unit L2 norm: trapz expects shape (n_samples, n_points)
    # so pass transposed array (num_pcs, nt)
    energy = trapz((eig_vector_temp**2).T, reg_grid)  # shape (num_pcs,)
    # Avoid division by zero (should not happen for positive eigenvalues, but be safe)
    # energy[energy == 0] = 1.0
    scaled_eig_vector = eig_vector_temp / np.sqrt(energy)

    # Align signs so that inner product with mean function is non-negative
    signs = np.sign(np.sum(scaled_eig_vector * reg_mu.reshape((-1, 1)), axis=0))
    signs[signs == 0] = 1.0
    fpca_phi = scaled_eig_vector * signs
    return fpca_lambda, fpca_phi


def get_fpca_ce_score(
    yy: np.ndarray,
    tt: np.ndarray,
    tid: np.ndarray,
    sid: np.ndarray,
    mu: np.ndarray,
    fitted_cov: np.ndarray,
    fpca_lambda: np.ndarray,
    fpca_phi: np.ndarray,
    sigma2: float,
) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
    if yy.ndim != 1 or tt.ndim != 1:
        raise ValueError("yy and tt must be 1D arrays.")
    if yy.size != tt.size:
        raise ValueError("yy and tt must have the same length.")
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
    num_pcs = len(fpca_lambda)
    if fpca_phi.shape[0] != mu.size or fpca_phi.shape[1] != num_pcs:
        raise ValueError("fpca_phi must have shape (mu.size, num_pcs).")

    sigma_y = (fitted_cov + np.eye(fitted_cov.shape[0]) * sigma2).astype(yy.dtype, copy=False)
    fpca_ce_score_func = fpca_ce_score_f64 if yy.dtype == np.float64 else fpca_ce_score_f32
    lambda_phi = np.ascontiguousarray(fpca_phi @ np.diag(fpca_lambda)).astype(yy.dtype, copy=False)
    xi, xi_var = fpca_ce_score_func(yy, tt, tid, mu, sigma_y, fpca_lambda, lambda_phi, unique_sid, sid_cnt)
    fitted_y = mu + fpca_phi @ xi.T
    return xi, xi_var, fitted_y


def estimate_rho(
    method_rho: str,
    yy: np.ndarray,
    tt: np.ndarray,
    tid: np.ndarray,
    sid: np.ndarray,
    mu: np.ndarray,
    fitted_cov: np.ndarray,
    fpca_lambda: np.ndarray,
    fpca_phi: np.ndarray,
    sigma2: float,
):
    _, _, fitted_y = get_fpca_ce_score(yy, tt, tid, sid, mu, fitted_cov, fpca_lambda, fpca_phi, sigma2)
    unique_sid = np.unique(sid, sorted=True)
    fitted_y_list = [fitted_y[tid[sid == i], i] for i in unique_sid]
    rss = np.mean([np.mean((fitted_y_i - mu[tid[sid == i]]) ** 2) for i, fitted_y_i in zip(unique_sid, fitted_y_list)])

    obs_grid = np.unique(tt, sorted=True)
    total_fpca_lambda = np.sum(fpca_lambda)
    if method_rho == "ridge":
        r = np.sqrt((trapz(obs_grid, mu**2) + total_fpca_lambda) / (obs_grid[-1] - obs_grid[0]))
        rho_candidates = np.exp(np.linspace(-13, -1.5, 50, dtype=yy.dtype)) * r
    else:
        rho_candidates = np.linspace(1, 10, 50, dtype=yy.dtype) * rss

    num_pcs = fpca_lambda.size
    rho_scores = np.zeros(len(rho_candidates), dtype=yy.dtype)
    for idx, rho in enumerate(rho_candidates):
        xi, _, _ = get_fpca_ce_score(yy, tt, tid, sid, mu, fitted_cov, fpca_lambda, fpca_phi, sigma2=rho)
        fitted_y = mu + xi[:, :num_pcs] @ fpca_phi[:, :num_pcs].T
        var_y = np.var(fitted_y, axis=0)
        rho_scores[idx] = (total_fpca_lambda - trapz(obs_grid, var_y)) ** 2
    return rho_candidates[np.argmin(rho_scores)], rho_candidates, rho_scores


def get_fpca_in_score():
    return np.array([]), [], np.array([])
