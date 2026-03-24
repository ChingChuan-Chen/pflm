"""Utility functions used for FPCA"""

# Authors: Ching-Chuan Chen
# SPDX-License-Identifier: MIT
import warnings
from typing import List, Literal, Optional, Tuple

import numpy as np

from pflm.interp import interp1d
from pflm.fpca.utils.log_lik import get_log_likelihood_f32, get_log_likelihood_f64
from pflm.utils.blas_helper import BLAS_Jobz, BLAS_Uplo
from pflm.utils.lapack_helper import _syevd_memview_f32, _syevd_memview_f64
from pflm.utils.utility import trapz


def get_eigen_analysis_results(reg_cov: np.ndarray, is_upper_triangular: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute eigenvalues and eigenvectors of a covariance matrix.

    Parameters
    ----------
    reg_cov : np.ndarray of shape (nt, nt)
        Regularized covariance matrix. Dtype determines the LAPACK routine
        (float64 -> f64 backend; otherwise f32).
    is_upper_triangular : bool, default=False
        Whether `reg_cov` contains only the upper triangular part (packed in a
        full matrix). If True, the routine will treat the lower part as
        unspecified.

    Returns
    -------
    eig_lambda : np.ndarray of shape (k,)
        Sorted eigenvalues (descending) filtered to positive and finite values.
    eig_vector : np.ndarray of shape (nt, k)
        Corresponding eigenvectors (columns) aligned with `eig_lambda`.

    Warns
    -----
    UserWarning
        If the LAPACK routine fails (`info != 0`) or eigenvalues contain NaN or
        negative values.

    Notes
    -----
    - Very small eigenvalues (<= 10 * eps) are discarded.
    - On failure, the function returns (None, None).
    """
    # initialize eigenvalues and eigenvectors
    nt = reg_cov.shape[0]
    eig_lambda = np.zeros(nt, dtype=reg_cov.dtype)
    eig_vector = reg_cov.copy()

    # compute eigenvalues and eigenvectors
    eig_func = _syevd_memview_f64 if reg_cov.dtype == np.float64 else _syevd_memview_f32
    uplo = BLAS_Uplo.Upper if is_upper_triangular else BLAS_Uplo.Lower
    info = eig_func(BLAS_Jobz.Vec, uplo, eig_vector, eig_lambda, nt)
    if info != 0:
        warnings.warn(f"LAPACK syevd failed with info={info}")
        return None, None
    if np.any(np.isnan(eig_lambda)):
        warnings.warn("Eigenvalues contain NaN values. The covariance function may be ill-conditioned.")
    elif np.any(eig_lambda < 0):
        max_eig = eig_lambda.max()
        # Warn only when negative eigenvalues are large relative to the
        # leading eigenvalue, not for harmless numerical noise.
        if max_eig <= 0 or np.abs(eig_lambda.min()) / max_eig > np.sqrt(np.finfo(eig_lambda.dtype).eps):
            warnings.warn(
                "Eigenvalues contain significant negative values. "
                "The covariance function may not be positive semi-definite."
            )

    # sort eigen values and corresponding eigen vectors
    mask = np.isfinite(eig_lambda) & (eig_lambda > 10.0 * np.finfo(eig_lambda.dtype).eps)  # only leave significant eigenvalues
    ord_idx = np.argsort(eig_lambda[mask])[::-1]
    return eig_lambda[mask][ord_idx], eig_vector[:, mask][:, ord_idx]


def select_num_pcs_fve(eig_lambda: np.ndarray, fve_threshold: float, max_components: int = 20):
    """
    Select the number of principal components based on cumulative explained variance.

    Parameters
    ----------
    eig_lambda : np.ndarray of shape (k,)
        Non-negative eigenvalues.
    fve_threshold : float
        Target fraction of variance explained (typically in (0, 1]).
    max_components : int, default=20
        Upper bound on the number of components considered.

    Returns
    -------
    cumulative_fve : np.ndarray of shape (k,)
        Cumulative explained variance curve.
    num_pcs : int
        Number of components needed to reach the threshold, clipped by `max_components`.

    Notes
    -----
    Assumes `eig_lambda` sums to a positive value; otherwise the result is undefined.
    """
    cumulative_fve = np.cumsum(eig_lambda) / np.sum(eig_lambda)
    num_pcs = min(np.searchsorted(cumulative_fve, fve_threshold) + 1, max_components)
    return cumulative_fve, num_pcs


def select_num_pcs_ic(
    criterion: Literal["AIC", "BIC"],
    y: List[np.ndarray],
    t: List[np.ndarray],
    obs_grid: np.ndarray,
    reg_grid: np.ndarray,
    reg_mu: np.ndarray,
    mu_obs: np.ndarray,
    eig_lambda: np.ndarray,
    eig_vector: np.ndarray,
    max_components: int = 20,
    measurement_error_variance: float = 0.0,
    rho: Optional[float] = None,
) -> Tuple[np.ndarray, int]:
    """Select number of PCs via AIC/BIC with fdapace-style early stopping."""
    if criterion not in ["AIC", "BIC"]:
        raise ValueError("criterion must be either 'AIC' or 'BIC'.")

    max_candidates = min(int(max_components), int(eig_lambda.size))
    if max_candidates < 1:
        raise ValueError("No available principal components for IC-based selection.")

    input_dtype = np.result_type(y[0].dtype, t[0].dtype)

    sigma2 = float(measurement_error_variance)
    if rho is not None and sigma2 <= float(rho):
        sigma2 = float(rho)
    if sigma2 < 0.0:
        sigma2 = 0.0
    if sigma2 == 0.0:
        y_all = np.concatenate(y)
        sigma2 = max(float(np.var(y_all)) * 1e-8, float(np.finfo(input_dtype).eps) * 10.0)

    cy_loglik_func = get_log_likelihood_f64 if input_dtype == np.float64 else get_log_likelihood_f32

    # Build flattened subject-wise arrays once; per-k updates only rebuild basis/covariance.
    sid_cnt = np.asarray([len(y_i) for y_i in y], dtype=np.int64)
    unique_sid = np.arange(len(y), dtype=np.int64)
    yy = np.ascontiguousarray(np.concatenate(y), dtype=input_dtype)
    tt = np.ascontiguousarray(np.concatenate(t), dtype=input_dtype)
    tid = np.ascontiguousarray(np.concatenate([np.searchsorted(obs_grid, t_i) for t_i in t]), dtype=np.int64)
    mu_obs_cast = np.ascontiguousarray(mu_obs, dtype=input_dtype)

    penalty = 2.0 if criterion == "AIC" else float(np.log(len(y)))
    ic_values = []
    selected_k = max_candidates

    for k in range(1, max_candidates + 1):
        fpca_lambda, fpca_phi_reg = get_fpca_phi(k, reg_grid, reg_mu, eig_lambda, eig_vector)
        fpca_lambda = np.ascontiguousarray(fpca_lambda, dtype=input_dtype)

        fpca_phi_obs = np.zeros((len(obs_grid), k), dtype=input_dtype)
        for i in range(k):
            fpca_phi_obs[:, i] = interp1d(reg_grid, fpca_phi_reg[:, i], obs_grid, method="spline")
        fpca_phi_obs = np.ascontiguousarray(fpca_phi_obs, dtype=input_dtype)

        sigma_y = fpca_phi_obs @ np.diag(fpca_lambda) @ fpca_phi_obs.T
        sigma_y = np.ascontiguousarray(sigma_y, dtype=input_dtype)
        if sigma2 > 0.0:
            sigma_y = sigma_y.copy()
            np.fill_diagonal(sigma_y, np.diagonal(sigma_y) + sigma2)

        lambda_phi = np.ascontiguousarray(fpca_phi_obs * fpca_lambda.reshape((1, -1)), dtype=input_dtype)

        loglik_k = cy_loglik_func(
            yy,
            tt,
            tid,
            mu_obs_cast,
            sigma_y,
            fpca_lambda,
            lambda_phi,
            unique_sid,
            sid_cnt,
        )
        ic_k = float(loglik_k + penalty * k)
        ic_values.append(ic_k)

        if k > 1 and ic_values[-1] > ic_values[-2]:
            selected_k = k - 1
            break
        if k == max_candidates:
            selected_k = k

    return np.asarray(ic_values, dtype=input_dtype), int(selected_k)


def get_fpca_phi(num_pcs: int, reg_grid: np.ndarray, reg_mu: np.ndarray, eig_lambda: np.ndarray, eig_vector: np.ndarray):
    """
    Build FPCA eigenvalues/eigenfunctions normalized on the grid.

    Parameters
    ----------
    num_pcs : int
        Number of components to return (first `num_pcs`).
    reg_grid : np.ndarray of shape (nt,)
        Grid points where eigenfunctions are sampled (monotonic).
    reg_mu : np.ndarray of shape (nt,)
        Mean values on `reg_grid`, used for sign alignment.
    eig_lambda : np.ndarray of shape (k,)
        Raw eigenvalues from the covariance decomposition.
    eig_vector : np.ndarray of shape (nt, k)
        Raw eigenvectors (columns) from the covariance decomposition.

    Returns
    -------
    fpca_lambda : np.ndarray of shape (num_pcs,)
        Grid-scaled eigenvalues (Riemann approximation).
    fpca_phi : np.ndarray of shape (nt, num_pcs)
        Grid-normalized eigenfunctions with sign aligned to `reg_mu`.

    Notes
    -----
    - Eigenvalues are scaled by the grid spacing; eigenvectors are normalized
      to unit L2 norm on `reg_grid`.
    - Signs are chosen so that <phi_j, reg_mu> >= 0 for each component.
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
