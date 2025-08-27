"""Utility functions used for FPCA"""

# Authors: Ching-Chuan Chen
# SPDX-License-Identifier: MIT
import warnings
from typing import Tuple

import numpy as np

from pflm.utils._lapack_helper import _syevd_memview_f32, _syevd_memview_f64
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
    eig_vector = reg_cov.copy().ravel()

    # compute eigenvalues and eigenvectors
    eig_func = _syevd_memview_f64 if reg_cov.dtype == np.float64 else _syevd_memview_f32
    uplo = 117 if is_upper_triangular else 108  # 'u'/'l'
    info = eig_func(eig_vector, eig_lambda, uplo, nt, nt)
    if info != 0:
        warnings.warn(f"LAPACK syevd failed with info={info}")
        return None, None
    if np.any(np.isnan(eig_lambda)) or np.any(eig_lambda < 0):
        warnings.warn("Eigenvalues contain NaN or negative values. The covariance function may not be positive semi-definite.")

    # sort eigen values and corresponding eigen vectors
    mask = np.isfinite(eig_lambda) & (eig_lambda > 10.0 * np.finfo(eig_lambda.dtype).eps)  # only leave significant eigenvalues
    ord_idx = np.argsort(eig_lambda[mask])[::-1]
    return eig_lambda[mask][ord_idx], eig_vector.reshape(-1, nt).T[:, mask][:, ord_idx]


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
