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
