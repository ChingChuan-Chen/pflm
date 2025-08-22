"""Utility functions used for FPCA"""

# Authors: Ching-Chuan Chen
# SPDX-License-Identifier: MIT
import copy
import warnings
from typing import List, Tuple

import numpy as np

from pflm.utils._fpca_score import fpca_ce_score_f32, fpca_ce_score_f64, fpca_in_score_f32, fpca_in_score_f64
from pflm.utils._lapack_helper import _syevd_memview_f32, _syevd_memview_f64
from pflm.utils.utility import FlattenFunctionalData, trapz


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
    eig_lambda = eig_lambda[mask][ord_idx]
    eig_vector = eig_vector.reshape(nt, -1).T[:, mask][:, ord_idx]
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
    flatten_func_data: FlattenFunctionalData,
    mu: np.ndarray,
    num_pcs: int,
    fpca_lambda: np.ndarray,
    fpca_phi: np.ndarray,
    fitted_cov: np.ndarray,
    sigma2: float,
) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
    """
    Compute the functional principal component analysis (FPCA) scores and fitted values.

    Parameters
    ----------
    flatten_func_data : FlattenFunctionalData
        The flattened functional data containing y, t, tid, unique_sid, and sid_cnt.
    mu : np.ndarray
        The mean function values at the grid points with shape (nt,).
    num_pcs : int
        The number of principal components to compute.
    fpca_lambda : np.ndarray
        The eigenvalues corresponding to the functional principal components.
    fpca_phi : np.ndarray
        The functional principal component basis functions of shape (nt, num_pcs).
    fitted_cov : np.ndarray
        The fitted covariance matrix of shape (nt, nt).
    sigma2 : float
        The noise variance.

    Returns
    -------
    xi : np.ndarray
        The FPCA scores of shape (num_samples, num_pcs).
    xi_var : np.ndarray
        The variances of the FPCA scores of shape (num_samples, num_pcs).
    fitted_y_mat : np.ndarray
        The fitted functional data values of shape (nt, num_samples).
    fitted_y : List[np.ndarray]
        The fitted functional data values for each unique subject ID.
    """
    nt = flatten_func_data.unique_tid.size
    if fitted_cov.shape != (nt, nt):
        raise ValueError("fitted_cov must have shape (nt, nt).")
    if num_pcs > fpca_lambda.size:
        raise ValueError("num_pcs must be less than or equal to the number of eigenvalues.")
    if fpca_phi.shape != (nt, fpca_lambda.size):
        raise ValueError("fpca_phi must have shape (nt, fpca_lambda.size).")

    input_dtype = flatten_func_data.y.dtype
    sigma_y = (fitted_cov + np.eye(fitted_cov.shape[0]) * sigma2).astype(input_dtype, copy=False)
    fpca_ce_score_func = fpca_ce_score_f64 if input_dtype == np.float64 else fpca_ce_score_f32
    lambda_phi = np.ascontiguousarray(fpca_phi @ np.diag(fpca_lambda)).astype(input_dtype, copy=False)
    xi, xi_var = fpca_ce_score_func(
        flatten_func_data.y,
        flatten_func_data.t,
        flatten_func_data.tid,
        mu,
        sigma_y,
        fpca_lambda,
        lambda_phi,
        flatten_func_data.unique_sid,
        flatten_func_data.sid_cnt,
    )
    fitted_y_mat = mu.reshape(-1, 1) + fpca_phi[:, :num_pcs] @ xi.T  # (nt, n_samples)
    fitted_y = [fitted_y_mat[flatten_func_data.tid[flatten_func_data.sid == i], i] for i in flatten_func_data.unique_sid]
    return xi, xi_var, fitted_y_mat, fitted_y


def estimate_rho(
    method_rho: str,
    flatten_func_data: FlattenFunctionalData,
    reg_grid: np.ndarray,
    mu_obs: np.ndarray,
    mu_reg: np.ndarray,
    fpca_lambda: np.ndarray,
    fpca_phi_obs: np.ndarray,
    fpca_phi_reg: np.ndarray,
    fitted_cov: np.ndarray,
    sigma2: float,
) -> float:
    """
    Estimate the optimal rho parameter for the FPCA model.

    Parameters
    ----------
    method_rho : str
        The method for estimating rho, either 'ridge' or 'trunc'.
    flatten_func_data : FlattenFunctionalData
        The flattened functional data containing y, t, tid, unique_sid, and sid_cnt.
    reg_grid : np.ndarray
        The registration grid points with shape (n_reg_grid,).
    mu_obs : np.ndarray
        The mean function values at the grid points with shape (nt,).
    mu_reg : np.ndarray
        The mean function values at the grid points for the registration data with shape (n_reg_grid,).
    fpca_lambda : np.ndarray
        The eigenvalues corresponding to the functional principal components.
    fpca_phi_obs : np.ndarray
        The functional principal component basis functions for the observed data of shape (nt, num_pcs).
    fpca_phi_reg : np.ndarray
        The functional principal component basis functions for the registration data of shape (n_reg_grid, num_pcs).
    fitted_cov : np.ndarray
        The fitted covariance matrix for the observed data of shape (nt, nt).
    sigma2 : float
        The noise variance.

    Returns
    -------
    float
        The estimated optimal rho value.
    """
    num_pcs = fpca_lambda.size
    obs_grid = flatten_func_data.unique_tid
    total_fpca_lambda = np.sum(fpca_lambda)
    if method_rho == "ridge":
        r = np.sqrt((trapz(mu_obs**2, obs_grid) + total_fpca_lambda) / (obs_grid[-1] - obs_grid[0]))
        rho_candidates = np.exp(np.linspace(-13, -1.5, 50)) * r
    else:
        for _ in range(2):
            _, _, fitted_y_mat, _ = get_fpca_ce_score(flatten_func_data, mu_obs, num_pcs, fpca_lambda, fpca_phi_obs, fitted_cov, sigma2)
            idx = flatten_func_data.tid * flatten_func_data.unique_tid.size + flatten_func_data.sid
            squared_residuals = (fitted_y_mat.ravel(order="C")[idx] - flatten_func_data.y) ** 2
            sigma2 = np.mean(np.bincount(flatten_func_data.sid, weights=squared_residuals) / flatten_func_data.sid_cnt)
        rho_candidates = np.linspace(1, 10, 50) * sigma2

    # calculate rho scores
    rho_scores = np.zeros(len(rho_candidates), dtype=np.float64)
    for idx, rho in enumerate(rho_candidates):
        xi, _, _, _ = get_fpca_ce_score(flatten_func_data, mu_obs, num_pcs, fpca_lambda, fpca_phi_obs, fitted_cov, sigma2=rho)
        fitted_y_mat_reg = mu_reg.reshape(-1, 1) + fpca_phi_reg[:, :num_pcs] @ xi.T
        var_y = np.var(fitted_y_mat_reg, axis=1, ddof=1)  # variance across samples per time point
        rho_scores[idx] = (total_fpca_lambda - trapz(var_y, reg_grid)) ** 2
    return rho_candidates[np.argmin(rho_scores)]


def get_fpca_in_score(
    flatten_func_data: FlattenFunctionalData,
    mu: np.ndarray,
    num_pcs: int,
    fpca_lambda: np.ndarray,
    fpca_phi: np.ndarray,
    sigma2: float,
    if_shrinkage: bool = False,
) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray, List[np.ndarray]]:
    """
    Compute the functional principal component analysis (FPCA) scores and fitted values.

    Parameters
    ----------
    flatten_func_data : FlattenFunctionalData
        The flattened functional data containing y, t, tid, unique_sid, and sid_cnt.
    mu : np.ndarray
        The mean function values at the grid points with shape (nt,).
    num_pcs : int
        The number of principal components to compute.
    fpca_lambda : np.ndarray
        The eigenvalues corresponding to the functional principal components.
    fpca_phi : np.ndarray
        The functional principal component basis functions of shape (nt, num_pcs).
    sigma2 : float
        The noise variance.
    if_shrinkage : bool
        Whether to apply shrinkage to the FPCA scores.

    Returns
    -------
    xi : np.ndarray
        The FPCA scores of shape (num_samples, num_pcs).
    xi_var : np.ndarray
        The variances of the FPCA scores of shape (num_samples, num_pcs).
    fitted_y_mat : np.ndarray
        The fitted functional data values of shape (nt, num_samples).
    fitted_y : List[np.ndarray]
        The fitted functional data values for each unique subject ID.
    """
    nt = flatten_func_data.unique_tid.size
    if num_pcs > fpca_lambda.size:
        raise ValueError("num_pcs must be less than or equal to the number of eigenvalues.")
    if fpca_phi.shape != (nt, fpca_lambda.size):
        raise ValueError("fpca_phi must have shape (nt, fpca_lambda.size).")
    if not isinstance(if_shrinkage, bool):
        raise ValueError("if_shrinkage must be a boolean.")

    input_dtype = flatten_func_data.y.dtype
    fpca_in_score_func = fpca_in_score_f64 if input_dtype == np.float64 else fpca_in_score_f32
    fpca_phi_ = np.ascontiguousarray(fpca_phi).astype(input_dtype, copy=False)
    t_range = flatten_func_data.unique_tid[-1] - flatten_func_data.unique_tid[0]
    xi = fpca_in_score_func(
        flatten_func_data.y,
        flatten_func_data.t,
        flatten_func_data.tid,
        mu,
        fpca_lambda,
        fpca_phi_,
        flatten_func_data.unique_sid,
        flatten_func_data.sid_cnt,
        sigma2,
        t_range,
        if_shrinkage,
    )
    xi_var = [np.zeros((num_pcs, num_pcs), dtype=input_dtype) for data_cnt in flatten_func_data.sid_cnt]
    fitted_y_mat = mu.reshape(-1, 1) + fpca_phi[:, :num_pcs] @ xi.T  # (nt, n_samples)
    fitted_y = [fitted_y_mat[flatten_func_data.tid[flatten_func_data.sid == i], i] for i in flatten_func_data.unique_sid]
    return xi, xi_var, fitted_y_mat, fitted_y
