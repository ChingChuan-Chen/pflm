"""Utility functions used for FPCA"""

# Authors: Ching-Chuan Chen
# SPDX-License-Identifier: MIT
from typing import List, Tuple

import numpy as np

from pflm.utils._fpca_score import fpca_ce_score_f32, fpca_ce_score_f64, fpca_in_score_f32, fpca_in_score_f64
from pflm.utils._lapack_helper import _gels_memview_f32, _gels_memview_f64
from pflm.utils.utility import FlattenFunctionalData, trapz


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
    xi, xi_var, fitted_y_mat, fitted_y = fpca_ce_score_func(
        flatten_func_data.y,
        flatten_func_data.t,
        flatten_func_data.tid,
        mu,
        sigma_y,
        fpca_lambda,
        fpca_phi,
        flatten_func_data.unique_sid,
        flatten_func_data.sid_cnt,
    )
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
    t_range = flatten_func_data.unique_tid[-1] - flatten_func_data.unique_tid[0]
    xi, fitted_y_mat, fitted_y = fpca_in_score_func(
        flatten_func_data.y,
        flatten_func_data.t,
        flatten_func_data.tid,
        mu,
        fpca_lambda,
        fpca_phi,
        flatten_func_data.unique_sid,
        flatten_func_data.sid_cnt,
        sigma2,
        t_range,
        if_shrinkage,
    )
    xi_var = [np.zeros((num_pcs, num_pcs), dtype=input_dtype) for _ in flatten_func_data.sid_cnt]
    return xi, xi_var, fitted_y_mat, fitted_y


def get_eigenvalue_fit(raw_cov: np.ndarray, obs_grid: np.ndarray, fpca_phi_obs: np.ndarray, num_pcs: int):
    """
    Get the fitted eigenvalues for the functional principal components.

    Parameters
    ----------
    raw_cov : np.ndarray
        The raw_cov matrix obtained from `get_raw_cov` function.
    obs_grid : np.ndarray
        The observation grid.
    fpca_phi_obs : np.ndarray
        The functional principal component basis functions.
    num_pcs : int
        The number of principal components.

    Returns
    -------
    np.ndarray
        The fitted eigenvalues.
    """
    nt = obs_grid.size
    if fpca_phi_obs.shape != (nt, num_pcs):
        raise ValueError("fpca_phi_obs must have shape (nt, num_pcs).")

    mask = raw_cov[:, 1] != raw_cov[:, 2]
    tid1 = np.searchsorted(obs_grid, raw_cov[mask, 1])
    tid2 = np.searchsorted(obs_grid, raw_cov[mask, 2])
    ev_fit_x = np.sqrt(raw_cov[mask, 3]).reshape(-1, 1) * fpca_phi_obs[tid1, :num_pcs] * fpca_phi_obs[tid2, :num_pcs]  # shape (n_pairs, num_pcs)
    ev_fit_y = np.sqrt(raw_cov[mask, 3]) * raw_cov[mask, 4]
    gles_func = _gels_memview_f64 if ev_fit_x.dtype == np.float64 else _gels_memview_f32
    gles_func(ev_fit_x.ravel(order="F"), ev_fit_y, ev_fit_x.shape[0], ev_fit_x.shape[1], 1, ev_fit_x.shape[0], ev_fit_y.shape[0])
    return ev_fit_y[:num_pcs]
