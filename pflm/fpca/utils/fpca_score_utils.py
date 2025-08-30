"""Utility functions used for FPCA"""

# Authors: Ching-Chuan Chen
# SPDX-License-Identifier: MIT
from typing import List, Tuple

import numpy as np

from pflm.fpca.utils.fpca_score import fpca_ce_score_f32, fpca_ce_score_f64, fpca_in_score_f32, fpca_in_score_f64
from pflm.utils.blas_helper import BLAS_Trans
from pflm.utils.lapack_helper import _gels_memview_f32, _gels_memview_f64
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
    Compute conditional-expectation (CE) FPCA scores and fitted curves.

    Parameters
    ----------
    flatten_func_data : FlattenFunctionalData
        Flattened data containing fields y, t, tid, unique_sid, sid_cnt.
    mu : np.ndarray of shape (nt,)
        Mean on the observation grid.
    num_pcs : int
        Number of principal components to use (<= len(fpca_lambda)).
    fpca_lambda : np.ndarray of shape (k,)
        Eigenvalues for FPCA components.
    fpca_phi : np.ndarray of shape (nt, k)
        Basis functions on the observation grid (columns are components).
    fitted_cov : np.ndarray of shape (nt, nt)
        Fitted covariance on the observation grid.
    sigma2 : float
        Measurement noise variance.

    Returns
    -------
    xi : np.ndarray of shape (n_samples, num_pcs)
        CE scores by subject.
    xi_var : List[np.ndarray]
        Per-subject score covariance matrices or variance summaries.
    fitted_y_mat : np.ndarray of shape (nt, n_samples)
        Fitted values on the observation grid.
    fitted_y : List[np.ndarray]
        Fitted values at the observed time points per subject.

    Raises
    ------
    ValueError
        If `fitted_cov` has wrong shape, `num_pcs` exceeds available eigenvalues,
        or `fpca_phi` has incompatible shape.

    See Also
    --------
    get_fpca_in_score : In-sample (projection-based) score computation.
    """
    nt = flatten_func_data.unique_tid.size
    if fitted_cov.shape != (nt, nt):
        raise ValueError("fitted_cov must have shape (nt, nt).")
    if num_pcs > fpca_lambda.size:
        raise ValueError("num_pcs must be less than or equal to the number of eigenvalues.")
    if fpca_phi.shape != (nt, fpca_lambda.size):
        raise ValueError("fpca_phi must have shape (nt, fpca_lambda.size).")

    input_dtype = flatten_func_data.y.dtype
    sigma_y = fitted_cov.astype(input_dtype, copy=False)
    if sigma2 > 0.0:
        np.fill_diagonal(sigma_y, np.diagonal(sigma_y) + sigma2)
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
    Estimate the optimal rho parameter for CE scoring.

    Parameters
    ----------
    method_rho : {"ridge", "truncated"}
        Estimation strategy for rho.
    flatten_func_data : FlattenFunctionalData
        Flattened data (y, t, tid, unique_sid, sid_cnt).
    reg_grid : np.ndarray of shape (n_reg_grid,)
        Regular grid used to compute variance of reconstructed curves.
    mu_obs : np.ndarray of shape (nt,)
        Mean on the observation grid.
    mu_reg : np.ndarray of shape (n_reg_grid,)
        Mean on the regular grid.
    fpca_lambda : np.ndarray of shape (k,)
        FPCA eigenvalues.
    fpca_phi_obs : np.ndarray of shape (nt, k)
        Basis on observation grid.
    fpca_phi_reg : np.ndarray of shape (n_reg_grid, k)
        Basis on regular grid.
    fitted_cov : np.ndarray of shape (nt, nt)
        Fitted covariance on observation grid.
    sigma2 : float
        Noise variance or starting value (for "truncated" path).

    Returns
    -------
    float
        Estimated rho value that best matches target variance.

    Notes
    -----
    Values of `method_rho` other than "ridge" are treated as "truncated".
    """
    num_pcs = fpca_lambda.size
    obs_grid = flatten_func_data.unique_tid
    total_fpca_lambda = np.sum(fpca_lambda)
    if method_rho == "ridge":
        min_rho_power = -13 if fpca_lambda.dtype == np.float64 else -9
        r = np.sqrt((trapz(mu_obs**2, obs_grid) + total_fpca_lambda) / (obs_grid[-1] - obs_grid[0]))
        rho_candidates = np.exp(np.linspace(min_rho_power, -1.5, 50)) * r
    else:
        order = "F" if fpca_phi_reg.flags.f_contiguous else "C"
        idx_vec = np.zeros(flatten_func_data.y.shape[0], dtype=np.int64)
        if order == "C":
            idx_vec = flatten_func_data.tid * flatten_func_data.unique_tid.size + flatten_func_data.sid
        elif order == "F":
            idx_vec = flatten_func_data.sid * flatten_func_data.unique_tid.size + flatten_func_data.tid
        for _ in range(2):
            _, _, fitted_y_mat, _ = get_fpca_ce_score(flatten_func_data, mu_obs, num_pcs, fpca_lambda, fpca_phi_obs, fitted_cov, sigma2)
            squared_residuals = (fitted_y_mat.ravel(order=order)[idx_vec] - flatten_func_data.y) ** 2
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
    Compute in-sample (projection-based) FPCA scores and fitted curves.

    Parameters
    ----------
    flatten_func_data : FlattenFunctionalData
        Flattened data containing fields y, t, tid, unique_sid, sid_cnt.
    mu : np.ndarray of shape (nt,)
        Mean on the observation grid.
    num_pcs : int
        Number of principal components to use (<= len(fpca_lambda)).
    fpca_lambda : np.ndarray of shape (k,)
        FPCA eigenvalues.
    fpca_phi : np.ndarray of shape (nt, k)
        Basis on the observation grid (columns are components).
    sigma2 : float
        Measurement noise variance used in shrinkage (if enabled).
    if_shrinkage : bool, default=False
        Whether to apply shrinkage to the IN scores.

    Returns
    -------
    xi : np.ndarray of shape (n_samples, num_pcs)
        IN scores by subject.
    xi_var : List[np.ndarray]
        Per-subject score covariance matrices or variance summaries.
    fitted_y_mat : np.ndarray of shape (nt, n_samples)
        Fitted values on the observation grid.
    fitted_y : List[np.ndarray]
        Fitted values at observed time points per subject.

    Raises
    ------
    ValueError
        If `num_pcs` exceeds available eigenvalues, `fpca_phi` has incompatible
        shape, or `if_shrinkage` is not a boolean.

    See Also
    --------
    get_fpca_ce_score : Conditional-expectation score computation.
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
    Fit eigenvalues by projecting raw covariance onto the FPCA subspace.

    Parameters
    ----------
    raw_cov : np.ndarray of shape (M, 5)
        Raw covariance entries (sid, t1, t2, w, cov).
    obs_grid : np.ndarray of shape (nt,)
        Sorted observation grid.
    fpca_phi_obs : np.ndarray of shape (nt, num_pcs)
        Basis on the observation grid (only the first `num_pcs` columns used).
    num_pcs : int
        Number of principal components to fit.

    Returns
    -------
    np.ndarray of shape (num_pcs,)
        Fitted eigenvalues in the FPCA subspace.

    Raises
    ------
    ValueError
        If `fpca_phi_obs` has incompatible shape.

    Notes
    -----
    The least-squares solve is performed with a low-level LAPACK GELS routine.
    """
    nt = obs_grid.size
    if fpca_phi_obs.shape != (nt, num_pcs):
        raise ValueError("fpca_phi_obs must have shape (nt, num_pcs).")

    mask = raw_cov[:, 1] != raw_cov[:, 2]
    tid1 = np.searchsorted(obs_grid, raw_cov[mask, 1])
    tid2 = np.searchsorted(obs_grid, raw_cov[mask, 2])
    ev_fit_x = np.sqrt(raw_cov[mask, 3]).reshape(-1, 1) * fpca_phi_obs[tid1, :num_pcs] * fpca_phi_obs[tid2, :num_pcs]  # shape (n_pairs, num_pcs)
    order_x = "F" if ev_fit_x.flags.f_contiguous else "C"
    ev_fit_y = (np.sqrt(raw_cov[mask, 3]) * raw_cov[mask, 4]).reshape(-1, 1).astype(dtype=ev_fit_x.dtype, order=order_x)
    gles_func = _gels_memview_f64 if ev_fit_x.dtype == np.float64 else _gels_memview_f32
    gles_func(BLAS_Trans.NoTrans, ev_fit_x, ev_fit_y, ev_fit_x.shape[0], ev_fit_x.shape[1], 1)
    return ev_fit_y[:num_pcs].ravel(order=order_x)
