"""The classes to save the results for FPCA fitting"""

# Authors: Ching-Chuan Chen
# SPDX-License-Identifier: MIT
from dataclasses import dataclass
from typing import Dict, Literal, Optional, Union

import numpy as np


@dataclass
class SmoothedModelResult:
    """Smoothed mean/covariance on a specific grid.

    Attributes
    ----------
    grid : np.ndarray of shape (nt,)
        The grid on which the smoothing results are defined.
    mu : np.ndarray of shape (nt,)
        Smoothed mean values on `grid`.
    cov : np.ndarray of shape (nt, nt)
        Smoothed covariance matrix on `grid`.
    grid_type : {"obs", "reg"}
        Grid kind: observation grid ("obs") or regular grid ("reg").

    Notes
    -----
    This dataclass is a container with no validation logic; shapes and
    consistency are assumed to be checked upstream.
    """

    grid: np.ndarray
    mu: np.ndarray
    cov: np.ndarray
    grid_type: Literal["obs", "reg"] = "obs"


@dataclass
class FpcaModelParams:
    """FPCA parameters, artifacts, and tuning metadata.

    Attributes
    ----------
    measurement_error_variance : float
        Estimated noise variance (sigma^2).
    eigen_results : dict
        Eigen decomposition results with keys like
        {"lambda": np.ndarray, "vector": np.ndarray}.
    select_num_pcs_criterion : np.ndarray, optional
        Criterion values used in selecting the number of PCs (e.g., FVE curve or
        information criteria).
    fpca_lambda : np.ndarray, optional
        Selected/processed eigenvalues for FPCA.
    fpca_phi : dict, optional
        Basis functions on different grids, e.g., {"obs": ..., "reg": ...}.
    num_pcs : int, optional
        Number of retained PCs.
    fitted_covariance : dict, optional
        Fitted covariance matrices by grid kind, e.g., {"obs": ..., "reg": ...}.
    rho : float, optional
        Truncation/ridge/vanilla parameter used in CE scoring if applicable.
    eigenvalue_fit : np.ndarray, optional
        Alternative eigenvalue estimates from projection-based fitting.
    method_select_num_pcs : int or {"FVE","AIC","BIC"}, optional
        Selection method or a fixed number of PCs.
    max_num_pcs : int, optional
        Upper bound used when searching the number of PCs.
    method_pcs : {"IN","CE"}, optional
        Score estimation method (In-sample or Conditional Expectation).
    method_rho : {"trunc","ridge","vanilla"}, optional
        Strategy for rho selection in CE.
    if_shrinkage : bool, optional
        Whether shrinkage was applied to IN scores.
    fve_threshold : float, optional
        Target FVE used when selecting the number of PCs.

    Notes
    -----
    This dataclass is a passive container; validation and consistency checks
    should be handled by the FPCA fitting pipeline.
    """

    measurement_error_variance: float
    eigen_results: Dict[str, np.ndarray]
    select_num_pcs_criterion: Optional[np.ndarray] = None
    fpca_lambda: Optional[np.ndarray] = None
    fpca_phi: Optional[Dict[str, np.ndarray]] = None
    num_pcs: Optional[int] = None
    fitted_covariance: Optional[Dict[str, np.ndarray]] = None
    rho: Optional[float] = None
    eigenvalue_fit: Optional[np.ndarray] = None
    method_select_num_pcs: Optional[Union[int, Literal["FVE", "AIC", "BIC"]]] = None
    max_num_pcs: Optional[int] = None
    method_pcs: Optional[Literal["IN", "CE"]] = None
    method_rho: Optional[Literal["trunc", "ridge", "vanilla"]] = None
    if_shrinkage: Optional[bool] = None
    fve_threshold: Optional[float] = None
