"""The classes to save the results for FPCA fitting"""

# Authors: Ching-Chuan Chen
# SPDX-License-Identifier: MIT
from dataclasses import dataclass
from typing import Dict, Literal, Optional, Union

import numpy as np


@dataclass
class SmoothedModelResult:
    """Smoothing outputs on a grid.

    Attributes
    ----------
    grid : np.ndarray of shape (nt,)
    mu : np.ndarray of shape (nt,)
    cov : np.ndarray of shape (nt, nt)
    grid_type : {"obs", "reg"}
        Grid kind: observation or regular grid.
    """

    grid: np.ndarray
    mu: np.ndarray
    cov: np.ndarray
    grid_type: Literal["obs", "reg"] = "obs"


@dataclass
class FpcaModelParams:
    """FPCA parameters and artifacts.

    Attributes
    ----------
    measurement_error_variance : float
    eigen_results : dict
        Keys like {"lambda": np.ndarray, "vector": np.ndarray}.
    select_num_pcs_criterion : np.ndarray or None
    fpca_lambda : np.ndarray or None
    fpca_phi : dict or None
        Keys like {"obs": np.ndarray, "reg": np.ndarray}.
    num_pcs : int or None
    fitted_covariance : dict or None
        Keys like {"obs": np.ndarray, "reg": np.ndarray}.
    rho : float or None
    eigenvalue_fit : np.ndarray or None
    method_select_num_pcs : int or {"FVE","AIC","BIC"} or None
    max_num_pcs : int or None
    method_pcs : {"IN","CE"} or None
    method_rho : {"trunc","ridge","vanilla"} or None
    if_shrinkage : bool or None
    fve_threshold : float or None
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
