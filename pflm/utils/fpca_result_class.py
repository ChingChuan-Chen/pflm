"""The classes to save the results for FPCA fitting"""

# Authors: Ching-Chuan Chen
# SPDX-License-Identifier: MIT
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Union

import numpy as np


@dataclass
class SmoothedModelResult:
    grid: np.ndarray
    mu: np.ndarray
    cov: np.ndarray
    grid_type: Literal["obs", "reg"] = "obs"


@dataclass
class FpcaModelParams:
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
