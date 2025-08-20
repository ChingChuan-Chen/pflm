"""The classes to save the results for FPCA fitting"""

# Authors: Ching-Chuan Chen
# SPDX-License-Identifier: MIT
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

import numpy as np


@dataclass
class SmoothedModelResult:
    grid: np.ndarray
    mu: np.ndarray
    cov: np.ndarray
    grid_type: Literal["obs", "reg"] = "obs"


@dataclass
class FpcaEigenFunction:
    grid: np.ndarray
    value: np.ndarray
    grid_type: Literal["obs", "reg"] = "obs"


@dataclass
class FpcaFittedCovariance:
    grid: np.ndarray
    value: np.ndarray
    grid_type: Literal["obs", "reg"] = "obs"


@dataclass
class FpcaModelParams:
    measurement_error_variance: float
    eigen_results: Dict[str, np.ndarray]
    select_num_pcs_result: Dict[str, Any]
    method_rho: str
    fpca_lambda: Optional[np.ndarray] = None
    fpca_phi: Optional[List[FpcaEigenFunction]] = None
    num_pcs: Optional[int] = None
    fitted_covariance: Optional[List[FpcaFittedCovariance]] = None
    rho: Optional[float] = None
    eigenvalue_fit: Optional[np.ndarray] = None
