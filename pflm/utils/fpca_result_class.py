"""The classes to save the results for FPCA fitting"""

# Authors: Ching-Chuan Chen
# SPDX-License-Identifier: MIT
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np


@dataclass
class SmoothedModelResult:
    grid: np.ndarray
    mu: np.ndarray
    cov: np.ndarray
    phi: Optional[np.ndarray] = None
    fitted_cov: Optional[np.ndarray] = None
    grid_type: Literal["obs", "reg"] = "obs"


@dataclass
class FpcaModelResult:
    xi: Optional[np.ndarray]
    xi_var: Optional[np.ndarray]
    num_pcs: int
    fpca_lambda: np.ndarray
    eigen_results: dict
    rho: Optional[float]
