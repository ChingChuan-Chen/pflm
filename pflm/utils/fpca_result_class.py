"""The classes to save the results for FPCA fitting"""

# Authors: Ching-Chuan Chen
# SPDX-License-Identifier: MIT
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np


@dataclass
class SmoothedModelResult:
    """Hold smoothed results for a single grid (observation or regular)."""

    grid: np.ndarray
    mu: np.ndarray
    cov: np.ndarray
    phi: Optional[np.ndarray] = None
    fitted_cov: Optional[np.ndarray] = None
    grid_type: Literal["obs", "reg"] = "obs"

    def __repr__(self):
        return f"SmoothedModelResult(grid_type={self.grid_type!r}, n_grid={self.grid.size})"


@dataclass
class FpcaModelResult:
    """Hold FPCA scores and eigen decomposition results."""

    xi: Optional[np.ndarray]
    xi_var: Optional[np.ndarray]
    num_pcs: int
    fpca_lambda: np.ndarray
    eigen_results: dict
    rho: Optional[float]

    def __repr__(self):
        return (
            f"FpcaModelResult(num_pcs={self.num_pcs}, "
            f"xi_shape={None if self.xi is None else self.xi.shape}, "
            f"fpca_lambda_shape={self.fpca_lambda.shape}, rho={self.rho})"
        )


class FlattenFunctionalData:
    t: np.ndarray
    y: np.ndarray
    w: np.ndarray
    unique_tid: np.ndarray
    inverse_tid: np.ndarray
    tid: np.ndarray
    sid: np.ndarray
    unique_sid: np.ndarray
    sid_cnt: np.ndarray

    def __init__(self, y: np.ndarray, t: np.ndarray, w: np.ndarray, sid: np.ndarray):
        self.y = y
        self.t = t
        self.sid = sid
        self.w = w
        self.unique_tid, self.inverse_tid = np.unique(t, return_inverse=True, sorted=True)
        self.tid = np.digitize(self.t, self.unique_tid, right=True)
        self.unique_sid, self.sid_cnt = np.unique(sid, return_counts=True)

    def __repr__(self):
        return f"FlattenFunctionalData(total_size={self.t.size}, num_samples={self.unique_sid.size}, nt={self.unique_tid.size})"
