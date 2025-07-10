"""This module provides covariance smoothing functions for pflm."""

# Authors: Ching-Chuan Chen
# SPDX-License-Identifier: MIT

import numpy as np

from pflm.smooth.kernel import KernelType


def cov_smooth(
    x_grid: np.ndarray,
    covs: np.ndarray,
    weights: np.ndarray,
    obs_grid: np.ndarray,
    reg_grid: np.ndarray,
    bandwidth: float,
    kernel: KernelType = KernelType.GAUSSIAN,
):
    pass


def diag_cov_smooth(
    x_grid: np.ndarray,
    covs: np.ndarray,
    weights: np.ndarray,
    reg_grid: np.ndarray,
    bandwidth: float,
    kernel: KernelType = KernelType.GAUSSIAN,
):
    pass
