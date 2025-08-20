"""Utilities to help with functional data analysis."""

# Authors: Ching-Chuan Chen
# SPDX-License-Identifier: MIT
from pflm.utils.covariance_utils import get_covariance_matrix, get_measurement_error_variance, get_raw_cov, rotate_polyfit2d
from pflm.utils.fpca_helpers import (
    estimate_rho,
    get_eigen_analysis_results,
    get_fpca_ce_score,
    get_fpca_in_score,
    get_fpca_phi,
    select_num_pcs_fve,
)
from pflm.utils.fpca_result_class import FpcaModelResult, SmoothedModelResult
from pflm.utils.utility import flatten_and_sort_data_matrices, trapz, FlattenFunctionalData

__all__ = [
    "FlattenFunctionalData",
    "FpcaModelResult",
    "SmoothedModelResult",
    "estimate_rho",
    "flatten_and_sort_data_matrices",
    "get_covariance_matrix",
    "get_eigen_analysis_results",
    "get_fpca_ce_score",
    "get_fpca_in_score",
    "get_fpca_phi",
    "get_measurement_error_variance",
    "get_raw_cov",
    "rotate_polyfit2d",
    "select_num_pcs_fve",
    "trapz",
]
