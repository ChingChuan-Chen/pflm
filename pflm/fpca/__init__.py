"""Utilities to help with functional data analysis."""

# Authors: Ching-Chuan Chen
# SPDX-License-Identifier: MIT

from pflm.fpca.covariance_utils import get_covariance_matrix, get_measurement_error_variance, get_raw_cov, rotate_polyfit2d
from pflm.fpca.fpca_base_func_utils import get_eigen_analysis_results, get_fpca_phi, select_num_pcs_fve
from pflm.fpca.fpca_result_class import FpcaModelParams, SmoothedModelResult
from pflm.fpca.fpca_score_utils import estimate_rho, get_eigenvalue_fit, get_fpca_ce_score, get_fpca_in_score
from pflm.fpca.functional_data_generator import FunctionalDataGenerator
from pflm.fpca.functional_pca import FunctionalPCA, FunctionalPCAMuCovParams, FunctionalPCAUserDefinedParams

__all__ = [
    "FpcaModelParams",
    "FunctionalDataGenerator",
    "FunctionalPCA",
    "FunctionalPCAMuCovParams",
    "FunctionalPCAUserDefinedParams",
    "SmoothedModelResult",
    "estimate_rho",
    "get_covariance_matrix",
    "get_eigen_analysis_results",
    "get_eigenvalue_fit",
    "get_fpca_ce_score",
    "get_fpca_in_score",
    "get_fpca_phi",
    "get_measurement_error_variance",
    "get_raw_cov",
    "rotate_polyfit2d",
    "select_num_pcs_fve",
]
