"""Utilities to help with functional data analysis."""

# Authors: Ching-Chuan Chen
# SPDX-License-Identifier: MIT
from pflm.utils.covariance_utils import get_covariance_matrix, get_measurement_error_variance, get_raw_cov, rotate_polyfit2d
from pflm.utils.utility import flatten_and_sort_data_matrices, get_eigen_results, trapz

__all__ = [
    "flatten_and_sort_data_matrices",
    "get_covariance_matrix",
    "get_eigen_results",
    "get_measurement_error_variance",
    "get_raw_cov",
    "rotate_polyfit2d",
    "trapz",
]
