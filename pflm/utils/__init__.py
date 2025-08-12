"""Utilities to help with functional data analysis."""

# Authors: Ching-Chuan Chen
# SPDX-License-Identifier: MIT
from pflm.utils.trapz import trapz
from pflm.utils.utility import flatten_and_sort_data_matrices, get_covariance_matrix, get_eigen_results, get_raw_cov, rotate_polyfit2d

__all__ = ["flatten_and_sort_data_matrices", "get_covariance_matrix", "get_eigen_results", "get_raw_cov", "rotate_polyfit2d", "trapz"]
