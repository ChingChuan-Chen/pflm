"""Utilities to help with functional data analysis."""

# Authors: Ching-Chuan Chen
# SPDX-License-Identifier: MIT
from pflm.utils.trapz import trapz
from pflm.utils.utility import flatten_data_matrix, get_eigen_results

__all__ = ["flatten_data_matrix", "get_eigen_results", "trapz"]
