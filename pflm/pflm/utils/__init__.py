"""Utility functions for partial functional linear model"""

# Authors: Ching-Chuan Chen
# SPDX-License-Identifier: MIT

from pflm.pflm.utils.elastic_net_solver import (
    fit_gaussian_f32,
    fit_gaussian_f64,
    fit_multinomial_f32,
    fit_multinomial_f64,
    fit_nongaussian_f32,
    fit_nongaussian_f64,
)
from pflm.pflm.utils.linear_model import ElasticNet, LinearModelFamily

__all__ = [
    "ElasticNet",
    "LinearModelFamily",
    "fit_gaussian_f32",
    "fit_gaussian_f64",
    "fit_multinomial_f32",
    "fit_multinomial_f64",
    "fit_nongaussian_f32",
    "fit_nongaussian_f64",
]
