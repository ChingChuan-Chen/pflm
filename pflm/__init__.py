"""Configure global settings and get information about the working environment."""

# Authors: Ching-Chuan Chen
# SPDX-License-Identifier: MIT

# Partial Functional Linear Models (pflm) for Python
# ==================================================
#
# pflm is a Python package for functional linear models, inspired by the MATLAB PACE (Principal Analysis by Conditional Expectation)
# package and the R fdapace package.
#
# It aims to provide an efficient and user-friendly interface for functional data analysis.
# The package includes functionalities for functional principal component analysis (FPCA) and partial functional linear models (PFLM).
# This package is designed to leverage scikit-learn's interface and utilities, making it easy to integrate with other machine learning workflows.

import importlib as _importlib
import logging

logger = logging.getLogger(__name__)


# PEP0440 compatible formatted version, see:
# https://www.python.org/dev/peps/pep-0440/
#
# Generic release markers:
#   X.Y.0   # For first release after an increment in Y
#   X.Y.Z   # For bugfix releases
#
# Admissible pre-release markers:
#   X.Y.ZaN   # Alpha release
#   X.Y.ZbN   # Beta release
#   X.Y.ZrcN  # Release Candidate
#   X.Y.Z     # Final release
#
# Dev branch marker is: 'X.Y.dev' or 'X.Y.devN' where N is an integer.
# 'X.Y.dev0' is the canonical version of 'X.Y.dev'

__version__ = "0.1.0.dev0"

from pflm.functional_data_generator import FunctionalDataGenerator  # noqa: F401 E402

_submodules = [
    "interp",
    "smooth",
    "FunctionalDataGenerator",
    "utils",
]

__all__ = _submodules + []


def __dir__():
    return __all__


def __getattr__(name):
    if name in _submodules:
        return _importlib.import_module(f"pflm.{name}")
    else:
        try:
            return globals()[name]
        except KeyError:
            raise AttributeError(f"Module 'pflm' has no attribute '{name}'")
