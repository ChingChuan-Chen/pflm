"""Kernel types for local regression in pflm."""

# Authors: Ching-Chuan Chen
# SPDX-License-Identifier: MIT

from enum import Enum


class KernelType(Enum):
    """Enum for kernel types used in local regression."""

    GAUSSIAN = 0
    LOGISTIC = 1
    SIGMOID = 2
    GAUSSIAN_VAR = 3
    RECTANGULAR = 100  # Uniform Kernel
    TRIANGULAR = 101
    EPANECHNIKOV = 102
    BIWEIGHT = 103  # Quartic Kernel
    TRIWEIGHT = 104
    TRICUBE = 105
    COSINE = 106
    # SILVERMAN = 107 is not included since it might produce negative weights that are not supported in our implementation.

    def __repr__(self):
        return f"KernelType.{self.name}"

    def __str__(self):
        return self.name
