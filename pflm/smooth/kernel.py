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
    RECTANGULAR = 4  # Uniform Kernel
    TRIANGULAR = 5
    EPANECHNIKOV = 6
    BIWEIGHT = 7     # Quartic Kernel
    TRIWEIGHT = 8
    TRICUBE = 9
    COSINE = 10

    def __repr__(self):
        return f"KernelType.{self.name}"

    def __str__(self):
        return self.name
