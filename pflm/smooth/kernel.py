"""Kernel types for local regression in pflm."""

# Authors: Ching-Chuan Chen
# SPDX-License-Identifier: MIT

from enum import Enum


class KernelType(Enum):
    """Enum for kernel types used in local regression."""

    GAUSSIAN = 0
    LOGISTIC = 1
    SIGMOID = 2
    SILVERMAN = 3
    GAUSSIAN_VAR = 4
    RECTANGULAR = 5  # Uniform Kernel
    TRIANGULAR = 6
    EPANECHNIKOV = 7
    BIWEIGHT = 8     # Quartic Kernel
    TRIWEIGHT = 9
    TRICUBE = 10

    def __repr__(self):
        return f"KernelType.{self.name}"

    def __str__(self):
        return self.name
