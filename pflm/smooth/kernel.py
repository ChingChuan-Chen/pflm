"""Kernel types for local regression in pflm."""

# Authors: Ching-Chuan Chen
# SPDX-License-Identifier: MIT

from enum import Enum


class KernelType(Enum):
    """Enum for kernel types used in local regression."""

    GAUSSIAN = 0
    GAUSSIAN_VAR = 1
    RECTANGULAR = 2
    TRIANGULAR = 3
    EPANECHNIKOV = 4
    BIWEIGHT = 5
    TRIWEIGHT = 6
    TRICUBE = 7
    COSINE = 8

    def __repr__(self):
        return f"KernelType.{self.name}"

    def __str__(self):
        return self.name
