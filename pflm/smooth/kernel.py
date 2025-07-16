"""Kernel types for local regression in pflm."""

# Authors: Ching-Chuan Chen
# SPDX-License-Identifier: MIT

from enum import Enum


class KernelType(Enum):
    """Enum for kernel types used in local regression."""

    GAUSSIAN = 0
    LOGISTIC = 1
    SIGMOID = 2
    # Consider to deal with the following kernels in the future.
    # The needed action is to ensure that our implementation can handle negative weights if needed.
    # GAUSSIAN_VAR = 3 # Shifted Gaussian kernel is not included since it might produce negative weights that are not supported.
    # SILVERMAN = 4 # Silverman kernel is not included since it might produce negative weights that are not supported.
    RECTANGULAR = 100  # Uniform Kernel
    TRIANGULAR = 101
    EPANECHNIKOV = 102
    BIWEIGHT = 103  # Quartic Kernel
    TRIWEIGHT = 104
    TRICUBE = 105
    COSINE = 106

    def __repr__(self):
        return f"KernelType.{self.name}"

    def __str__(self):
        return self.name
