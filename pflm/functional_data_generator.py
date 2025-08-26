"""Functional Data Generator"""

# Authors: Ching-Chuan Chen
# SPDX-License-Identifier: MIT

from math import sqrt
from typing import Callable, List, Optional, Tuple

import numpy as np
import scipy.special

from pflm.utils import get_eigen_analysis_results, get_fpca_phi, select_num_pcs_fve


class FunctionalDataGenerator(object):
    """
    FunctionalDataGenerator
    ========================
    A class for generating functional data samples based on a mean function and a variance function.
    It uses functional principal component analysis (FPCA) to generate data samples with a specified number of components.

    parameters
    ----------
    t : array_like
        The time points at which the functional data is defined. It should be a 1D array where each element corresponds to a time point.
        The length of `t`, `nt`, determines the number of time points in the generated functional data samples.
    mean_func : Callable[[np.ndarray], np.ndarray]
        A callable function that takes an array of time points and returns the mean function values at those time points.
    var_func : Callable[[np.ndarray], np.ndarray]
        A callable function that takes an array of time points and returns the variance function values at those time points.
    corr_func : Callable[[np.ndarray], np.ndarray], optional
        A callable function that defines the correlation structure of the functional data. Defaults to scipy.special.j0
        (Bessel function of the first kind).
    variation_prop_thresh : float, optional, default=0.999999
        The threshold for the proportion of variation explained by the functional principal components. It must be between 0 and 1.
        This is only used to determine the number of principal components to retain when `num_pcs` is not specified.
    num_pcs : int, optional, default=None
        The number of principal components to retain. If None, the number of components will be determined based on the
        `variation_prop_thresh`. It must be a positive integer between 1 and the length of `t`.
    error_var : float, optional, default=1.0
        The variance of the error term added to the generated functional data samples.
    """

    def __init__(
        self,
        t: np.ndarray,
        mean_func: Callable[[np.ndarray], np.ndarray],
        var_func: Callable[[np.ndarray], np.ndarray],
        corr_func: Callable[[np.ndarray], np.ndarray] = scipy.special.j0,
        variation_prop_thresh: float = 0.999999,
        num_pcs: Optional[int] = None,
        error_var: float = 1.0,
    ):
        self.t: np.ndarray = t
        self.mean_func: Callable[[np.ndarray], np.ndarray] = mean_func
        self.var_func: Callable[[np.ndarray], np.ndarray] = var_func
        self.corr_func: Callable[[np.ndarray], np.ndarray] = corr_func
        if not (0 < variation_prop_thresh < 1):
            raise ValueError("variation_prop_thresh must be between 0 and 1.")
        if num_pcs is not None:
            if not isinstance(num_pcs, int):
                raise ValueError("num_pcs must be an integer.")
            if not (1 <= num_pcs <= len(t)):
                raise ValueError("num_pcs must be a positive integer between 1 and length of t.")
        self.variation_prop_thresh: float = variation_prop_thresh
        self.error_var: float = error_var
        self._num_pcs: Optional[int] = num_pcs
        self._fpca_phi: Optional[np.ndarray] = None

    def __calculate_fpca_phi(self):
        nt = len(self.t)
        corr = self.corr_func(np.abs(self.t))
        corr_mat = np.zeros((nt, nt))
        for i in range(nt):
            corr_mat[i, i:nt] = corr[0: nt - i]
        mean_func = self.mean_func(self.t)
        eig_lambda, eig_vector = get_eigen_analysis_results(corr_mat, is_upper_triangular=True)
        if self._num_pcs is None:
            _, self._num_pcs = select_num_pcs_fve(eig_lambda, self.variation_prop_thresh)
        _, self._fpca_phi = get_fpca_phi(self._num_pcs, self.t, mean_func, eig_lambda, eig_vector)

    def get_fpca_phi(self) -> np.ndarray:
        """Get the functional principal component basis functions.

        Returns
        -------
        fpca_phi : array_like
            The functional principal component basis functions.
        """
        if self._fpca_phi is None:
            self.__calculate_fpca_phi()
        return self._fpca_phi

    def get_num_pcs(self) -> int:
        """Get the number of functional principal components.

        Returns
        -------
        num_pcs : int
            The number of functional principal components.
        """
        if self._num_pcs is None:
            self.__calculate_fpca_phi()
        return self._num_pcs

    def generate(self, n: int, seed: Optional[int] = None) -> np.ndarray:
        """Generate functional data samples.

        Parameters
        ----------
        n : int
            The number of functional data samples to generate.
            It must be a positive integer.
        seed : Optional[int], optional
            Random seed for reproducibility. If None, the random number generator will not be seeded.

        Returns
        -------
        y : list of array_like
            The generated functional data samples. Each element in the list corresponds to a sample and is a 1D array of shape (nt,).
        t : list of array_like
            The time points corresponding to each sample. Each element in the list is a 1D array of shape (nt,).
        """
        rng = np.random.default_rng(seed)
        if self._fpca_phi is None:
            self.__calculate_fpca_phi()
        nt = len(self.t)
        fpc_scores = rng.multivariate_normal(np.zeros(self._num_pcs), np.eye(self._num_pcs), n)
        y_mat = (
            np.matmul(fpc_scores, self._fpca_phi.T) * np.sqrt(self.var_func(self.t))
            + rng.normal(0, sqrt(self.error_var), (n, nt))
            + self.mean_func(self.t)
        )

        # put data into lists
        y = []
        t = []
        for i in range(n):
            y.append(y_mat[i, :])
            t.append(self.t)
        return y, t

    @staticmethod
    def make_missing(
        y: List[np.ndarray], t: List[np.ndarray], missing_number: int, seed: Optional[int] = None
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Introduce missing values into the functional data samples.

        Parameters
        ----------
        y : list of array_like
            The generated functional data samples. Each element in the list corresponds to a sample and is a 1D array of shape (nt,).
        t : list of array_like
            The time points corresponding to each sample. Each element in the list is a 1D array of shape (nt,).
        missing_number : int
            The number of missing values to introduce in each sample. It must be between 1 and the length of `y[0]`.
            If `missing_number` is less than 1 or greater than or equal to the length of `y[0]`, a ValueError will be raised.
        seed : Optional[int], optional
            Random seed for reproducibility. If None, the random number generator will not be seeded.

        Returns
        -------
        y : list of array_like
            The generated functional data samples. Each element in the list corresponds to a sample and is a 1D array of shape (nt_i,), i=0,...,n-1.
        t : list of array_like
            The time points corresponding to each sample. Each element in the list is a 1D array of shape (nt_i,), i=0,...,n-1.
        """
        rng = np.random.default_rng(seed)
        nt = len(t[0])
        if missing_number < 1 or missing_number >= nt:
            raise ValueError("missing_number must be between 1 and the length of y[0]/t[0].")
        if any(np.isnan(y[i]).sum() > 0 for i in range(len(y))):
            raise ValueError("y contains NaN values.")

        new_y = []
        new_t = []
        for i in range(len(y)):
            non_nan_indices = rng.choice(nt, nt - missing_number, replace=False)
            new_y.append(y[i][non_nan_indices])
            new_t.append(t[i][non_nan_indices])
        return new_y, new_t
