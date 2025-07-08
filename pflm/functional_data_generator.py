import numpy as np
from math import sqrt
from typing import Optional, Callable
import scipy.special
from pflm.utils import get_eigen_results

"""
Functional Data Generator
=========================

This module provides a class for generating functional data samples based on a mean function and a variance function.
It uses functional principal component analysis (FPCA) to generate data samples with a specified number of components.

Class
-------
FunctionalDataGenerator
"""


class FunctionalDataGenerator(object):
    """
    FunctionalDataGenerator
    ========================
    A class for generating functional data samples based on a mean function and a variance function.
    It uses functional principal component analysis (FPCA) to generate data samples with a specified number of components.

    parameters
    ----------
    t (np.ndarray): The time points at which the functional data is defined.
    mean_func (Callable[[np.ndarray], np.ndarray]):
        A callable function that takes an array of time points and returns the mean function values at those time points.
    var_func (Callable[[np.ndarray], np.ndarray]):
        A callable function that takes an array of time points and returns the variance function values at those time points.
    corr_func (Callable[[np.ndarray], np.ndarray], optional):
        A callable function that defines the correlation structure of the functional data. Defaults to scipy.special.j0 (Bessel function of the first kind).
    variation_prop_thresh (float, optional):
        The threshold for the proportion of variation explained by the functional principal components. It must be between 0 and 1 (exclusive).
        Defaults to 0.999999.
    error_var (float, optional):
        The variance of the error term added to the generated functional data samples. Defaults to 1.0.
    """

    def __init__(
        self,
        t: np.ndarray,
        mean_func: Callable[[np.ndarray], np.ndarray],
        var_func: Callable[[np.ndarray], np.ndarray],
        corr_func: Callable[[np.ndarray], np.ndarray] = scipy.special.j0,
        variation_prop_thresh: float = 0.999999,
        error_var: float = 1.0
    ):
        self.t: np.ndarray = t
        self.mean_func: Callable[[np.ndarray], np.ndarray] = mean_func
        self.var_func: Callable[[np.ndarray], np.ndarray] = var_func
        self.corr_func: Callable[[np.ndarray], np.ndarray] = corr_func
        if not (0 < variation_prop_thresh < 1):
            raise ValueError("variation_prop_thresh must be between 0 and 1 (exclusive).")
        self.variation_prop_thresh: float = variation_prop_thresh
        self.error_var: float = error_var
        self._num_fpc: Optional[int] = None
        self._fpca_phi: Optional[np.ndarray] = None

    def __calculate_fpca_phi(self):
        nt = len(self.t)
        corr = self.corr_func(np.abs(self.t))
        corr_mat = np.zeros((nt, nt))
        for i in range(nt):
            corr_mat[i, i:nt] = corr[0:nt - i]
        mean_func = self.mean_func(self.t)
        self._num_fpc, _, self._fpca_phi, _ = get_eigen_results(self.t, mean_func, corr_mat, self.variation_prop_thresh)

    def get_fpca_phi(self) -> np.ndarray:
        """Get the functional principal component basis functions.

        Returns
        -------
            np.ndarray: The functional principal component basis functions.
        """
        if self._fpca_phi is None:
            self.__calculate_fpca_phi()
        return self._fpca_phi

    def get_num_fpc(self) -> int:
        """Get the number of functional principal components.

        Returns
        -------
            int: The number of functional principal components.
        """
        if self._num_fpc is None:
            self.__calculate_fpca_phi()
        return self._num_fpc

    def generate(self, n: int, seed: Optional[int] = None) -> np.ndarray:
        """Generate functional data samples.

        Parameters
        ----------
            n (int): The number of samples to generate.
            seed (Optional[int], optional): Random seed for reproducibility. Defaults to None.

        Returns
        -------
            np.ndarray: The generated functional data samples.
        """
        rng = np.random.default_rng(seed)
        if self._fpca_phi is None:
            self.__calculate_fpca_phi()
        nt = len(self.t)
        fpc_scores = rng.multivariate_normal(np.zeros(self._num_fpc), np.eye(self._num_fpc), n)
        y = np.matmul(fpc_scores, self._fpca_phi.T) * np.sqrt(self.var_func(self.t)) + \
            rng.normal(0, sqrt(self.error_var), (n, nt)) + self.mean_func(self.t)
        return y

    @staticmethod
    def make_missing(y: np.ndarray, missing_number: int, seed: Optional[int] = None) -> np.ndarray:
        """Introduce missing values into the functional data samples.

        Parameters
        ----------
            y (np.ndarray): The functional data samples.
            missing_number (int): The number of missing values to introduce per sample.
            seed (Optional[int], optional): Random seed for reproducibility. Defaults to None.

        Returns
        -------
            np.ndarray: The functional data samples with missing values.
        """
        rng = np.random.default_rng(seed)
        output = y.copy()
        if missing_number < 1 or missing_number >= y.shape[1]:
            raise ValueError("missing_number must be between 1 and the number of columns in y (exclusive).")
        if np.isnan(output).any():
            raise ValueError("Input data y should not contain NaN values.")

        for i in range(y.shape[0]):
            nan_indices = rng.choice(y.shape[1], missing_number, replace=False)
            output[i, nan_indices] = np.nan
        return output
