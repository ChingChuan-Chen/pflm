"""Functional data generation utilities.

This module provides a class to synthesize functional observations using
a mean function, marginal variance, and correlation structure, with FPCA
to draw low‑rank samples on a grid.
"""

# Authors: Ching-Chuan Chen
# SPDX-License-Identifier: MIT

from math import sqrt
from typing import Callable, List, Optional, Tuple

import numpy as np
from scipy.special import j0

from pflm.fpca import get_eigen_analysis_results, get_fpca_phi, select_num_pcs_fve


class FunctionalDataGenerator:
    """Generator for synthetic functional data on a fixed grid.

    This class builds a stationary covariance surface from a marginal variance
    function and a correlation kernel, performs an FPCA on the implied
    covariance, and samples low-rank functional signals with optional
    Gaussian measurement noise.

    Parameters
    ----------
    t : np.ndarray of shape (nt,)
        Monotonic grid of time points.
    mean_func : Callable[[np.ndarray], np.ndarray]
        Mean function evaluated on `t`, returns shape (nt,).
    var_func : Callable[[np.ndarray], np.ndarray]
        Marginal variance function evaluated on `t`, returns shape (nt,).
    corr_func : Callable[[np.ndarray], np.ndarray], default=scipy.special.j0
        Correlation kernel k(h) used to build the covariance surface, where
        h is the absolute time lag.
    variation_prop_thresh : float, default=0.999999
        Threshold of fraction of variance explained (FVE) to choose the number
        of components if `num_pcs` is None. Must satisfy 0 < thresh < 1.
    num_pcs : int or None, default=None
        Number of principal components to retain. If None, it is determined
        by the FVE threshold.
    error_var : float, default=1.0
        Gaussian noise variance added to generated curves.

    Attributes
    ----------
    t : np.ndarray of shape (nt,)
        Copy of the input grid.
    mean_func : Callable[[np.ndarray], np.ndarray]
        Mean function handle used during generation.
    var_func : Callable[[np.ndarray], np.ndarray]
        Marginal variance function handle used during generation.
    corr_func : Callable[[np.ndarray], np.ndarray]
        Correlation kernel used to build the covariance surface.
    variation_prop_thresh : float
        FVE threshold used when `num_pcs` is not specified.
    error_var : float
        Measurement noise variance for generation.

    Notes
    -----
    - The covariance is constructed as
      sqrt(var_func(t_i)) * corr(|t_i - t_j|) * sqrt(var_func(t_j)).
    - FPCA components (eigenstructure) are computed lazily on first use.
    - Private caches:
      - `_num_pcs`: Optional[int]
      - `_fpca_phi`: Optional[np.ndarray of shape (nt, k)]
    """

    def __init__(
        self,
        t: np.ndarray,
        mean_func: Callable[[np.ndarray], np.ndarray],
        var_func: Callable[[np.ndarray], np.ndarray],
        corr_func: Callable[[np.ndarray], np.ndarray] = j0,
        variation_prop_thresh: float = 0.999999,
        num_pcs: Optional[int] = None,
        error_var: float = 1.0,
    ):
        """Initialize the generator and validate basic inputs.

        Parameters
        ----------
        t : np.ndarray of shape (nt,)
            Monotonic grid of time points.
        mean_func : Callable[[np.ndarray], np.ndarray]
            Mean function evaluated on `t`.
        var_func : Callable[[np.ndarray], np.ndarray]
            Marginal variance function evaluated on `t`.
        corr_func : Callable[[np.ndarray], np.ndarray], default=scipy.special.j0
            Correlation kernel k(h) with h = |t_i - t_j|.
        variation_prop_thresh : float, default=0.999999
            FVE threshold in (0, 1) used when `num_pcs` is None.
        num_pcs : int or None, default=None
            Fixed number of components; if provided, must be an integer in [1, nt].
        error_var : float, default=1.0
            Gaussian noise variance for generation.

        Raises
        ------
        ValueError
            If `variation_prop_thresh` is not in (0, 1), or if `num_pcs` is not
            a valid positive integer within the range [1, len(t)] when provided.
        """
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
        """Compute and cache FPCA eigenfunctions on the provided grid.

        This routine:
        1) Builds the upper-triangular correlation matrix implied by `corr_func`
           on the absolute lags of `t`.
        2) Performs eigen analysis to obtain eigenvalues/vectors.
        3) Selects the number of components by FVE if `_num_pcs` is None.
        4) Scales/normalizes eigenvectors into FPCA basis using `get_fpca_phi`,
           then caches the result in `_fpca_phi` and `_num_pcs`.

        Side Effects
        ------------
        Sets the private fields `_num_pcs` and `_fpca_phi`.

        Notes
        -----
        The eigen decomposition uses the upper-triangular flag to avoid
        reconstructing the full symmetric matrix. Downstream utilities handle
        normalization with respect to the grid spacing and mean alignment.

        See Also
        --------
        pflm.utils.get_eigen_analysis_results
        pflm.utils.get_fpca_phi
        pflm.utils.select_num_pcs_fve
        """
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
        """Return the FPCA basis functions evaluated on `t`.

        Returns
        -------
        fpca_phi : np.ndarray of shape (nt, k)
            The functional principal component basis functions.

        Notes
        -----
        The FPCA basis is computed lazily on the first call and cached.
        """
        if self._fpca_phi is None:
            self.__calculate_fpca_phi()
        return self._fpca_phi

    def get_num_pcs(self) -> int:
        """Return the number of retained principal components.

        Returns
        -------
        num_pcs : int
            Effective number of retained FPCA components.

        Notes
        -----
        If not set explicitly, the value is determined by the FVE threshold
        on first access and then cached.
        """
        if self._num_pcs is None:
            self.__calculate_fpca_phi()
        return self._num_pcs

    def generate(self, n: int, seed: Optional[int] = None) -> np.ndarray:
        """Generate functional data samples.

        Parameters
        ----------
        n : int
            Number of functional samples to generate (typically n > 0).
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        y : List[np.ndarray]
            List of length `n`; each element has shape (nt,) and represents one sample.
        t : List[np.ndarray]
            List of length `n`; each element is the time grid of shape (nt,).

        Notes
        -----
        - Scores are drawn from N(0, diag(lambda)) implicitly via an identity
          covariance in score space and rescaled by the FPCA basis and variance.
        - Gaussian noise with variance `error_var` is added independently per point.
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
        """Introduce missing values into each functional sample.

        Parameters
        ----------
        y : List[np.ndarray]
            Functional samples; each array has shape (nt_i,).
        t : List[np.ndarray]
            Time grids corresponding to `y`; each array has shape (nt_i,).
        missing_number : int
            Number of indices to drop per sample. Must satisfy 1 <= m < nt_i.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        new_y : List[np.ndarray]
            Samples with missing entries removed.
        new_t : List[np.ndarray]
            Corresponding time points with the same indices removed.

        Raises
        ------
        ValueError
            If `missing_number` is not in [1, len(y[0]) - 1] or if input `y`
            already contains NaN values.
        """
        rng = np.random.default_rng(seed)
        nt = len(t[0])
        if missing_number < 1 or missing_number >= nt:
            raise ValueError("missing_number must be between 1 and the length of y[0]/t[0].")
        if any(np.isnan(y[i]).sum() > 0 for i in range(len(y))):
            raise ValueError("y contains NaN values.")

        new_y = []
        new_t = []
        for i, (yi, ti) in enumerate(zip(y, t)):
            non_nan_indices = rng.choice(nt, nt - missing_number, replace=False)
            new_y.append(yi[non_nan_indices])
            new_t.append(ti[non_nan_indices])
        return new_y, new_t
