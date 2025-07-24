"""polyfit1d model for 1D and 2D polynomial fitting with kernel smoothing."""

# Authors: Ching-Chuan Chen
# SPDX-License-Identifier: MIT

import math
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils._array_api import get_namespace_and_device, supported_float_dtypes
from sklearn.utils.validation import check_array, check_is_fitted

from pflm.interp import interp1d, interp2d
from pflm.smooth._polyfit import (
    calculate_kernel_value_f32,
    calculate_kernel_value_f64,
    polyfit1d_f32,
    polyfit1d_f64,
    polyfit2d_f32,
    polyfit2d_f64,
)
from pflm.smooth.kernel import KernelType


class Polyfit1DModel(BaseEstimator, RegressorMixin):
    """
    1D polynomial fitting model with kernel smoothing.

    This model fits a local polynomial regression using kernel smoothing and
    uses interpolation for efficient prediction.

    Parameters
    ----------
    kernel_type : KernelType, default=KernelType.GAUSSIAN
        The type of kernel to use for smoothing.
    degree : int, default=1
        The degree of the polynomial. Must be >= 1.
    deriv : int, default=0
        The derivative order to compute. Must be >= 0 and <= degree.
    obs_grid : array-like, default=None
        Custom grid points for interpolation. If None, will create uniform grid.
        Must be within the range of input X.
    n_interp_points : int, default=100
        Number of points to use for interpolation grid (only used if obs_grid is None).
    interp_kind : str, default='linear'
        Type of interpolation ('linear', 'spline').
    random_seed : int, default=None
        Random seed for reproducibility.

    Attributes
    ----------
    X_ : ndarray of shape (n_samples,)
        The input data from the last fit.
    y_ : ndarray of shape (n_samples,)
        The target values from the last fit.
    sample_weight_ : ndarray of shape (n_samples,)
        The sample weights from the last fit.
    n_features_in_ : int
        Number of features seen during fit (always 1 for 1D).
    obs_grid_ : ndarray
        The interpolation grid points.
    obs_fitted_values_ : ndarray
        The fitted values at interpolation grid points.
    bandwidth_ : float
        The bandwidth used for kernel smoothing. It might be selected automatically or provided by the user.
    bandwidth_selection_results_ : dict, optional
        Results of bandwidth selection, including candidates, scores, and the best bandwidth.
    """

    def __init__(
        self,
        *,
        kernel_type: KernelType = KernelType.GAUSSIAN,
        degree: int = 1,
        deriv: int = 0,
        obs_grid: Optional[Union[np.ndarray, List[float]]] = None,
        n_interp_points: int = 100,
        interp_kind: Literal["linear", "spline"] = "linear",
        random_seed: Optional[int] = None,
    ) -> None:
        if kernel_type not in KernelType:
            raise ValueError(f"kernel must be one of {list(KernelType)}.")
        if degree is None or not isinstance(degree, int):
            raise TypeError("Degree of polynomial, degree, should be an integer.")
        if deriv is None or not isinstance(deriv, int):
            raise TypeError("Order of derivative, deriv, should be an integer.")
        if degree <= 0:
            raise ValueError("Degree of polynomial, degree, should be positive.")
        if deriv < 0:
            raise ValueError("Order of derivative, deriv, should be positive.")
        if degree < deriv:
            raise ValueError("Degree of polynomial, degree, should be greater than or equal to order of derivative, deriv.")
        if interp_kind not in ["linear", "spline"]:
            raise ValueError(f"interp_kind must be one of ['linear', 'spline'], got {interp_kind}")

        self.kernel_type = kernel_type
        self.degree = degree
        self.deriv = deriv
        self.obs_grid_ = obs_grid
        self.n_interp_points = n_interp_points
        self.interp_kind = interp_kind
        self.rng = np.random.default_rng(random_seed)

    def _generate_bandwidth_candidates(self, sorted_unique_support: np.ndarray, num_bw_candidates: int) -> np.ndarray:
        """
        Generate bandwidth candidates for selection.

        Parameters
        ----------
        sorted_unique_support : np.ndarray
            Sorted unique values of X for support calculation.
        num_bw_candidates : int
            Number of bandwidth candidates to generate.

        Returns
        -------
        np.ndarray
            Array of bandwidth candidates.
        """
        if sorted_unique_support.size <= self.degree + 1:
            raise ValueError(f"Not enough unique support points ({sorted_unique_support.size}) to fit a polynomial of degree {self.degree}.")

        lag = self.degree + 1
        d_star = np.max(sorted_unique_support[lag:] - sorted_unique_support[:-lag])
        r = sorted_unique_support[-1] - sorted_unique_support[0]
        h0 = min(1.5 * d_star, r)
        q = math.pow(0.25 * r / h0, 1.0 / (num_bw_candidates - 1))
        return h0 * (q ** np.linspace(0, num_bw_candidates - 1, num_bw_candidates))

    def _compute_cv_score(self, sorted_unique_X: np.ndarray, bandwidth: np.floating, unique_idx: np.ndarray, cv_folds: int = 5) -> np.floating:
        """
        Compute cross-validation score for a given bandwidth.

        Parameters
        ----------
        sorted_unique_X : np.ndarray
            Sorted unique values of X.
        bandwidth : float
            Bandwidth parameter.
        cv_folds : int, default=5
            Number of cross-validation folds.

        Returns
        -------
        float
            Cross-validation score.
        """

        cv_index = np.arange(len(self.X_)) % cv_folds
        self.rng.shuffle(cv_index)

        cv_scores = np.zeros(cv_folds, dtype=self._input_dtype)
        for fold in range(cv_folds):
            train_mask = cv_index != fold
            test_mask = cv_index == fold

            y_pred = self._polyfit1d_func(
                self.sorted_X_[train_mask],
                self.sorted_y_[train_mask],
                self.sorted_sample_weight_[train_mask],
                sorted_unique_X,
                bandwidth,
                self.kernel_type.value,
                self.degree,
                self.deriv,
            )

            if np.isnan(y_pred).any():
                return np.inf

            cv_scores[fold] = sum(
                self.sorted_sample_weight_[unique_idx[test_mask]] * (self.sorted_y_[unique_idx[test_mask]] - y_pred[unique_idx[test_mask]]) ** 2
            )
        return np.mean(cv_scores) / self._sum_sample_weight

    def _compute_gcv_score(
        self, sorted_unique_X: np.ndarray, bandwidth: np.floating, unique_idx: np.ndarray, k0: float, r: np.floating
    ) -> np.floating:
        """
        Compute Generalized Cross-Validation score for a given bandwidth.

        Parameters
        ----------
        sorted_unique_X : np.ndarray
            Sorted unique values of X.
        bandwidth : float
            Bandwidth parameter.
        unique_idx : np.ndarray
            Indices of the original X values in the sorted unique array.
        k0 : float
            Kernel value at zero.
        r : float
            Range of the sorted unique X values.

        Returns
        -------
        float
            GCV score.
        """

        y_pred = self._polyfit1d_func(
            self.sorted_X_,
            self.sorted_y_,
            self.sorted_sample_weight_,
            sorted_unique_X,
            bandwidth,
            self.kernel_type.value,
            self.degree,
            self.deriv,
        )
        if np.isnan(y_pred).any():
            return np.inf
        numerator = sum(self.sorted_sample_weight_ * (self.sorted_y_ - y_pred[unique_idx]) ** 2)
        denominator = math.pow(1.0 - r * k0 / bandwidth / self._sum_sample_weight, 2.0)
        return numerator / denominator

    def _select_bandwidth(
        self,
        num_bw_candidates: int = 21,
        method: Literal["cv", "gcv"] = "gcv",
        custom_bw_candidates: Optional[np.ndarray] = None,
        cv_folds: int = 5,
    ) -> np.floating:
        """
        Select bandwidth using cross-validation.

        Parameters
        ----------
        num_bw_candidates : int, default=21
            Number of bandwidth candidates to generate. Only used if custom_bw_candidates is None.
        method : str, default='gcv'
            Method for bandwidth selection. Options: 'cv', 'gcv'.
            If 'cv', uses cross-validation; if 'gcv', uses Generalized Cross-Validation.
        custom_bw_candidates: Optional[np.ndarray], default=None
            Custom bandwidth candidates to use instead of generating them.
        cv_folds : int, default=5
            Number of cross-validation folds.

        Returns
        -------
        float
            The bandwidth with the best score.
        """
        sorted_unique_X, unique_idx = np.unique(self.sorted_X_, return_inverse=True)
        if custom_bw_candidates is not None:
            bandwidth_candidates_ = custom_bw_candidates
        else:
            bandwidth_candidates_ = self._generate_bandwidth_candidates(sorted_unique_X, num_bw_candidates)

        # Store for inspection
        self.bandwidth_selection_results_ = {
            "bandwidth_candidates": bandwidth_candidates_,
            "bandwidth_selection_method": method,
        }

        self._sum_sample_weight = np.sum(self.sample_weight_)
        if method == "cv":
            cv_scores = np.array([self._compute_cv_score(sorted_unique_X, bw, unique_idx, cv_folds) for bw in bandwidth_candidates_])
            if (~np.isfinite(cv_scores)).all():
                raise ValueError("All CV scores are non-finite. Check your data and bandwidth candidates.")
            self.bandwidth_selection_results_["cv_scores"] = cv_scores
            self.bandwidth_selection_results_["best_bandwidth"] = bandwidth_candidates_[np.argmin(cv_scores)]
            self.bandwidth_selection_results_["cv_folds"] = cv_folds
        elif method == "gcv":
            k0 = (
                calculate_kernel_value_f32(0, self.kernel_type.value)
                if self._input_dtype == np.float32
                else calculate_kernel_value_f64(0, self.kernel_type.value)
            )
            r = sorted_unique_X[-1] - sorted_unique_X[0]
            gcv_scores = np.array([self._compute_gcv_score(sorted_unique_X, bw, unique_idx, k0, r) for bw in bandwidth_candidates_])
            if (~np.isfinite(gcv_scores)).all():
                raise ValueError("All GCV scores are non-finite. Check your data and bandwidth candidates.")
            self.bandwidth_selection_results_["gcv_scores"] = gcv_scores
            self.bandwidth_selection_results_["best_bandwidth"] = bandwidth_candidates_[np.argmin(gcv_scores)]

        return self.bandwidth_selection_results_["best_bandwidth"]

    def fit(
        self,
        X: Union[np.ndarray, List[float]],
        y: Union[np.ndarray, List[float]],
        sample_weight: Optional[Union[np.ndarray, List[float]]] = None,
        bandwidth: Optional[float] = None,
        num_bw_candidates: int = 21,
        bandwidth_selection_method: Literal["cv", "gcv"] = "gcv",
        custom_bw_candidates: Optional[np.ndarray] = None,
        cv_folds: int = 5,
    ) -> "Polyfit1DModel":
        """
        Fit the 1D polynomial model.

        Parameters
        ----------
        X : array-like of shape (n_samples, 1) or (n_samples,)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.
        bandwidth : float, default=None
            The bandwidth parameter for kernel smoothing. If None, will be selected
            using the method specified in bandwidth_selection_method.
        num_bw_candidates : int, default=21
            Number of bandwidth candidates to generate.
        bandwidth_selection_method : str, default='gcv'
            Method for bandwidth selection. Options: 'cv', 'gcv'.
        custom_bw_candidates: Optional[np.ndarray], default=None
            Custom bandwidth candidates to use instead of generating them.
        cv_folds : int, default=5
            Number of cross-validation folds (only used if bandwidth_selection_method='cv').

        Returns
        -------
        Polyfit1DModel
            Returns self.
        """
        # Validate bandwidth_selection_method
        if bandwidth_selection_method not in ["cv", "gcv"]:
            raise ValueError(f"bandwidth_selection_method must be one of ['cv', 'gcv'], got {bandwidth_selection_method}")
        if bandwidth_selection_method == "cv" and cv_folds < 2:
            raise ValueError("Number of cross-validation folds, cv_folds, should be at least 2 for 'cv' method.")

        if bandwidth is not None and not isinstance(bandwidth, (float, int)):
            raise ValueError("bandwidth must be positive float or integer.")
        elif bandwidth is not None and np.isnan(bandwidth):
            raise ValueError("bandwidth must not be NaN")
        elif bandwidth is not None and bandwidth <= 0:
            raise ValueError("bandwidth must be positive.")

        if num_bw_candidates is None or not isinstance(num_bw_candidates, int):
            raise TypeError("Number of bandwidth candidates, num_bw_candidates, should be an integer.")
        if num_bw_candidates < 2:
            raise ValueError("Number of bandwidth candidates, num_bw_candidates, should be at least 2.")

        xp, *_ = get_namespace_and_device(X, y, sample_weight)

        # Handle both (n_samples,) and (n_samples, 1) input shapes
        X = check_array(X, ensure_2d=False, dtype=supported_float_dtypes(xp))
        if X.ndim == 2:
            if X.shape[1] != 1:
                raise ValueError(f"X must have exactly 1 feature, got {X.shape[1]}")
            X = X.ravel()
        y = check_array(y, ensure_2d=False, dtype=X.dtype)
        if y.size != X.size:
            raise ValueError("y must have the same size as X.")

        # Store input dtype for later use
        self._input_dtype = X.dtype
        # Fit polynomial at grid points
        self._polyfit1d_func = polyfit1d_f32 if self._input_dtype == np.float32 else polyfit1d_f64

        # Handle sample weights
        if sample_weight is not None:
            sample_weight = check_array(sample_weight, ensure_2d=False, dtype=self._input_dtype)
            if sample_weight.size != y.size:
                raise ValueError(f"sample_weight must have the same length as y, got {sample_weight.shape[0]} vs {y.shape[0]}")
            if np.any(sample_weight < 0):
                raise ValueError("All sample weights must be non-negative")
        else:
            sample_weight = np.ones_like(y, dtype=self._input_dtype)

        # Store fitted data
        self.X_ = X
        self.y_ = y
        self.sample_weight_ = sample_weight
        self.n_features_in_ = 1

        # Sort training data by X for polyfit1d requirement
        sort_idx = np.argsort(X)
        self.sorted_X_ = X[sort_idx]
        self.sorted_y_ = y[sort_idx]
        self.sorted_sample_weight_ = sample_weight[sort_idx]

        # Select bandwidth if needed
        if bandwidth is None:
            if custom_bw_candidates is not None:
                custom_bw_candidates = check_array(custom_bw_candidates, ensure_2d=False, dtype=self._input_dtype)
                if custom_bw_candidates.ndim == 2:
                    if custom_bw_candidates.shape[1] != 1:
                        raise ValueError(f"custom_bw_candidates must have exactly 1 feature, got {custom_bw_candidates.shape[1]}")
                    custom_bw_candidates = custom_bw_candidates.ravel()
            self.bandwidth_ = self._select_bandwidth(
                num_bw_candidates,
                bandwidth_selection_method,
                custom_bw_candidates,
                cv_folds,
            )
        else:
            self.bandwidth_ = float(bandwidth)

        # Create interpolation grid
        x_min, x_max = np.min(X), np.max(X)

        if self.obs_grid_ is not None:
            # Use custom grid
            self.obs_grid_ = check_array(self.obs_grid_, ensure_2d=False, dtype=self._input_dtype)
            if self.obs_grid_.ndim != 1:
                raise ValueError("obs_grid must be a 1D array")
            if len(self.obs_grid_) < 2:
                raise ValueError("obs_grid must have at least 2 points")

            # Check if grid is within the range of input X
            if np.min(self.obs_grid_) < x_min or np.max(self.obs_grid_) > x_max:
                raise ValueError(
                    f"obs_grid must be within the range of input X [{x_min:.6f}, {x_max:.6f}]. "
                    f"Got obs_grid range [{np.min(self.obs_grid_):.6f}, {np.max(self.obs_grid_):.6f}]"
                )
        else:
            # Create uniform grid within the range of input X
            self.obs_grid_ = np.linspace(x_min, x_max, self.n_interp_points)

        try:
            self.obs_fitted_values_ = self._polyfit1d_func(
                self.sorted_X_,
                self.sorted_y_,
                self.sorted_sample_weight_,
                self.obs_grid_,
                self.bandwidth_,
                self.kernel_type.value,
                self.degree,
                self.deriv,
            )
        except Exception as e:
            raise ValueError(f"Error in polyfit1d: {e!s}") from e

        return self

    def predict(self, X: Union[np.ndarray, List[float]], use_model_interp: bool = True) -> np.ndarray:
        """
        Predict using the 1D polynomial model via interpolation.

        Parameters
        ----------
        X : array-like of shape (n_samples, 1) or (n_samples,)
            Samples to predict.
        use_model_interp : bool, default=True
            If True, use the model's interpolation grid for prediction.
            If False, use direct polyfit1d call for prediction.

        Returns
        -------
        np.ndarray
            Predicted values of shape (n_samples,).
        """
        check_is_fitted(self, ["obs_fitted_values_", "obs_grid_", "bandwidth_"])

        # Handle both (n_samples,) and (n_samples, 1) input shapes
        X = check_array(X, ensure_2d=False, dtype=self._input_dtype)
        if X.ndim == 2:
            if X.shape[1] != 1:
                raise ValueError(f"X must have exactly 1 feature, got {X.shape[1]}")
            X = X.ravel()
        ord = np.argsort(X)

        # Use pflm.interp.interp1d for interpolation
        if use_model_interp:
            try:
                y_pred = interp1d(self.obs_grid_, self.obs_fitted_values_, X[ord], self.interp_kind)
            except Exception as e:
                raise ValueError(f"Error during interpolation: {e!s}") from e
        else:
            # Predict using direct polyfit1d call
            try:
                y_pred = self._polyfit1d_func(
                    self.sorted_X_,
                    self.sorted_y_,
                    self.sorted_sample_weight_,
                    X[ord],
                    self.bandwidth_,
                    self.kernel_type.value,
                    self.degree,
                    self.deriv,
                )
            except Exception as e:
                raise ValueError(f"Error in polyfit1d: {e!s}") from e

        inverse_sort_idx = np.argsort(ord)
        return y_pred[inverse_sort_idx]

    def get_fitted_grids(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the fitted values at the interpolation grid points.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            A tuple containing:
            - obs_grid: The interpolation grid points.
            - obs_fitted_values: The fitted values at interpolation grid points.
        """
        check_is_fitted(self, ["obs_fitted_values_", "obs_grid_", "bandwidth_"])
        return self.obs_grid_.copy(), self.obs_fitted_values_.copy()


class Polyfit2DModel(BaseEstimator, RegressorMixin):
    """
    2D polynomial fitting model with kernel smoothing.

    Parameters
    ----------
    kernel_type : KernelType, default=KernelType.GAUSSIAN
        The type of kernel to use for smoothing.
    degree : int, default=1
        The degree of the polynomial.
    deriv1 : int, default=0
        The derivative order for the first dimension.
    deriv2 : int, default=0
        The derivative order for the second dimension.
    obs_grid1 : array-like, default=None
        Custom grid points for interpolation in the first dimension. If None, will create uniform grid.
        Must be within the range of the first dimension of input X.
    obs_grid2 : np.ndarray, default=None
        Custom grid points for interpolation in the second dimension. If None, will create uniform grid.
        Must be within the range of the second dimension of input Y.
    n_interp_points : int, default=100
        Number of points for interpolation grid in each dimension (only used if obs_grid is None).
    interp_kind : str, default='linear'
        Interpolation method ('linear', 'cubic', 'quintic').
    random_seed : int, default=None
        Random seed for reproducibility.

    Attributes
    ----------
    X_ : ndarray of shape (n_samples, 2)
        The input data from the last fit.
    y_ : ndarray of shape (n_samples,)
        The target values from the last fit.
    sample_weight_ : ndarray of shape (n_samples,)
        The sample weights from the last fit.
    n_features_in_ : int
        Number of features seen during fit (always 2 for 2D).
    obs_grid1_ : ndarray
        The interpolation grid points in the first dimension.
    obs_grid2_ : ndarray
        The interpolation grid points in the second dimension.
    obs_fitted_values_ : ndarray
        A 2D array of fitted values at the interpolation grid points.
    bandwidth1_ : float
        The selected bandwidth for the first dimension after fitting.
    bandwidth2_ : float
        The selected bandwidth for the second dimension after fitting.
    cv_scores_ : ndarray, optional
        CV scores for each bandwidth candidate pair.
    bandwidth_candidates_ : ndarray, optional
        Bandwidth candidate pairs evaluated during selection.
    """

    def __init__(
        self,
        *,
        kernel_type: KernelType = KernelType.GAUSSIAN,
        degree: int = 1,
        deriv1: int = 0,
        deriv2: int = 0,
        obs_grid1: Optional[Union[np.ndarray, List[float]]] = None,
        obs_grid2: Optional[Union[np.ndarray, List[float]]] = None,
        n_interp_points: int = 100,
        interp_kind: Literal["linear", "spline"] = "linear",
        random_seed: Optional[int] = None,
    ) -> None:
        if kernel_type not in KernelType:
            raise ValueError(f"kernel_type must be one of {list(KernelType)}.")
        if degree is None or not isinstance(degree, int):
            raise TypeError("Degree of polynomial, degree, should be an integer.")
        if deriv1 is None or not isinstance(deriv1, int):
            raise TypeError("Order of derivative in first dimension, deriv1, should be an integer.")
        if deriv2 is None or not isinstance(deriv2, int):
            raise TypeError("Order of derivative in second dimension, deriv2, should be an integer.")
        if degree <= 0:
            raise ValueError("Degree of polynomial, degree, should be positive.")
        if deriv1 < 0 or deriv2 < 0:
            raise ValueError("Order of derivative, deriv1 and deriv2, should be non-negative.")
        if degree < deriv1 + deriv2:
            raise ValueError("Degree of polynomial, degree, should be greater than or equal to sum of orders of derivatives, deriv1 + deriv2.")
        if interp_kind not in ["linear", "spline"]:
            raise ValueError(f"interp_kind must be one of ['linear', 'spline'], got {interp_kind!r}")

        self.kernel_type = kernel_type
        self.degree = degree
        self.deriv1 = deriv1
        self.deriv2 = deriv2
        self.obs_grid1_ = obs_grid1
        self.obs_grid2_ = obs_grid2
        self.n_interp_points = n_interp_points
        self.interp_kind = interp_kind
        self.rng = np.random.default_rng(random_seed)

    def _get_bandwidth_candidates(self, d_star: float, r: float) -> np.ndarray:
        h0 = min(2.0 * d_star, r)
        q = math.pow(0.25 * r / h0, 1.0 / (num_bw_candidates - 1))
        return h0 * (q ** np.linspace(0, num_bw_candidates - 1, num_bw_candidates))

    def _generate_bandwidth_candidates(self, sorted_grid_pairs: np.ndarray, num_bw_candidates: int) -> np.ndarray:
        """
        Generate bandwidth candidates for 2D selection.

        Parameters
        ----------
        sorted_grid_pairs : np.ndarray
            Sorted unique pairs of grid points.
        num_bw_candidates : int
            Number of bandwidth candidates to generate.

        Returns
        -------
        List[Tuple[float, float]]
            List of bandwidth candidate pairs (bandwidth1, bandwidth2).
        """
        if sorted_grid_pairs.shape[0] <= self.degree + 1:
            raise ValueError(f"Not enough unique support points ({sorted_grid_pairs.shape[0]}) to fit a polynomial of degree {self.degree}.")

        lag = self.degree + 1
        d_stars = [
            np.max(sorted_grid_pairs[lag:, 0] - sorted_grid_pairs[:-lag, 0]),
            np.max(sorted_grid_pairs[lag:, 1] - sorted_grid_pairs[:-lag, 1])
        ]
        # sqrt(2) because the window is circular.
        r1 = (sorted_grid_pairs[-1, 0] - sorted_grid_pairs[0, 0]) * math.sqrt(2.0)
        r2 = (np.max(sorted_grid_pairs[:, 1]) - np.min(sorted_grid_pairs[:, 1])) * math.sqrt(2.0)
        bw1v, bw2v = np.meshgrid(self._get_bandwidth_candidates(d_stars[0], r1), self._get_bandwidth_candidates(d_stars[1], r2))
        return np.vstack((bw2v.ravel(), bw1v.ravel()))

    def _compute_cv_score(
        self, sorted_grid_pairs: np.ndarray, bandwidth1: np.floating, bandwidth2: np.floating, cv_folds: int = 5
    ) -> np.floating:
        """
        Compute cross-validation score for given bandwidths.

        Parameters
        ----------
        sorted_grid_pairs : np.ndarray
            Sorted unique pairs of grid points.
        bandwidth1 : float
            Bandwidth for first dimension.
        bandwidth2 : float
            Bandwidth for second dimension.
        cv_folds : int, default=5
            Number of cross-validation folds.

        Returns
        -------
        float
            Cross-validation score.
        """
        # TODO: Implement 2D CV score calculation
        # This is a placeholder - you'll provide the actual implementation
        return np.random.rand()  # Example random score

    def _compute_gcv_score(
        self, sorted_grid_pairs: np.ndarray, bandwidth1: np.floating, bandwidth2: np.floating, unique_idx: np.ndarray, k0: float, r: np.floating
    ) -> np.floating:
        """
        Compute Generalized Cross-Validation score for given bandwidths.

        Parameters
        ----------
        sorted_grid_pairs : np.ndarray
            Sorted unique pairs of grid points.
        bandwidth1 : float
            Bandwidth for first dimension.
        bandwidth2 : float
            Bandwidth for second dimension.
        k0 : float
            Kernel value at zero.
        r : float
            Range of the sorted unique grid pairs.

        Returns
        -------
        float
            GCV score.
        """
        # TODO: Implement 2D GCV score calculation
        # This is a placeholder - you'll provide the actual implementation
        return np.random.rand()  # Example random score

    def _select_bandwidth(
        self,
        num_bw_candidates: int = 21,
        method: Literal["cv", "gcv"] = "gcv",
        same_bandwidth_for_2dim: bool = False,
        custom_bw_candidates: Optional[np.ndarray] = None,
        cv_folds: int = 5,
    ) -> Tuple[np.floating, np.floating]:
        """
        Select bandwidths using cross-validation.

        Parameters
        ----------
        num_bw_candidates : int, default=21
            Number of bandwidth candidates to generate. Only used if custom_bw_candidates is None.
        method : str, default='gcv'
            Method for bandwidth selection. Options: 'cv', 'gcv'.
            If 'cv', uses cross-validation; if 'gcv', uses Generalized Cross-Validation.
        same_bandwidth_for_2dim: bool, default=False
            If True, use the same bandwidth for both dimensions.
            If False, use separate bandwidths for each dimension.
        custom_bw_candidates: Optional[np.ndarray], default=None
            Custom bandwidth candidates to use instead of generating them.
        cv_folds : int, default=5
            Number of cross-validation folds.

        Returns
        -------
        Tuple[float, float]
            The bandwidth pair with the best score.
        """
        sorted_unique_X, unique_idx = np.unique(self.sorted_X_, return_inverse=True)
        if custom_bw_candidates is not None:
            bandwidth_candidates_ = custom_bw_candidates
        else:
            bandwidth_candidates_ = self._generate_bandwidth_candidates(sorted_unique_X, num_bw_candidates)

        if same_bandwidth_for_2dim:
            bandwidth_candidates_ = bandwidth_candidates_[bandwidth_candidates_[:, 0] == bandwidth_candidates_[:, 1], :]

        # Store for inspection
        self.bandwidth_selection_results_ = {
            "bandwidth_candidates": bandwidth_candidates_,
            "bandwidth_selection_method": method,
        }

        if method == "cv":
            cv_scores = np.zeros(bandwidth_candidates_.shape[0], dtype=self._input_dtype)
            for i in range(bandwidth_candidates_.shape[0]):
                cv_scores[i] = self._compute_cv_score(
                    sorted_unique_X, bandwidth_candidates_[i, 0], bandwidth_candidates_[i, 1], unique_idx, cv_folds
                )
            if (~np.isfinite(cv_scores)).all():
                raise ValueError("All CV scores are non-finite. Check your data and bandwidth candidates.")
            self.bandwidth_selection_results_["cv_scores"] = cv_scores
            self.bandwidth_selection_results_["best_bandwidth"] = bandwidth_candidates_[np.argmin(cv_scores)]
            self.bandwidth_selection_results_["cv_folds"] = cv_folds
        elif method == "gcv":
            k0 = (
                calculate_kernel_value_f32(0, self.kernel_type.value)
                if self._input_dtype == np.float32
                else calculate_kernel_value_f64(0, self.kernel_type.value)
            )
            r = sorted_unique_X[-1] - sorted_unique_X[0]
            gcv_scores = np.zeros(bandwidth_candidates_.shape[0], dtype=self._input_dtype)
            for i in range(bandwidth_candidates_.shape[0]):
                gcv_scores[i] = self._compute_gcv_score(
                    sorted_unique_X, unique_idx, bandwidth_candidates_[i, 0], bandwidth_candidates_[i, 1], k0, r
                )
            if (~np.isfinite(gcv_scores)).all():
                raise ValueError("All GCV scores are non-finite. Check your data and bandwidth candidates.")
            self.bandwidth_selection_results_["gcv_scores"] = gcv_scores
            self.bandwidth_selection_results_["best_bandwidth"] = bandwidth_candidates_[np.argmin(gcv_scores)]

        return self.bandwidth_selection_results_["best_bandwidth"]

    def fit(
        self,
        X: Union[np.ndarray, List[List[float]]],
        y: Union[np.ndarray, List[float]],
        sample_weight: Optional[Union[np.ndarray, List[float]]] = None,
        bandwidth1: Optional[float] = None,
        bandwidth2: Optional[float] = None,
        num_bw_candidates: int = 21,
        bandwidth_selection_method: Literal["cv", "gcv"] = "gcv",
        same_bandwidth_for_2dim: bool = False,
        custom_bw_candidates: Optional[np.ndarray] = None,
        cv_folds: int = 5,
    ) -> "Polyfit2DModel":
        """
        Fit the 2D polynomial model.

        Parameters
        ----------
        X : array-like of shape (n_samples, 2)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.
        bandwidth1 : float, default=None
            The bandwidth parameter for the first dimension.
        bandwidth2 : float, default=None
            The bandwidth parameter for the second dimension.
        num_bw_candidates : int, default=21
            Number of bandwidth candidates to generate.
            Only used if custom_bw_candidates is None.
        bandwidth_selection_method : str, default='gcv'
            Method for bandwidth selection. Options: 'cv', 'gcv'.
        same_bandwidth_for_2dim: bool, default=False
            If True, use the same bandwidth for both dimensions.
            If False, use separate bandwidths for each dimension.
        custom_bw_candidates: Optional[np.ndarray], default=None
            Custom bandwidth candidates to use instead of generating them.
        cv_folds : int, default=5
            Number of cross-validation folds (only used if bandwidth_selection='cv').

        Returns
        -------
        Polyfit2DModel
            Returns self.
        """
        # Validate bandwidth_selection_method
        if bandwidth_selection_method not in ["cv", "gcv"]:
            raise ValueError(f"bandwidth_selection_method must be one of ['cv', 'gcv'], got {bandwidth_selection_method}")
        if bandwidth_selection_method == "cv" and cv_folds < 2:
            raise ValueError("Number of cross-validation folds, cv_folds, should be at least 2 for 'cv' method.")

        if bandwidth1 is not None and not isinstance(bandwidth1, (float, int)):
            raise ValueError("bandwidth1 must be positive float or integer.")
        elif bandwidth1 is not None and np.isnan(bandwidth1):
            raise ValueError("bandwidth1 must not be NaN")
        elif bandwidth1 is not None and bandwidth1 <= 0:
            raise ValueError("bandwidth1 must be positive.")

        if bandwidth2 is not None and not isinstance(bandwidth2, (float, int)):
            raise ValueError("bandwidth2 must be positive float or integer.")
        elif bandwidth2 is not None and np.isnan(bandwidth2):
            raise ValueError("bandwidth2 must not be NaN")
        elif bandwidth2 is not None and bandwidth2 <= 0:
            raise ValueError("bandwidth2 must be positive.")

        if num_bw_candidates is None or not isinstance(num_bw_candidates, int):
            raise TypeError("Number of bandwidth candidates, num_bw_candidates, should be an integer.")
        if num_bw_candidates < 2:
            raise ValueError("Number of bandwidth candidates, num_bw_candidates, should be at least 2.")

        if num_bw_candidates is None or not isinstance(num_bw_candidates, int):
            raise TypeError("Number of bandwidth candidates, num_bw_candidates, should be an integer.")
        if num_bw_candidates < 2:
            raise ValueError("Number of bandwidth candidates, num_bw_candidates, should be at least 2.")

        xp, *_ = get_namespace_and_device(X, y, sample_weight)

        # Handle input validation
        X = check_array(X, ensure_2d=True, dtype=supported_float_dtypes(xp))
        if X.shape[1] != 2:
            raise ValueError(f"X must have exactly 2 features for 2D model, got {X.shape[1]}")
        y = check_array(y, ensure_2d=False, dtype=X.dtype)
        if y.size != X.shape[0]:
            raise ValueError("y must have the same size as X.")

        self._input_dtype = X.dtype
        self._polyfit2d_func = polyfit2d_f32 if self._input_dtype == np.float32 else polyfit2d_f64

        # Handle sample weights
        if sample_weight is not None:
            sample_weight = check_array(sample_weight, ensure_2d=False, dtype=self._input_dtype)
            if sample_weight.size != y.size:
                raise ValueError(f"sample_weight must have the same length as y, got {sample_weight.size} vs {y.size}")
            if np.any(sample_weight < 0):
                raise ValueError("All sample weights must be non-negative")
        else:
            sample_weight = np.ones_like(y, dtype=self._input_dtype)

        # Store fitted data
        self.X_ = X
        self.y_ = y
        self.sample_weight_ = sample_weight
        self.n_features_in_ = 2

        # sort training data by X for polyfit2d requirement
        sort_idx = np.lexsort((X[:, 1], X[:, 0]))
        self.sorted_X_ = X[sort_idx, :]
        self.sorted_y_ = y[sort_idx]
        self.sorted_sample_weight_ = sample_weight[sort_idx]

        # Select bandwidths if needed
        if bandwidth1 is None or bandwidth2 is None:
            if bandwidth1 is not None or bandwidth2 is not None:
                raise ValueError("If one bandwidth is provided, both must be provided.")
            if custom_bw_candidates is not None:
                custom_bw_candidates = check_array(custom_bw_candidates, ensure_2d=True, dtype=self._input_dtype)
                if custom_bw_candidates.shape[1] != 2:
                    raise ValueError(f"custom_bw_candidates must have exactly 2 features, got {custom_bw_candidates.shape[1]}")
            self.bandwidth1_, self.bandwidth2_ = self._select_bandwidth(
                num_bw_candidates, bandwidth_selection_method, same_bandwidth_for_2dim, custom_bw_candidates, cv_folds
            )
        else:
            self.bandwidth1_ = float(bandwidth1)
            self.bandwidth2_ = float(bandwidth2)

        # Create interpolation grids
        x1_min, x1_max = np.min(X[:, 0]), np.max(X[:, 0])
        x2_min, x2_max = np.min(X[:, 1]), np.max(X[:, 1])
        if self.obs_grid1_ is not None:
            # Use custom grid for first dimension
            self.obs_grid1_ = check_array(self.obs_grid1_, ensure_2d=False, dtype=self._input_dtype)
            if self.obs_grid1_.ndim != 1:
                raise ValueError("obs_grid1 must be a 1D array")
            if len(self.obs_grid1_) < 2:
                raise ValueError("obs_grid1 must have at least 2 points")
            if np.min(self.obs_grid1_) < x1_min or np.max(self.obs_grid1_) > x1_max:
                raise ValueError(
                    f"obs_grid1 must be within the range of input X[:, 0] [{x1_min:.6f}, {x1_max:.6f}]. "
                    f"Got obs_grid1 range [{np.min(self.obs_grid1_):.6f}, {np.max(self.obs_grid1_):.6f}]"
                )
        else:
            # Create uniform grid for first dimension
            self.obs_grid1_ = np.linspace(x1_min, x1_max, self.n_interp_points)
        if self.obs_grid2_ is not None:
            # Use custom grid for second dimension
            self.obs_grid2_ = check_array(self.obs_grid2_, ensure_2d=False, dtype=self._input_dtype)
            if self.obs_grid2_.ndim != 1:
                raise ValueError("obs_grid2 must be a 1D array")
            if len(self.obs_grid2_) < 2:
                raise ValueError("obs_grid2 must have at least 2 points")
            if np.min(self.obs_grid2_) < x2_min or np.max(self.obs_grid2_) > x2_max:
                raise ValueError(
                    f"obs_grid2 must be within the range of input X[:, 1] [{x2_min:.6f}, {x2_max:.6f}]. "
                    f"Got obs_grid2 range [{np.min(self.obs_grid2_):.6f}, {np.max(self.obs_grid2_):.6f}]"
                )
        else:
            # Create uniform grid for second dimension
            self.obs_grid2_ = np.linspace(x2_min, x2_max, self.n_interp_points)

        try:
            self.obs_fitted_values_ = self._polyfit2d_func(
                np.ascontiguousarray(self.sorted_X_.T),
                self.sorted_y_,
                self.sorted_sample_weight_,
                self.obs_grid1_,
                self.obs_grid2_,
                self.bandwidth1_,
                self.bandwidth2_,
                self.kernel_type.value,
                self.degree,
                self.deriv1,
                self.deriv2,
            )
        except Exception as e:
            raise ValueError(f"Error in polyfit2d: {e!s}") from e

        return self

    def predict(self, X1: np.ndarray, X2: np.ndarray, use_model_interp: bool = True) -> np.ndarray:
        """
        Predict using the 2D polynomial model via interpolation.

        Parameters
        ----------
        X1 : array-like of shape (n_samples,)
            First feature to predict.
        X2 : array-like of shape (n_samples,)
            Second feature to predict.
        use_model_interp : bool, default=True
            If True, use the model's interpolation grid for prediction.
            If False, use direct polyfit2d call for prediction.

        Returns
        -------
        np.ndarray
            Predicted values of shape (n_samples,).
        """
        check_is_fitted(self, ["obs_fitted_values_", "obs_grid1_", "obs_grid2_", "bandwidth1_", "bandwidth2_"])

        # Handle input validation
        X1 = check_array(X1, ensure_2d=False, dtype=self._input_dtype)
        X2 = check_array(X2, ensure_2d=False, dtype=self._input_dtype)
        if X1.ndim != 1 or X2.ndim != 1:
            raise ValueError(f"X1 and X2 must be 1D arrays, got {X1.ndim}D and {X2.ndim}D")
        if X1.shape[0] != X2.shape[0]:
            raise ValueError(f"X1 and X2 must have the same number of samples, got {X1.shape[0]} and {X2.shape[0]}")

        X1_ord = np.argsort(X1)
        X2_ord = np.argsort(X2)
        if use_model_interp:
            try:
                y_pred = interp2d(self.obs_grid1_, self.obs_grid2_, self.obs_fitted_values_, X1[X1_ord], X2[X2_ord], self.interp_kind)
            except Exception as e:
                raise ValueError(f"Error during interpolation: {e!s}") from e
        else:
            try:
                y_pred = self._polyfit2d_func(
                    np.ascontiguousarray(self.sorted_X_.T),
                    self.sorted_y_,
                    self.sorted_sample_weight_,
                    X1[X1_ord],
                    X2[X2_ord],
                    self.bandwidth1_,
                    self.bandwidth2_,
                    self.kernel_type.value,
                    self.degree,
                    self.deriv1,
                    self.deriv2,
                )
            except Exception as e:
                raise ValueError(f"Error in polyfit2d: {e!s}") from e

        inverse_X1_sort_idx = np.argsort(X1_ord)
        inverse_X2_sort_idx = np.argsort(X2_ord)
        return y_pred[inverse_X1_sort_idx, :][:, inverse_X2_sort_idx]

    def get_fitted_grids(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the fitted values at the interpolation grid points.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            A tuple containing:
            - obs_grid1: The interpolation grid points for the first dimension.
            - obs_grid2: The interpolation grid points for the second dimension.
            - obs_fitted_values: The fitted values at interpolation grid points.
        """
        check_is_fitted(self, ["obs_fitted_values_", "obs_grid1_", "obs_grid2_", "bandwidth1_", "bandwidth2_"])
        return self.obs_grid1_.copy(), self.obs_grid2_.copy(), self.obs_fitted_values_.copy()
