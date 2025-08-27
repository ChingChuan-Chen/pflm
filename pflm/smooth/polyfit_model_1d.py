"""polyfit1d model for 1D and 2D polynomial fitting with kernel smoothing."""

# Authors: Ching-Chuan Chen
# SPDX-License-Identifier: MIT

import math
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils._array_api import get_namespace_and_device, supported_float_dtypes
from sklearn.utils.validation import check_array, check_is_fitted

from pflm.interp import interp1d
from pflm.smooth import KernelType
from pflm.smooth._polyfit import calculate_kernel_value_f32, calculate_kernel_value_f64, polyfit1d_f32, polyfit1d_f64


class Polyfit1DModel(BaseEstimator, RegressorMixin):
    """
    1D polynomial fitting with kernel smoothing and fast interpolation.

    This estimator performs local polynomial regression using a selectable kernel.
    It can choose bandwidth by CV/GCV and predicts efficiently via interpolation.

    Parameters
    ----------
    kernel_type : KernelType, default=KernelType.GAUSSIAN
        Kernel used for smoothing.
    degree : int, default=1
        Polynomial degree (>= 1).
    deriv : int, default=0
        Derivative order (0 <= deriv <= degree).
    interp_kind : {"linear", "spline"}, default="linear"
        Interpolation method used for fast prediction.
    random_seed : int, optional
        Random seed for reproducibility (e.g., CV shuffles).

    Attributes
    ----------
    X_ : np.ndarray of shape (n_samples,)
        Training inputs (sorted copy stored as `sorted_X_`).
    y_ : np.ndarray of shape (n_samples,)
        Training targets (sorted copy stored as `sorted_y_`).
    sample_weight_ : np.ndarray of shape (n_samples,)
        Sample weights (sorted copy stored as `sorted_sample_weight_`).
    n_features_in_ : int
        Number of features during fit (always 1 for 1D).
    reg_grid_ : np.ndarray of shape (m,)
        Interpolation grid used for fast predictions.
    reg_fitted_values_ : np.ndarray of shape (m,)
        Fitted values evaluated on `reg_grid_`.
    obs_grid_ : np.ndarray of shape (n_obs_grid,)
        Unique observed grid from `X_`.
    bandwidth_ : float
        Selected/used bandwidth.
    bandwidth_selection_results_ : dict
        Selection details: candidates, method, scores, and chosen bandwidth.

    See Also
    --------
    Polyfit2DModel : Local polynomial regression in 2D.
    """

    def __init__(
        self,
        kernel_type: KernelType = KernelType.GAUSSIAN,
        degree: int = 1,
        deriv: int = 0,
        interp_kind: Literal["linear", "spline"] = "linear",
        random_seed: Optional[int] = None,
    ) -> None:
        """Initialize the 1D polynomial model.

        Parameters
        ----------
        kernel_type : KernelType, default=KernelType.GAUSSIAN
        degree : int, default=1
        deriv : int, default=0
        interp_kind : {"linear", "spline"}, default="linear"
        random_seed : int, optional
        """
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
        self.interp_kind = interp_kind
        self.rng = np.random.default_rng(random_seed)

    def _generate_bandwidth_candidates(self, num_bw_candidates: int) -> np.ndarray:
        """Generate sorted bandwidth candidates on a log scale.

        Parameters
        ----------
        num_bw_candidates : int
            Number of candidates to generate (prefer odd to include a center).

        Returns
        -------
        np.ndarray of shape (num_bw_candidates,)
            Monotonically increasing bandwidth candidates.
        """
        if self.obs_grid_.size <= self.degree + 1:
            raise ValueError(f"Not enough unique support points ({self.obs_grid_.size}) to fit a polynomial of degree {self.degree}.")

        lag = self.degree + 1
        d_star = np.max(self.obs_grid_[lag:] - self.obs_grid_[:-lag])
        r = self.obs_grid_[-1] - self.obs_grid_[0]
        h0 = min(1.5 * d_star, r)
        q = math.pow(0.25 * r / h0, 1.0 / (num_bw_candidates - 1))
        return h0 * (q ** np.linspace(0, num_bw_candidates - 1, num_bw_candidates, dtype=self._input_dtype))

    def _compute_cv_score(self, bandwidth: np.floating, cv_folds: int = 5) -> np.floating:
        """Compute K-fold cross-validation score for a given bandwidth.

        Parameters
        ----------
        bandwidth : float
            Bandwidth to evaluate.
        cv_folds : int, default=5
            Number of folds.

        Returns
        -------
        float
            Average CV loss (lower is better).
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
                self.obs_grid_,
                bandwidth,
                self.kernel_type.value,
                self.degree,
                self.deriv,
            )

            if np.isnan(y_pred).any():
                return np.inf

            cv_scores[fold] = sum(
                self.sorted_sample_weight_[test_mask] * (self.sorted_y_[test_mask] - y_pred[self.obs_grid_idx_[test_mask]]) ** 2
            )

        return np.mean(cv_scores) / self._sum_sample_weight

    def _compute_gcv_score(self, bandwidth: np.floating, k0: float, r: np.floating) -> np.floating:
        """Compute GCV score given a bandwidth and smoothing constants.

        Parameters
        ----------
        bandwidth : float
            Bandwidth to evaluate.
        k0 : float
            Kernel value at zero.
        r : float
            Range of the observed unique grid.

        Returns
        -------
        float
            GCV score (lower is better).
        """

        y_pred = self._polyfit1d_func(
            self.sorted_X_,
            self.sorted_y_,
            self.sorted_sample_weight_,
            self.obs_grid_,
            bandwidth,
            self.kernel_type.value,
            self.degree,
            self.deriv,
        )
        if np.isnan(y_pred).any():
            return np.inf

        # GCV score = n * RSS / tr(I - S)^2, where RSS is the residual sum of squares
        # x_i belongs [a, b], kernel is symmetric, so S_{ii} will be approximately K(0) / (h * n / r) = K(0) * r / (h * n)
        # where h is bandwidth, n is number of points, r is range of x_i
        # So, tr(S) = sum_i S_{ii} = n * K(0) * r / (h * n) = K(0) * r / h, then tr(I - S) = n - tr(S)
        # Last, the GCV score can be computed as n * RSS / (n - K(0) * r / h) ^ 2 = RSS / n / (1 - K(0) * r / (h * n)) ^ 2
        # We substitute n with the sum of sample weights to account for weighted regression
        # and we can ignore the n factor in the denominator for the comparison.
        rss = sum(self.sorted_sample_weight_ * (self.sorted_y_ - y_pred[self.obs_grid_idx_]) ** 2)
        trace_s = k0 * r / bandwidth
        denominator = math.pow(max(1.0 - trace_s / self._sum_sample_weight, 0.0), 2.0)
        return rss / denominator if denominator > 0 else np.inf

    def _select_bandwidth(
        self,
        num_bw_candidates: int = 21,
        method: Literal["cv", "gcv"] = "gcv",
        custom_bw_candidates: Optional[np.ndarray] = None,
        cv_folds: int = 5,
    ) -> np.floating:
        """Select bandwidth via CV or GCV.

        Parameters
        ----------
        num_bw_candidates : int, default=21
            Used only if `custom_bw_candidates` is None.
        method : {"cv", "gcv"}, default="gcv"
            Selection method.
        custom_bw_candidates : np.ndarray, optional
            External candidate list.
        cv_folds : int, default=5
            Number of folds if using CV.

        Returns
        -------
        float
            Best bandwidth according to the chosen criterion.
        """
        if custom_bw_candidates is not None:
            bandwidth_candidates_ = check_array(custom_bw_candidates, ensure_2d=False, dtype=self._input_dtype)
        else:
            bandwidth_candidates_ = self._generate_bandwidth_candidates(num_bw_candidates)

        # Store for inspection
        self.bandwidth_selection_results_ = {
            "bandwidth_candidates": bandwidth_candidates_,
            "bandwidth_selection_method": method,
        }

        self._sum_sample_weight = np.sum(self.sample_weight_)
        if method == "cv":
            cv_scores = np.array([self._compute_cv_score(bw, cv_folds) for bw in bandwidth_candidates_])
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
            r = self.obs_grid_[-1] - self.obs_grid_[0]
            gcv_scores = np.array([self._compute_gcv_score(bw, k0, r) for bw in bandwidth_candidates_])
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
        reg_grid: Optional[Union[np.ndarray, List[float]]] = None,
        num_bw_candidates: int = 21,
        bandwidth_selection_method: Literal["cv", "gcv"] = "gcv",
        num_points_reg_grid: int = 100,
        custom_bw_candidates: Optional[np.ndarray] = None,
        cv_folds: int = 5,
    ) -> "Polyfit1DModel":
        """Fit the 1D local polynomial model with kernel smoothing.

        Parameters
        ----------
        X : array-like of shape (n_samples,) or (n_samples, 1)
            Training inputs.
        y : array-like of shape (n_samples,)
            Training targets.
        sample_weight : array-like of shape (n_samples,), optional
            Sample weights.
        bandwidth : float, optional
            If not provided, selected by `bandwidth_selection_method`.
        reg_grid : array-like of shape (m,), optional
            Interpolation grid. If None, a uniform grid is created.
        num_bw_candidates : int, default=21
        bandwidth_selection_method : {"cv", "gcv"}, default="gcv"
        num_points_reg_grid : int, default=100
            Number of points for the internal interpolation grid (used if `reg_grid` is None).
        custom_bw_candidates : np.ndarray, optional
        cv_folds : int, default=5

        Returns
        -------
        Polyfit1DModel
            Fitted estimator (self).

        Raises
        ------
        ValueError
            If inputs are invalid or selection arguments are inconsistent.
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

        if reg_grid is None:
            if num_points_reg_grid is None or not isinstance(num_points_reg_grid, int):
                raise TypeError("Number of points for interpolation grid, num_points_reg_grid, should be an integer.")
        if bandwidth_selection_method == "cv":
            if cv_folds is None or not isinstance(cv_folds, int):
                raise TypeError("Number of cross-validation folds, cv_folds, should be an integer.")

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

        # Create interpolation grid
        x_min, x_max = np.min(X), np.max(X)

        # create reg_grid_
        if reg_grid is not None:
            # Use custom grid
            self.reg_grid_ = check_array(reg_grid, ensure_2d=False, dtype=self._input_dtype)
            if self.reg_grid_.ndim != 1:
                raise ValueError("reg_grid must be a 1D array")
            if len(self.reg_grid_) < 2:
                raise ValueError("reg_grid must have at least 2 points")

            # Check if grid is within the range of input X
            if np.min(self.reg_grid_) < x_min or np.max(self.reg_grid_) > x_max:
                raise ValueError(
                    f"reg_grid must be within the range of input X [{x_min:.6f}, {x_max:.6f}]. "
                    f"Got reg_grid range [{np.min(self.reg_grid_):.6f}, {np.max(self.reg_grid_):.6f}]"
                )
        else:
            # Create uniform grid within the range of input X
            self.reg_grid_ = np.linspace(x_min, x_max, num_points_reg_grid, dtype=self._input_dtype)

        # create obs_grid_
        self.obs_grid_, self.obs_grid_idx_ = np.unique(self.sorted_X_, return_inverse=True, sorted=True)

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

        try:
            self.reg_fitted_values_ = self._polyfit1d_func(
                self.sorted_X_,
                self.sorted_y_,
                self.sorted_sample_weight_,
                self.reg_grid_,
                self.bandwidth_,
                self.kernel_type.value,
                self.degree,
                self.deriv,
            )
        except Exception as e:
            raise ValueError(f"Error in polyfit1d: {e!s}") from e

        return self

    def predict(self, X: Union[np.ndarray, List[float]], use_model_interp: bool = True) -> np.ndarray:
        """Predict responses at new inputs.

        Parameters
        ----------
        X : array-like of shape (m,) or (m, 1)
            Query points.
        use_model_interp : bool, default=True
            If True, interpolate from `reg_grid_`; if False, evaluate directly.

        Returns
        -------
        np.ndarray of shape (m,)
            Predicted values.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If the model is not fitted.
        """
        check_is_fitted(self, ["reg_fitted_values_", "reg_grid_", "bandwidth_"])

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
                y_pred = interp1d(self.reg_grid_, self.reg_fitted_values_, X[ord], self.interp_kind)
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

    def fitted_values(self) -> np.ndarray:
        """
        Return fitted values on the interpolation grid.

        Returns
        -------
        np.ndarray of shape (len(reg_grid_),)
            Fitted values evaluated on ``reg_grid_``.
        """
        check_is_fitted(self, ["reg_fitted_values_", "reg_grid_", "bandwidth_"])
        return self.reg_fitted_values_.copy()

    def get_fitted_grids(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return observation and interpolation grids.

        Returns
        -------
        obs_grid : np.ndarray of shape (n_obs_grid,)
            Unique observed grid used during fit.
        reg_grid : np.ndarray of shape (m,)
            Interpolation grid used for fast predictions.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If the model is not fitted.
        """
        check_is_fitted(self, ["reg_fitted_values_", "reg_grid_", "bandwidth_"])
        return self.reg_grid_.copy(), self.reg_fitted_values_.copy()
