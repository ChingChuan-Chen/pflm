"""polyfit1d model for 1D and 2D polynomial fitting with kernel smoothing."""

# Authors: Ching-Chuan Chen
# SPDX-License-Identifier: MIT

import math
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils._array_api import get_namespace_and_device, supported_float_dtypes
from sklearn.utils.validation import check_array, check_is_fitted

from pflm.interp import interp2d
from pflm.smooth import KernelType
from pflm.smooth._polyfit import calculate_kernel_value_f32, calculate_kernel_value_f64, polyfit2d_f32, polyfit2d_f64


class Polyfit2DModel(BaseEstimator, RegressorMixin):
    """Local polynomial (2D) regression with kernel smoothing and interpolation.

    Parameters
    ----------
    kernel_type : KernelType, default=KernelType.GAUSSIAN
        Kernel used for smoothing.
    degree : int, default=1
        Polynomial degree (>= 1).
    deriv1 : int, default=0
        Derivative order along the first dimension (>= 0).
    deriv2 : int, default=0
        Derivative order along the second dimension (>= 0).
    interp_kind : {"linear", "spline"}, default="linear"
        Interpolation method used for fast prediction.
    random_seed : int, optional
        Random seed for reproducibility (e.g., CV shuffles).

    Attributes
    ----------
    X_ : np.ndarray of shape (n_samples, 2)
        Training inputs.
    y_ : np.ndarray of shape (n_samples,)
        Training targets.
    sample_weight_ : np.ndarray of shape (n_samples,)
        Sample weights.
    reg_grid1_ : np.ndarray of shape (m1,)
        Interpolation grid along the first dimension.
    reg_grid2_ : np.ndarray of shape (m2,)
        Interpolation grid along the second dimension.
    reg_fitted_values_ : np.ndarray of shape (m1, m2)
        Fitted values evaluated on the interpolation grid mesh.
    obs_grid1_ : np.ndarray of shape (n_obs_grid1,)
        Unique observed grid from X[:, 0].
    obs_grid2_ : np.ndarray of shape (n_obs_grid2,)
        Unique observed grid from X[:, 1].
    bandwidth1_ : float
        Selected/used bandwidth for the first dimension.
    bandwidth2_ : float
        Selected/used bandwidth for the second dimension.
    bandwidth_selection_results_ : dict
        Selection details including candidates, method, and chosen pair.


    See Also
    --------
    Polyfit1DModel : Local polynomial regression in 1D.
    """

    def __init__(
        self,
        kernel_type: KernelType = KernelType.GAUSSIAN,
        degree: int = 1,
        deriv1: int = 0,
        deriv2: int = 0,
        interp_kind: Literal["linear", "spline"] = "linear",
        random_seed: Optional[int] = None,
    ) -> None:
        """Initialize the 2D polynomial model.

        Parameters
        ----------
        kernel_type : KernelType, default=KernelType.GAUSSIAN
        degree : int, default=1
        deriv1 : int, default=0
        deriv2 : int, default=0
        interp_kind : {"linear", "spline"}, default="linear"
        random_seed : int, optional
        """
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
        self.interp_kind = interp_kind
        self.rng = np.random.default_rng(random_seed)

    def _get_bandwidth_candidates(self, sorted_unique_support: np.ndarray, num_bw_candidates: int, lag: int) -> np.ndarray:
        """Compute a 1D set of bandwidth candidates from a sorted support.

        Parameters
        ----------
        sorted_unique_support : np.ndarray of shape (n_unique,)
            Sorted unique coordinates.
        num_bw_candidates : int
            Number of candidates.
        lag : int
            Spacing used to estimate a representative distance.

        Returns
        -------
        np.ndarray of shape (num_bw_candidates,)
            Monotonically increasing bandwidth candidates.
        """
        d_star = np.max(sorted_unique_support[lag:] - sorted_unique_support[:-lag])
        r = sorted_unique_support[-1] - sorted_unique_support[0]
        h0 = min(2.0 * d_star, r)
        q = math.pow(0.25 * r / h0, 1.0 / (num_bw_candidates - 1))
        return h0 * (q ** np.linspace(0, num_bw_candidates - 1, num_bw_candidates, dtype=self._input_dtype))

    def _generate_bandwidth_candidates(self, num_bw_candidates: int, same_bandwidth_for_2dim: bool = False) -> np.ndarray:
        """Generate 2D bandwidth candidates for selection.

        Parameters
        ----------
        num_bw_candidates : int
            Number of candidates per axis.
        same_bandwidth_for_2dim : bool, default=False
            If True, enforce identical bandwidths on both axes.

        Returns
        -------
        np.ndarray
            If `same_bandwidth_for_2dim` is True, shape (1, num_bw_candidates)
            containing shared bandwidths; otherwise shape (2, num_bw_candidates)
            with per-axis candidates.
        """
        if self.obs_grid1_.shape[0] <= self.degree + 1:
            raise ValueError(f"Not enough unique support points ({self.obs_grid1_.shape[0]}) to fit a polynomial of degree {self.degree}.")
        if self.obs_grid2_.shape[0] <= self.degree + 1:
            raise ValueError(f"Not enough unique support points ({self.obs_grid2_.shape[0]}) to fit a polynomial of degree {self.degree}.")

        # return bandwidth candidates
        bw1_candidates = self._get_bandwidth_candidates(self.obs_grid1_, num_bw_candidates, self.degree + 1)
        bw2_candidates = self._get_bandwidth_candidates(self.obs_grid2_, num_bw_candidates, self.degree + 1)

        # return candidates
        if same_bandwidth_for_2dim:
            return np.vstack((bw1_candidates, bw2_candidates))
        else:
            bw1v, bw2v = np.meshgrid(bw1_candidates, bw2_candidates)
            return np.vstack((bw1v.ravel(), bw2v.ravel()))

    def _compute_cv_score(
        self,
        bandwidth1: np.floating,
        bandwidth2: np.floating,
        cv_folds: int = 5,
    ) -> np.floating:
        """Compute K-fold CV score for a bandwidth pair.

        Parameters
        ----------
        bandwidth1 : float
            Bandwidth for the first dimension.
        bandwidth2 : float
            Bandwidth for the second dimension.
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

            y_pred = self._polyfit2d_func(
                np.ascontiguousarray(self.sorted_X_[train_mask, :].T),
                self.sorted_y_[train_mask],
                self.sorted_sample_weight_[train_mask],
                self.obs_grid1_,
                self.obs_grid2_,
                bandwidth1,
                bandwidth2,
                self.kernel_type.value,
                self.degree,
                self.deriv1,
                self.deriv2,
            )

            if np.isnan(y_pred).any():
                return np.inf

            cv_scores[fold] = np.sum(
                self.sorted_sample_weight_[test_mask]
                * (self.sorted_y_[test_mask] - y_pred[self.obs_grid1_idx_[test_mask], self.obs_grid2_idx_[test_mask]]) ** 2
            )

        return np.mean(cv_scores) / self._sum_sample_weight

    def _compute_gcv_score(
        self,
        bandwidth1: np.floating,
        bandwidth2: np.floating,
        k0: float,
        r1: np.floating,
        r2: np.floating,
    ) -> np.floating:
        """Compute GCV score for a bandwidth pair with smoothing constants.

        Parameters
        ----------
        bandwidth1 : float
        bandwidth2 : float
        k0 : float
            Kernel value at zero.
        r1 : float
            Range of the observed unique grid along axis-1.
        r2 : float
            Range of the observed unique grid along axis-2.

        Returns
        -------
        float
            GCV score (lower is better).
        """
        y_pred = self._polyfit2d_func(
            np.ascontiguousarray(self.sorted_X_.T),
            self.sorted_y_,
            self.sorted_sample_weight_,
            self.obs_grid1_,
            self.obs_grid2_,
            bandwidth1,
            bandwidth2,
            self.kernel_type.value,
            self.degree,
            self.deriv1,
            self.deriv2,
        )
        if np.isnan(y_pred).any():
            return np.inf

        # As in 1D, we compute the GCV score as follows:
        # GCV score = n * RSS / tr(I - S)^2, where RSS is the residual sum of squares
        # In 1D, it was an approximation of the trace of S in the first-order accuracy.
        # In 2D, we use the second-order accuracy approximation.
        # tr(S) = n * K(0) * K(0) * r1 * r2 / (h1 * h2 * n^2) * 3
        # tr(I - S) = n - tr(S)
        # Last, the GCV score can be computed as n * RSS / (n - tr(S))^2 = RSS / n / (1 - tr(S) / n)^2
        # We substitute n with the sum of sample weights to account for weighted regression
        # and we can ignore the n factor in the denominator for the comparison.
        rss = np.sum((self.sorted_y_ - y_pred[self.obs_grid1_idx_, self.obs_grid2_idx_]) ** 2 * self.sorted_sample_weight_)
        trace_s = k0 * k0 * r1 * r2 / bandwidth1 / bandwidth2 * 3
        denominator = math.pow(max(1.0 - trace_s / self._sum_sample_weight, 0.0), 2.0)
        return rss / denominator if denominator > 0 else np.inf

    def _select_bandwidth(
        self,
        num_bw_candidates: int = 21,
        method: Literal["cv", "gcv"] = "gcv",
        same_bandwidth_for_2dim: bool = False,
        custom_bw_candidates: Optional[np.ndarray] = None,
        cv_folds: int = 5,
    ) -> Tuple[float, float]:
        """Select 2D bandwidths via CV or GCV.

        Parameters
        ----------
        num_bw_candidates : int, default=21
        method : {"cv", "gcv"}, default="gcv"
        same_bandwidth_for_2dim : bool, default=False
        custom_bw_candidates : np.ndarray, optional
        cv_folds : int, default=5

        Returns
        -------
        Tuple[float, float]
            Best (bandwidth1, bandwidth2) pair.
        """
        if custom_bw_candidates is not None:
            bandwidth_candidates_ = check_array(custom_bw_candidates, ensure_2d=True, dtype=self._input_dtype)
            if same_bandwidth_for_2dim:
                bandwidth_candidates_ = bandwidth_candidates_[bandwidth_candidates_[:, 0] == bandwidth_candidates_[:, 1], :]
        else:
            bandwidth_candidates_ = self._generate_bandwidth_candidates(num_bw_candidates, same_bandwidth_for_2dim)

        # Store for inspection
        self.bandwidth_selection_results_ = {
            "bandwidth_candidates": bandwidth_candidates_,
            "bandwidth_selection_method": method,
        }

        self._sum_sample_weight = np.sum(self.sample_weight_)
        if method == "cv":
            cv_scores = np.array(
                [
                    self._compute_cv_score(bandwidth_candidates_[i, 0], bandwidth_candidates_[i, 1], cv_folds)
                    for i in range(bandwidth_candidates_.shape[0])
                ]
            )
            if (~np.isfinite(cv_scores)).all():
                raise ValueError("All CV scores are non-finite. Check your data and bandwidth candidates.")
            self.bandwidth_selection_results_["cv_scores"] = cv_scores
            self.bandwidth_selection_results_["best_bandwidth"] = bandwidth_candidates_[np.argmin(cv_scores), :]
            self.bandwidth_selection_results_["cv_folds"] = cv_folds
        elif method == "gcv":
            k0 = (
                calculate_kernel_value_f32(0, self.kernel_type.value)
                if self._input_dtype == np.float32
                else calculate_kernel_value_f64(0, self.kernel_type.value)
            )
            r1 = self.obs_grid1_[-1] - self.obs_grid1_[0]
            r2 = self.obs_grid2_[-1] - self.obs_grid2_[0]
            gcv_scores = np.array(
                [
                    self._compute_gcv_score(bandwidth_candidates_[i, 0], bandwidth_candidates_[i, 1], k0, r1, r2)
                    for i in range(bandwidth_candidates_.shape[0])
                ]
            )
            if (~np.isfinite(gcv_scores)).all():
                raise ValueError("All GCV scores are non-finite. Check your data and bandwidth candidates.")
            self.bandwidth_selection_results_["gcv_scores"] = gcv_scores
            self.bandwidth_selection_results_["best_bandwidth"] = bandwidth_candidates_[np.argmin(gcv_scores), :]

        return self.bandwidth_selection_results_["best_bandwidth"][0], self.bandwidth_selection_results_["best_bandwidth"][1]

    def fit(
        self,
        X: Union[np.ndarray, List[List[float]]],
        y: Union[np.ndarray, List[float]],
        sample_weight: Optional[Union[np.ndarray, List[float]]] = None,
        bandwidth1: Optional[float] = None,
        bandwidth2: Optional[float] = None,
        reg_grid1: Optional[Union[np.ndarray, List[float]]] = None,
        reg_grid2: Optional[Union[np.ndarray, List[float]]] = None,
        num_bw_candidates: int = 21,
        bandwidth_selection_method: Literal["cv", "gcv"] = "gcv",
        num_points_reg_grid: int = 100,
        same_bandwidth_for_2dim: bool = False,
        custom_bw_candidates: Optional[np.ndarray] = None,
        cv_folds: int = 5,
    ):
        """Fit the 2D local polynomial model with kernel smoothing.

        Parameters
        ----------
        X : array-like of shape (n_samples, 2)
            Training inputs.
        y : array-like of shape (n_samples,)
            Training targets.
        sample_weight : array-like of shape (n_samples,), optional
            Sample weights.
        bandwidth1 : float, optional
            Bandwidth for the first axis.
        bandwidth2 : float, optional
            Bandwidth for the second axis.
        reg_grid1 : array-like of shape (m1,), optional
            Interpolation grid on axis-1. If None, a uniform grid is created.
        reg_grid2 : array-like of shape (m2,), optional
            Interpolation grid on axis-2. If None, a uniform grid is created.
        num_bw_candidates : int, default=21
        bandwidth_selection_method : {"cv", "gcv"}, default="gcv"
        num_points_reg_grid : int, default=100
            Number of points for each axis of the internal interpolation grid
            (used if `reg_grid1`/`reg_grid2` are None).
        same_bandwidth_for_2dim : bool, default=False
        custom_bw_candidates : np.ndarray, optional
        cv_folds : int, default=5

        Returns
        -------
        Polyfit2DModel
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

        if reg_grid1 is None or reg_grid2 is None:
            if num_points_reg_grid is None or not isinstance(num_points_reg_grid, int):
                raise TypeError("Number of points for interpolation grid, num_points_reg_grid, should be an integer.")
        if bandwidth_selection_method == "cv":
            if cv_folds is None or not isinstance(cv_folds, int):
                raise TypeError("Number of cross-validation folds, cv_folds, should be an integer.")

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

        # Create regular grids
        x1_min, x1_max = np.min(X[:, 0]), np.max(X[:, 0])
        x2_min, x2_max = np.min(X[:, 1]), np.max(X[:, 1])
        if reg_grid1 is not None:
            # Use custom grid for first dimension
            self.reg_grid1_ = check_array(reg_grid1, ensure_2d=False, dtype=self._input_dtype)
            if self.reg_grid1_.ndim != 1:
                raise ValueError("reg_grid1 must be a 1D array")
            if len(self.reg_grid1_) < 2:
                raise ValueError("reg_grid1 must have at least 2 points")
            if np.min(self.reg_grid1_) < x1_min or np.max(self.reg_grid1_) > x1_max:
                raise ValueError(
                    f"reg_grid1 must be within the range of input X[:, 0] [{x1_min:.6f}, {x1_max:.6f}]. "
                    f"Got reg_grid1 range [{np.min(self.reg_grid1_):.6f}, {np.max(self.reg_grid1_):.6f}]"
                )
        else:
            # Create uniform grid for first dimension
            self.reg_grid1_ = np.linspace(x1_min, x1_max, num_points_reg_grid, dtype=self._input_dtype)

        if reg_grid2 is not None:
            # Use custom grid for second dimension
            self.reg_grid2_ = check_array(reg_grid2, ensure_2d=False, dtype=self._input_dtype)
            if self.reg_grid2_.ndim != 1:
                raise ValueError("reg_grid2 must be a 1D array")
            if len(self.reg_grid2_) < 2:
                raise ValueError("reg_grid2 must have at least 2 points")
            if np.min(self.reg_grid2_) < x2_min or np.max(self.reg_grid2_) > x2_max:
                raise ValueError(
                    f"reg_grid2 must be within the range of input X[:, 1] [{x2_min:.6f}, {x2_max:.6f}]. "
                    f"Got reg_grid2 range [{np.min(self.reg_grid2_):.6f}, {np.max(self.reg_grid2_):.6f}]"
                )
        else:
            # Create uniform grid for second dimension
            self.reg_grid2_ = np.linspace(x2_min, x2_max, num_points_reg_grid, dtype=self._input_dtype)

        # create observation grids
        self.obs_grid1_, self.obs_grid1_idx_ = np.unique(self.sorted_X_[:, 0], return_inverse=True, sorted=True)
        self.obs_grid2_, self.obs_grid2_idx_ = np.unique(self.sorted_X_[:, 1], return_inverse=True, sorted=True)

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

        try:
            self.reg_fitted_values_ = self._polyfit2d_func(
                np.ascontiguousarray(self.sorted_X_.T),
                self.sorted_y_,
                self.sorted_sample_weight_,
                self.reg_grid1_,
                self.reg_grid2_,
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
        """Predict responses on a 2D grid.

        Parameters
        ----------
        X1 : np.ndarray of shape (n_x1,)
            Query points along axis-1.
        X2 : np.ndarray of shape (n_x2,)
            Query points along axis-2.
        use_model_interp : bool, default=True
            If True, interpolate from the fitted grid; if False, evaluate directly.

        Returns
        -------
        np.ndarray of shape (n_x2, n_x1)
            Predicted values (note the row-major order).

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If the model is not fitted.
        """
        check_is_fitted(self, ["reg_fitted_values_", "reg_grid1_", "reg_grid2_", "bandwidth1_", "bandwidth2_"])

        # Handle input validation
        X1 = check_array(X1, ensure_2d=False, dtype=self._input_dtype)
        X2 = check_array(X2, ensure_2d=False, dtype=self._input_dtype)
        if X1.ndim != 1 or X2.ndim != 1:
            raise ValueError(f"X1 and X2 must be 1D arrays, got {X1.ndim}D and {X2.ndim}D")

        X1_ord = np.argsort(X1)
        X2_ord = np.argsort(X2)
        if use_model_interp:
            try:
                y_pred = interp2d(self.reg_grid1_, self.reg_grid2_, self.reg_fitted_values_, X1[X1_ord], X2[X2_ord], self.interp_kind)
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

    def fitted_values(self) -> np.ndarray:
        """
        Return fitted values on the model's 2D interpolation grid.

        Returns
        -------
        np.ndarray of shape (len(``reg_grid1_``) * len(``reg_grid2_``),)
            Fitted values evaluated on the 2D interpolation mesh.
        """
        check_is_fitted(self, ["reg_fitted_values_", "reg_grid1_", "reg_grid2_", "bandwidth1_", "bandwidth2_"])
        return self.reg_fitted_values_.copy()

    def get_fitted_grids(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return interpolation grids and fitted values.

        Returns
        -------
        reg_grid1 : np.ndarray of shape (m1,)
            Copy of ``reg_grid1_``.
        reg_grid2 : np.ndarray of shape (m2,)
            Copy of ``reg_grid2_``.
        reg_fitted_values : np.ndarray of shape (m1, m2)
            Copy of ``reg_fitted_values_``.
        """
        check_is_fitted(self, ["reg_fitted_values_", "reg_grid1_", "reg_grid2_", "bandwidth1_", "bandwidth2_"])
        return self.reg_grid1_.copy(), self.reg_grid2_.copy(), self.reg_fitted_values_.copy()
