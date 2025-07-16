"""polyfit1d model for 1D and 2D polynomial fitting with kernel smoothing."""

# Authors: Ching-Chuan Chen
# SPDX-License-Identifier: MIT

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils._array_api import get_namespace_and_device, supported_float_dtypes
from sklearn.utils.validation import check_array, check_is_fitted

from pflm.interp import interp1d, interp2d
from pflm.smooth._polyfit import polyfit1d_f32, polyfit1d_f64, polyfit2d_f32, polyfit2d_f64
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
    obs_fitted_values : ndarray
        The fitted values at interpolation grid points.
    bandwidth_ : float
        The selected bandwidth after fitting.
    cv_scores_ : ndarray, optional
        CV/GCV scores for each bandwidth candidate.
    bandwidth_candidates_ : ndarray, optional
        Bandwidth candidates evaluated during selection.
    """

    def __init__(
        self,
        *,
        kernel_type=KernelType.GAUSSIAN,
        degree=1,
        deriv=0,
        obs_grid=None,
        n_interp_points=100,
        interp_kind="linear",
    ):
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

    def _generate_bandwidth_candidates(self, X, y, sample_weight):
        """
        Generate bandwidth candidates for selection.

        Parameters
        ----------
        X : array-like
            Input data.
        y : array-like
            Target values.
        sample_weight : array-like
            Sample weights.

        Returns
        -------
        candidates : ndarray
            Array of bandwidth candidates.
        """
        # TODO: Implement bandwidth candidate generation
        # This is a placeholder - you'll provide the actual implementation
        return np.linspace(0.1, 10.0, 100)  # Example candidates

    def _compute_cv_score(self, X, y, sample_weight, bandwidth, cv_folds=5):
        """
        Compute cross-validation score for a given bandwidth.

        Parameters
        ----------
        X : array-like
            Input data.
        y : array-like
            Target values.
        sample_weight : array-like
            Sample weights.
        bandwidth : float
            Bandwidth parameter.
        cv_folds : int, default=5
            Number of cross-validation folds.

        Returns
        -------
        cv_score : float
            Cross-validation score.
        """
        # TODO: Implement CV score calculation
        # This is a placeholder - you'll provide the actual implementation
        return np.random.rand()  # Example random score

    def _compute_gcv_score(self, X, y, sample_weight, bandwidth):
        """
        Compute Generalized Cross-Validation score for a given bandwidth.

        Parameters
        ----------
        X : array-like
            Input data.
        y : array-like
            Target values.
        sample_weight : array-like
            Sample weights.
        bandwidth : float
            Bandwidth parameter.

        Returns
        -------
        gcv_score : float
            GCV score.
        """
        # TODO: Implement GCV score calculation
        # This is a placeholder - you'll provide the actual implementation
        return np.random.rand()  # Example random score

    def _select_bandwidth(self, X, y, sample_weight, method="gcv", cv_folds=5):
        """
        Select bandwidth using cross-validation.

        Parameters
        ----------
        X : array-like
            Input data.
        y : array-like
            Target values.
        sample_weight : array-like
            Sample weights.
        method : str, default='gcv'
            Method for bandwidth selection. Options: 'cv', 'gcv'.
            If 'cv', uses cross-validation; if 'gcv', uses Generalized Cross-Validation.
        cv_folds : int, default=5
            Number of cross-validation folds.

        Returns
        -------
        best_bandwidth : float
            The bandwidth with the best score.
        """
        self.bandwidth_candidates_ = self._generate_bandwidth_candidates(X, y, sample_weight)

        if method == "cv":
            cv_scores = np.array([self._compute_cv_score(X, y, sample_weight, bw, cv_folds) for bw in self.bandwidth_candidates_])
        elif method == "gcv":
            cv_scores = np.array([self._compute_gcv_score(X, y, sample_weight, bw) for bw in self.bandwidth_candidates_])
        else:
            raise ValueError(f"Invalid method '{method}'. Use 'cv' or 'gcv'.")

        # Store for inspection
        self.cv_scores_ = cv_scores

        # Return bandwidth with minimum CV score (assuming lower is better)
        best_idx = np.argmin(cv_scores)
        return self.bandwidth_candidates_[best_idx]

    def fit(self, X, y, sample_weight=None, bandwidth=None, bandwidth_selection="gcv", cv_folds=5):
        """
        Fit the 1D polynomial model.

        Parameters
        ----------
        X: array - like of shape(n_samples, 1) or (n_samples,)
            Training data.
        y: array - like of shape(n_samples,)
            Target values.
        sample_weight: array - like of shape(n_samples,), default = None
            Sample weights.
        bandwidth: float, default = None
            The bandwidth parameter for kernel smoothing. If None, will be selected
            using the method specified in bandwidth_selection.
        bandwidth_selection: str, default = 'gcv'
            Method for bandwidth selection. Options: 'cv', 'gcv'.
        cv_folds: int, default = 5
            Number of cross - validation folds(only used if bandwidth_selection='cv').

        Returns
        -------
        self: object
            Returns self.
        """
        # Validate bandwidth_selection
        if bandwidth_selection not in ["cv", "gcv"]:
            raise ValueError(f"bandwidth_selection must be one of ['cv', 'gcv'], got {bandwidth_selection}")

        if bandwidth is not None and not isinstance(bandwidth, (float, int)):
            raise ValueError("bandwidth must be positive float or integer.")
        elif bandwidth is not None and np.isnan(bandwidth):
            raise ValueError("bandwidth must not be NaN")
        elif bandwidth is not None and bandwidth <= 0:
            raise ValueError("bandwidth must be positive.")

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

        self._input_dtype = X.dtype

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

        # Select bandwidth if needed
        if bandwidth is None:
            self.bandwidth_ = self._select_bandwidth(X, y, sample_weight, bandwidth_selection, cv_folds)
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

        # Sort training data by X for polyfit1d requirement
        self._sort_idx = np.argsort(X)
        X_sorted = X[self._sort_idx]
        y_sorted = y[self._sort_idx]
        sample_weight_sorted = sample_weight[self._sort_idx]

        # Fit polynomial at grid points
        self._polyfit1d_func = polyfit1d_f32 if self._input_dtype == np.float32 else polyfit1d_f64
        try:
            self.obs_fitted_values_ = self._polyfit1d_func(
                X_sorted, y_sorted, sample_weight_sorted, self.obs_grid_, self.bandwidth_, self.kernel_type.value, self.degree, self.deriv
            )
        except Exception as e:
            raise ValueError(f"Error in polyfit1d: {e!s}") from e

        return self

    def predict(self, X, use_model_interp=True):
        """
        Predict using the 1D polynomial model via interpolation.

        Parameters
        ----------
        X: array - like of shape(n_samples, 1) or (n_samples,)
            Samples to predict.
        use_model_interp: bool, default=True
            If True, use the model's interpolation grid for prediction.
            If False, use direct polyfit1d call for prediction.

        Returns
        -------
        y_pred: ndarray of shape(n_samples,)
            Predicted values.
        """
        check_is_fitted(self)
        if not hasattr(self, "obs_fitted_values_"):
            raise ValueError("Model must be fitted before calling predict.")

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
            X_sorted = self.X_[self._sort_idx]
            y_sorted = self.y_[self._sort_idx]
            sample_weight_sorted = self.sample_weight_[self._sort_idx]
            # Predict using direct polyfit1d call
            try:
                y_pred = self._polyfit1d_func(
                    X_sorted, y_sorted, sample_weight_sorted, X[ord], self.bandwidth_, self.kernel_type.value, self.degree, self.deriv
                )
            except Exception as e:
                raise ValueError(f"Error in polyfit1d: {e!s}") from e

        inverse_sort_idx = np.argsort(ord)
        return y_pred[inverse_sort_idx]

    def get_fitted_grids(self):
        """
        Get the fitted values at the interpolation grid points.

        Returns
        -------
        obs_grid: ndarray
            The interpolation grid points.
        interp_grid: ndarray
            The fitted values at interpolation grid points.
        """
        check_is_fitted(self)
        return self.obs_grid_.copy(), self.obs_fitted_values_.copy()

    def _more_tags(self):
        return {
            "requires_y": True,
            "requires_fit": True,
            "X_types": ["1darray", "2darray"],
            "y_types": ["1darray"],
            "no_validation": False,
        }


class Polyfit2DModel(BaseEstimator, RegressorMixin):
    """
    2D polynomial fitting model with kernel smoothing.

    Parameters
    ----------
    kernel_type : KernelType, default = KernelType.GAUSSIAN
        The type of kernel to use for smoothing.
    degree : int, default = 1
        The degree of the polynomial.
    deriv1 : int, default = 0
        The derivative order for the first dimension.
    deriv2 : int, default = 0
        The derivative order for the second dimension.
    obs_grid1 : array-like, default=None
        Custom grid points for interpolation in the first dimension. If None, will create uniform grid.
        Must be within the range of the first dimension of input X.
    obs_grid2 : np.ndarray, default = None
        Custom grid points for interpolation in the second dimension. If None, will create uniform grid.
        Must be within the range of the second dimension of input Y.
    n_interp_points: int, default = 100
        Number of points for interpolation grid in each dimension(only used if obs_grid is None).
    interp_kind: str, default = 'linear'
        Interpolation method('linear', 'cubic', 'quintic').

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
    bandwidth1_: float
        The selected bandwidth for the first dimension after fitting.
    bandwidth2_: float
        The selected bandwidth for the second dimension after fitting.
    cv_scores_: ndarray, optional
        CV scores for each bandwidth candidate pair.
    bandwidth_candidates_: ndarray, optional
        Bandwidth candidate pairs evaluated during selection.
    """

    def __init__(
        self,
        *,
        kernel_type=KernelType.GAUSSIAN,
        degree=1,
        deriv1=0,
        deriv2=0,
        obs_grid1=None,
        obs_grid2=None,
        n_interp_points=100,
        interp_kind="linear",
    ):
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

    def _generate_bandwidth_candidates(self, X, y, sample_weight):
        """
        Generate bandwidth candidates for 2D selection.

        Parameters
        ----------
        X: array - like
            Input data.
        y: array - like
            Target values.
        sample_weight: array - like
            Sample weights.

        Returns
        -------
        candidates: ndarray
            Array of bandwidth candidate pairs(bandwidth1, bandwidth2).
        """
        # TODO: Implement 2D bandwidth candidate generation
        # This is a placeholder - you'll provide the actual implementation
        return zip(*[np.linspace(0.1, 10.0, 100), np.linspace(0.1, 10.0, 100)])  # example candidates

    def _compute_cv_score(self, X, y, sample_weight, bandwidth1, bandwidth2, cv_folds=5):
        """
        Compute cross - validation score for given bandwidths.

        Parameters
        ----------
        X: array - like
            Input data.
        y: array - like
            Target values.
        sample_weight: array - like
            Sample weights.
        bandwidth1: float
            Bandwidth for first dimension.
        bandwidth2: float
            Bandwidth for second dimension.
        cv_folds: int, default = 5
            Number of cross - validation folds.

        Returns
        -------
        cv_score: float
            Cross - validation score.
        """
        # TODO: Implement 2D CV score calculation
        # This is a placeholder - you'll provide the actual implementation
        return np.random.rand()  # Example random score

    def _compute_gcv_score(self, X, y, sample_weight, bandwidth1, bandwidth2):
        """
        Compute Generalized Cross - Validation score for given bandwidths.

        Parameters
        ----------
        X: array - like
            Input data.
        y: array - like
            Target values.
        sample_weight: array - like
            Sample weights.
        bandwidth1: float
            Bandwidth for first dimension.
        bandwidth2: float
            Bandwidth for second dimension.

        Returns
        -------
        gcv_score: float
            GCV score.
        """
        # TODO: Implement 2D GCV score calculation
        # This is a placeholder - you'll provide the actual implementation
        return np.random.rand()  # Example random score

    def _select_bandwidth(self, X, y, sample_weight, method="gcv", cv_folds=5):
        """
        Select bandwidths using cross - validation.

        Parameters
        ----------
        X: array - like
            Input data.
        y: array - like
            Target values.
        sample_weight: array - like
            Sample weights.
        method : str, default='gcv'
            Method for bandwidth selection. Options: 'cv', 'gcv'.
            If 'cv', uses cross-validation; if 'gcv', uses Generalized Cross-Validation.
        cv_folds : int, default=5
            Number of cross-validation folds.

        Returns
        -------
        best_bandwidth : float
            The bandwidth with the best score.
        """
        self.bandwidth_candidates_ = self._generate_bandwidth_candidates(X, y, sample_weight)

        if method == "cv":
            cv_scores = np.array([self._compute_cv_score(X, y, sample_weight, bw1, bw2, cv_folds) for bw1, bw2 in self.bandwidth_candidates_])
        elif method == "gcv":
            cv_scores = np.array([self._compute_gcv_score(X, y, sample_weight, bw1, bw2) for bw1, bw2 in self.bandwidth_candidates_])
        else:
            raise ValueError(f"Invalid method '{method}'. Use 'cv' or 'gcv'.")

        # Store for inspection
        self.cv_scores_ = cv_scores

        # Return bandwidths with minimum CV score
        best_idx = np.argmin(cv_scores)
        return self.bandwidth_candidates_[best_idx]

    def fit(self, X, y, sample_weight=None, bandwidth1=None, bandwidth2=None, bandwidth_selection="gcv", cv_folds=5):
        """
        Fit the 2D polynomial model.

        Parameters
        ----------
        X: array - like of shape(n_samples, 2)
            Training data.
        y: array - like of shape(n_samples,)
            Target values.
        sample_weight: array - like of shape(n_samples,), default = None
            Sample weights.
        bandwidth1: float, default = None
            The bandwidth parameter for the first dimension.
        bandwidth2: float, default = None
            The bandwidth parameter for the second dimension.
        bandwidth_selection: str, default = 'gcv'
            Method for bandwidth selection. Options: 'cv', 'gcv'.
        cv_folds: int, default = 5
            Number of cross - validation folds(only used if bandwidth_selection='cv').

        Returns
        -------
        self: object
            Returns self.
        """
        # Validate bandwidth_selection
        if bandwidth_selection not in ["cv", "gcv"]:
            raise ValueError(f"bandwidth_selection must be one of ['cv', 'gcv'], got {bandwidth_selection}")

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

        xp, *_ = get_namespace_and_device(X, y, sample_weight)

        # Handle both (n_samples,) and (n_samples, 1) input shapes
        X = check_array(X, ensure_2d=True, dtype=supported_float_dtypes(xp))
        y = check_array(y, ensure_2d=False, dtype=X.dtype)
        if y.size != X.shape[0]:
            raise ValueError("y must have the same size as X.")

        self._input_dtype = X.dtype

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

        # Select bandwidths if needed
        if bandwidth1 is None or bandwidth2 is None:
            if bandwidth1 is not None or bandwidth2 is not None:
                raise ValueError("If one bandwidth is provided, both must be provided.")
            self.bandwidth1_, self.bandwidth2_ = self._select_bandwidth(X, y, sample_weight, bandwidth_selection, cv_folds)
        else:
            if bandwidth_selection == "cv":
                self.bandwidth1_, self.bandwidth2_ = self._select_bandwidth(X, y, sample_weight, bandwidth_selection, cv_folds)
            elif bandwidth_selection == "gcv":
                self.bandwidth1_, self.bandwidth2_ = self._select_bandwidth(X, y, sample_weight, bandwidth_selection)
            else:
                bandwidth1 = float(bandwidth1)
                bandwidth2 = float(bandwidth2)

        # Create interpolation grid
        x1_min, x1_max = np.min(X[:, 0]), np.max(X[:, 0])
        x2_min, x2_max = np.min(X[:, 1]), np.max(X[:, 1])

        if self.obs_grid1_ is not None and self.obs_grid2_ is not None:
            # Use custom grid
            self.obs_grid1_ = check_array(self.obs_grid1_, dtype=self._input_dtype)
            self.obs_grid2_ = check_array(self.obs_grid2_, dtype=self._input_dtype)
            if self.obs_grid1_.ndim != 2 or self.obs_grid1_.shape[1] != 2:
                raise ValueError("obs_grid1_ must be a 2D array with shape (n_points, 2)")
            if len(self.obs_grid1_) < 4:
                raise ValueError("obs_grid1_ must have at least 4 points")
            if np.isnan(self.obs_grid1_).any():
                raise ValueError("obs_grid1_ contains NaN values")

            if self.obs_grid2_.ndim != 2 or self.obs_grid2_.shape[1] != 2:
                raise ValueError("obs_grid2_ must be a 2D array with shape (n_points, 2)")
            if len(self.obs_grid2_) < 4:
                raise ValueError("obs_grid2_ must have at least 4 points")
            if np.isnan(self.obs_grid2_).any():
                raise ValueError("obs_grid2_ contains NaN values")

            # Check if grid is within the range of input X
            if np.min(self.obs_grid1_) < x1_min or np.max(self.obs_grid1_) > x1_max:
                raise ValueError(
                    f"obs_grid must be within the range of input X. "
                    f"X1 range: [{x1_min:.6f}, {x1_max:.6f}], "
                    f"Got obs_grid1_ X1 range: [{np.min(self.obs_grid1_):.6f}, {np.max(self.obs_grid1_):.6f}]"
                )
            if np.min(self.obs_grid2_) < x2_min or np.max(self.obs_grid2_) > x2_max:
                raise ValueError(
                    f"obs_grid must be within the range of input X. "
                    f"X2 range: [{x2_min:.6f}, {x2_max:.6f}]. "
                    f"Got obs_grid2_ X2 range: [{np.min(self.obs_grid2_):.6f}, {np.max(self.obs_grid2_):.6f}]"
                )
        else:
            if self.obs_grid1_ is None or self.obs_grid2_ is None:
                raise ValueError("Both obs_grid1_ and obs_grid2_ must be provided or both must be None.")
            # Create uniform grid within the range of input X
            self.obs_grid1_ = np.linspace(x1_min, x1_max, self.n_interp_points, dtype=self._input_dtype)
            self.obs_grid2_ = np.linspace(x2_min, x2_max, self.n_interp_points, dtype=self._input_dtype)

        # sort training data by X for polyfit2d requirement
        self._sort_idx = np.lexsort((X[:, 1], X[:, 0]))
        X_sorted = X[self._sort_idx]
        y_sorted = y[self._sort_idx]
        sample_weight_sorted = sample_weight[self._sort_idx]

        # Fit polynomial at grid points
        self._polyfit2d_func = polyfit2d_f32 if self._input_dtype == np.float32 else polyfit2d_f64
        try:
            self.obs_fitted_values_ = self._polyfit2d_func(
                X_sorted,
                y_sorted,
                sample_weight_sorted,
                self.obs_grid1_,
                self.obs_grid2_,
                self.bandwidth1_,
                self.bandwidth2_,
                self.kernel_type,
                self.degree,
                self.deriv1,
                self.deriv2,
            )
        except Exception as e:
            raise ValueError(f"Error in polyfit2d: {e!s}") from e

        return self

    def predict(self, x_new1, x_new2, use_model_interp=True):
        """
        Predict using the 2D polynomial model via interpolation.

        Parameters
        ----------
        x_new1 : array-like of shape(n_samples1,)
            Samples to predict in the first dimension.
        x_new2 : array-like of shape(n_samples2,)
            Samples to predict in the second dimension.
        use_model_interp : bool, default=True
            If True, use the model's interpolation grid for prediction.
            If False, use direct polyfit2d call for prediction.

        Returns
        -------
        y_pred: ndarray of shape(n_samples1, n_samples2)
            Predicted values at the specified grid points.
        """
        check_is_fitted(self)
        if not hasattr(self, "obs_fitted_values_"):
            raise ValueError("Model must be fitted before calling predict.")

        # Validate input
        x_new1 = check_array(x_new1, ensure_2d=False, dtype=self._input_dtype)
        x_new2 = check_array(x_new2, ensure_2d=False, dtype=self._input_dtype)
        if x_new1.ndim != 1 or x_new2.ndim != 1:
            raise ValueError("x_new1 and x_new2 must be 1D arrays")
        if np.isnan(x_new1).any() or np.isnan(x_new2).any():
            raise ValueError("Input arrays x_new1 and x_new2 contain NaN values")

        if use_model_interp:
            # Use pflm.interp.interp2d for interpolation
            try:
                y_pred = interp2d(self.obs_grid1_, self.obs_grid2_, self.obs_fitted_values_, x_new1, x_new2, self.interp_method)
            except Exception as e:
                raise ValueError(f"Error during interpolation: {e!s}") from e
        else:
            # Predict using direct polyfit2d call
            X_sorted = self.X_[self._sort_idx]
            y_sorted = self.y_[self._sort_idx]
            sample_weight_sorted = self.sample_weight_[self._sort_idx]
            try:
                y_pred = self._polyfit2d_func(
                    X_sorted,
                    y_sorted,
                    sample_weight_sorted,
                    x_new1,
                    x_new2,
                    self.bandwidth1_,
                    self.bandwidth2_,
                    self.kernel_type.value,
                    self.degree,
                    self.deriv1,
                    self.deriv2,
                )
            except Exception as e:
                raise ValueError(f"Error in polyfit2d: {e!s}") from e

        return y_pred

    def get_fitted_grids(self):
        """
        Get the interpolation grid points and values.

        Returns
        -------
        obs_grid1: ndarray
            The interpolation grid points in the first dimension.
        obs_grid2: ndarray
            The interpolation grid points in the second dimension.
        obs_fitted_value: ndarray
            The fitted values at interpolation grid points.
        """
        check_is_fitted(self)
        return self.obs_grid1_.copy(), self.obs_grid2_.copy(), self.obs_fitted_values_.copy()

    def _more_tags(self):
        return {
            "requires_y": True,
            "requires_fit": True,
            "X_types": ["2darray"],
            "y_types": ["1darray"],
            "no_validation": False,
        }
