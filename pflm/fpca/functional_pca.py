"""Functional Principal Component Analysis (FPCA) implementation."""

# Authors: Ching-Chuan Chen
# SPDX-License-Identifier: MIT

import time
import warnings
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils._array_api import get_namespace_and_device, supported_float_dtypes
from sklearn.utils.validation import check_array, check_is_fitted

from pflm.interp import interp1d, interp2d
from pflm.smooth import KernelType, Polyfit1DModel, Polyfit2DModel
from pflm.utils import FlattenFunctionalData, flatten_and_sort_data_matrices, trapz
from pflm.fpca import (
    FpcaModelParams,
    SmoothedModelResult,
    estimate_rho,
    get_covariance_matrix,
    get_eigen_analysis_results,
    get_eigenvalue_fit,
    get_fpca_ce_score,
    get_fpca_in_score,
    get_fpca_phi,
    get_measurement_error_variance,
    get_raw_cov,
    select_num_pcs_fve,
)


class FunctionalPCAMuCovParams:
    """
    Parameters for mean and covariance functions in Functional PCA.

    Parameters
    ----------
    bw_mu : float, optional
        Bandwidth for the mean function. If None, it will be estimated.
    bw_cov : float, optional
        Bandwidth for the covariance function. If None, it will be estimated.
    estimate_method : {'smooth', 'cross-sectional'}, default='smooth'
        Method to estimate the mean and covariance functions.
    kernel_type : KernelType, default=KernelType.EPANECHNIKOV
        Type of kernel to use for smoothing.
    method_select_mu_bw : {'cv', 'gcv'}, default='gcv'
        Method to select bandwidth for the mean function.
    method_select_cov_bw : {'cv', 'gcv'}, default='gcv'
        Method to select bandwidth for the covariance function.
    apply_geo_avg_cov_bw : bool, default=False
        Whether to apply geometric averaging when selecting covariance bandwidth.
    cv_folds_mu : int, default=10
        Number of folds for cross-validation when selecting bandwidth for the mean function.
    cv_folds_cov : int, default=10
        Number of folds for cross-validation when selecting bandwidth for the covariance function.
    random_seed : int, optional
        Random seed for reproducibility which is used only in CV. If None, no seed is set.
    """

    def __init__(
        self,
        bw_mu: Optional[float] = None,
        bw_cov: Optional[float] = None,
        estimate_method: Literal["smooth", "cross-sectional"] = "smooth",
        kernel_type: KernelType = KernelType.EPANECHNIKOV,
        method_select_mu_bw: Literal["cv", "gcv"] = "gcv",
        method_select_cov_bw: Literal["cv", "gcv"] = "gcv",
        apply_geo_avg_cov_bw: bool = False,
        cv_folds_mu: int = 10,
        cv_folds_cov: int = 10,
        random_seed: Optional[int] = None,
    ):
        """Validate and store smoothing-related parameters.

        Parameters
        ----------
        bw_mu : float, optional
            Positive bandwidth for the mean smoother, or None to auto-select.
        bw_cov : float, optional
            Positive bandwidth for the covariance smoother, or None to auto-select.
        estimate_method : {'smooth', 'cross-sectional'}, default='smooth'
            Mean/covariance estimation approach.
        kernel_type : KernelType, default=KernelType.EPANECHNIKOV
            Kernel for local regression.
        method_select_mu_bw : {'cv', 'gcv'}, default='gcv'
            Bandwidth selection for mean.
        method_select_cov_bw : {'cv', 'gcv'}, default='gcv'
            Bandwidth selection for covariance.
        apply_geo_avg_cov_bw : bool, default=False
            Whether to geometric-average candidate bandwidths for covariance.
        cv_folds_mu : int, default=10
            K-folds for CV when selecting mean bandwidth.
        cv_folds_cov : int, default=10
            K-folds for CV when selecting covariance bandwidth.
        random_seed : int, optional
            Random seed used in CV (if applicable).

        Raises
        ------
        ValueError
            If any parameter violates its expected type or range.
        """
        if bw_mu is not None and bw_mu <= 0:
            raise ValueError("Bandwidth for mean function bw_mu must be a positive scalar.")
        if bw_cov is not None and bw_cov <= 0:
            raise ValueError("Bandwidth for covariance function bw_cov must be a positive scalar.")
        if estimate_method not in ["smooth", "cross-sectional"]:
            raise ValueError("estimate_method must be either 'smooth' or 'cross-sectional'.")
        if not isinstance(kernel_type, KernelType):
            raise ValueError("kernel_type must be an instance of KernelType Enum.")
        if method_select_mu_bw not in ["gcv", "cv"]:
            raise ValueError("method_select_mu_bw must be either 'gcv' or 'cv'.")
        if method_select_cov_bw not in ["gcv", "cv"]:
            raise ValueError("method_select_cov_bw must be either 'gcv' or 'cv'.")
        if not isinstance(apply_geo_avg_cov_bw, bool):
            raise ValueError("apply_geo_avg_cov_bw must be a boolean value.")
        if cv_folds_mu <= 0 or not isinstance(cv_folds_mu, int):
            raise ValueError("cv_folds_mu must be a positive integer.")
        if cv_folds_cov <= 0 or not isinstance(cv_folds_cov, int):
            raise ValueError("cv_folds_cov must be a positive integer.")
        if random_seed is not None and (not isinstance(random_seed, int) or random_seed < 0):
            raise ValueError("random_seed must be a non-negative integer.")

        self.bw_mu = bw_mu
        self.bw_cov = bw_cov
        self.estimate_method = estimate_method
        self.kernel_type = kernel_type
        self.method_select_mu_bw = method_select_mu_bw
        self.method_select_cov_bw = method_select_cov_bw
        self.apply_geo_avg_cov_bw = apply_geo_avg_cov_bw
        self.cv_folds_mu = cv_folds_mu
        self.cv_folds_cov = cv_folds_cov
        self.random_seed = random_seed

    def __repr__(self):
        """Return a concise representation of parameters for logging/debugging."""
        return (
            f"FPCASmoothingParams(bw_mu={self.bw_mu}, bw_cov={self.bw_cov}, estimate_method='{self.estimate_method}', "
            f"kernel_type={self.kernel_type}, method_select_mu_bw='{self.method_select_mu_bw}', "
            f"method_select_cov_bw='{self.method_select_cov_bw}', apply_geo_avg_cov_bw={self.apply_geo_avg_cov_bw}, "
            f"k_fold_mu={self.cv_folds_mu}, k_fold_cov={self.cv_folds_cov}, random_seed={self.random_seed})"
        )


class FunctionalPCAUserDefinedParams:
    """
    User-defined parameters for Functional PCA.

    Parameters
    ----------
    t_mu : np.ndarray or List[float], optional
        Time points for the mean function. If provided, must match the length of `mu`.
    mu : np.ndarray or List[float], optional
        Mean function values at the time points in `t_mu`.
    t_cov : np.ndarray or List[float], optional
        Time points for the covariance function. If provided, must match the dimensions of `cov`.
    cov : np.ndarray or List[List[float]], optional
        Covariance function values at the time points in `t_cov`.
    sigma2 : float, optional
        Variance of the measurement error.
    rho : float, optional
        The user-defined measurement truncation threshold used for conditional expectations estimation on the principal component scores.
        If provided, must be a non-negative scalar.
    """

    def __init__(
        self,
        t_mu: Union[np.ndarray, List[float]] = None,
        mu: Union[np.ndarray, List[float]] = None,
        t_cov: Union[np.ndarray, List[float]] = None,
        cov: Union[np.ndarray, List[List[float]]] = None,
        sigma2: Optional[float] = None,
        rho: Optional[float] = None,
    ):
        """Validate shapes/types for optional user-specified artifacts.

        Parameters
        ----------
        t_mu, mu : array-like, optional
            1D arrays with equal length if provided; both must be given together.
        t_cov : array-like, optional
            1D array of grid points for covariance; must match `cov` dimensions.
        cov : array-like, optional
            2D array of covariance values with square shape compatible with `t_cov`.
        sigma2 : float, optional
            Non-negative measurement error variance.
        rho : float, optional
            Non-negative rho used for CE scoring if provided.

        Raises
        ------
        ValueError
            If only one of (t_mu, mu) is provided, or only one of (t_cov, cov) is
            provided, or if shapes/types are inconsistent, or if sigma2/rho are invalid.
        """
        if t_mu is not None and mu is not None:
            if len(t_mu) != len(mu):
                raise ValueError("t_mu and mu must have the same length.")
        else:
            if t_mu is not None or mu is not None:
                raise ValueError("Both t_mu and mu must be provided together.")

        if t_cov is not None and cov is not None:
            if len(t_cov) != len(cov) or len(t_cov) != len(cov[0]):
                raise ValueError("t_cov must match the dimensions of cov.")
        else:
            if t_cov is not None or cov is not None:
                raise ValueError("Both t_cov and cov must be provided together.")

        if sigma2 is not None and (not isinstance(sigma2, (int, float)) or sigma2 < 0):
            raise ValueError("Variance sigma2 must be a non-negative scalar.")

        if rho is not None and (not isinstance(rho, (int, float)) or rho < 0):
            raise ValueError("Correlation rho must be a non-negative scalar.")

        self.t_mu = t_mu
        self.mu = mu
        self.t_cov = t_cov
        self.cov = cov
        self.sigma2 = float(sigma2) if sigma2 is not None else None
        self.rho = float(rho) if rho is not None else None

    def __repr__(self):
        """Return a compact string representation of user-defined parameters."""
        cov_repr = " ".join(list(map(lambda x: x.strip(), repr(self.cov).split("\n")))) if self.cov is not None else "None"
        return (
            f"FunctionalPCAUserDefinedParams(t_mu={self.t_mu!r}, mu={self.mu!r}, "
            f"t_cov={self.t_cov!r}, cov={cov_repr}, sigma2={self.sigma2}, rho={self.rho})"
        )


class FunctionalPCA(BaseEstimator):
    """
    Functional Principal Component Analysis (FPCA) for functional data.

    Parameters
    ----------
    assume_measurement_error : bool, default=True
        Whether to estimate and account for measurement error.
    num_points_reg_grid : int, default=51
        Number of points in the internal regular grid used for regression/interpolation.
    mu_cov_params : FunctionalPCAMuCovParams, default=FunctionalPCAMuCovParams()
        Smoothing and bandwidth selection settings.
    user_params : FunctionalPCAUserDefinedParams, default=FunctionalPCAUserDefinedParams()
        Optional user-specified mean/covariance/sigma2/rho to override parts of the pipeline.
    verbose : bool, default=False
        If True, record and expose timing diagnostics in `elapsed_time_`.

    Attributes
    ----------
    y_ : List[np.ndarray]
        Functional observations (per-sample vectors).
    t_ : List[np.ndarray]
        Time grids corresponding to `y_` (per-sample vectors).
    flatten_func_data_ : FlattenFunctionalData
        Flattened/validated dataset with fields: y, t, w, tid, unique_tid, inverse_tid_idx, sid, unique_sid, sid_cnt.
    raw_cov_ : np.ndarray of shape (M, 5)
        Raw covariance entries: (sid, t1, t2, w, cov).
    smoothed_model_result_obs_ : SmoothedModelResult
        Smoothed mean/covariance on the observation grid.
    smoothed_model_result_reg_ : SmoothedModelResult
        Smoothed mean/covariance on the regular grid.
    fpca_model_params_ : FpcaModelParams
        FPCA artifacts: eigen decomposition, selected phi/lambda, fitted covariances, rho, eigenvalue fit, etc.
    xi_ : np.ndarray of shape (n_samples, k)
        Estimated principal component scores.
    xi_var_ : np.ndarray or List[np.ndarray]
        Estimated per-sample score variances or covariance summaries (method-dependent).
    fitted_y_mat_ : np.ndarray of shape (nt_obs, n_samples)
        Fitted values on the observation grid (columns are samples).
    fitted_y_ : List[np.ndarray]
        Fitted values at observed time points per subject.
    elapsed_time_ : Dict[str, float]
        Timings (seconds) per pipeline stage.

    See Also
    --------
    FunctionalPCAMuCovParams
    FunctionalPCAUserDefinedParams
    """

    def __init__(
        self,
        assume_measurement_error: bool = True,
        num_points_reg_grid: int = 51,
        mu_cov_params: FunctionalPCAMuCovParams = FunctionalPCAMuCovParams(),
        user_params: FunctionalPCAUserDefinedParams = FunctionalPCAUserDefinedParams(),
        verbose: bool = False,
    ) -> None:
        """Initialize estimator and validate top-level configuration.

        Parameters
        ----------
        assume_measurement_error : bool, default=True
            If True, estimate sigma2 (unless user provides).
        num_points_reg_grid : int, default=51
            Size of the internal regular grid used for smoothing/interpolation.
        mu_cov_params : FunctionalPCAMuCovParams
            Smoothing hyperparameters and selection policies.
        user_params : FunctionalPCAUserDefinedParams
            Optional user-specified mean/cov/sigma2/rho inputs.
        verbose : bool, default=False
            If True, enable timing diagnostics.

        Raises
        ------
        ValueError
            If any parameter violates expected type/range consistency.
        """
        # Initialize parameters
        self.assume_measurement_error = assume_measurement_error
        self.num_points_reg_grid = num_points_reg_grid
        self.mu_cov_params = mu_cov_params
        self.user_params = user_params
        self.verbose = verbose

        # Validate user-defined sigma2 and rho
        if assume_measurement_error and user_params.sigma2 is not None and user_params.sigma2 > 0:
            raise ValueError(
                "Measurement error is assumed to be false, but user-defined sigma2 is provided and greater than 0. "
                + "Please set assume_measurement_error to True or set sigma2 to None or 0."
            )
        if assume_measurement_error and user_params.rho is not None and user_params.rho > 0:
            raise ValueError(
                "Measurement error is assumed to be true, but user-defined rho is provided and greater than 0. "
                + "Please set assume_measurement_error to True or set rho to None or 0."
            )

    def __check_fit_params(
        self,
        num_samples: int,
        method_pcs: Literal["IN", "CE"] = "CE",
        method_select_num_pcs: Union[int, Literal["FVE", "AIC", "BIC"]] = "FVE",
        method_rho: Literal["trunc", "ridge", "vanilla"] = "vanilla",
        max_num_pcs: Optional[int] = None,
        if_impute_scores: bool = True,
        if_shrinkage: bool = False,
        if_fit_eigen_values: bool = False,
        fve_threshold: float = 0.99,
    ):
        """Validate scoring and selection hyperparameters before computation.

        Parameters
        ----------
        num_samples : int
            Number of samples in the dataset (used for bounds).
        method_pcs : {"IN", "CE"}, default="CE"
            Score computation method (In-sample or Conditional Expectation).
        method_select_num_pcs : int or {"FVE", "AIC", "BIC"}, default="FVE"
            Strategy to select the number of components or a fixed integer.
        method_rho : {"trunc", "ridge", "vanilla"}, default="vanilla"
            Rho strategy for CE scoring ("trunc" = truncate sigma2, "ridge" = ridge parameter).
        max_num_pcs : int, optional
            Upper bound for PCs; inferred if None.
        if_impute_scores : bool, default=True
            Whether to compute and impute scores during fit.
        if_shrinkage : bool, default=False
            Whether to apply shrinkage for IN scores.
        if_fit_eigen_values : bool, default=False
            Whether to compute a regression fit for eigenvalues.
        fve_threshold : float, default=0.99
            FVE threshold in (0, 1].

        Raises
        ------
        ValueError
            If any parameter violates expected type/range or allowed choices.
        """
        if method_pcs not in ["IN", "CE"]:
            raise ValueError("method_pcs must be either 'IN' (Numerical Integration) or 'CE' (Conditional Expectation).")
        if not isinstance(method_select_num_pcs, (int, str)):
            raise ValueError("method_select_num_pcs must be either a positive integer or one of 'FVE', 'AIC', 'BIC'.")
        if isinstance(method_select_num_pcs, int) and method_select_num_pcs <= 0:
            raise ValueError("If method_select_num_pcs is an integer, it must be a positive integer.")
        if isinstance(method_select_num_pcs, str) and method_select_num_pcs not in ["FVE", "AIC", "BIC"]:
            raise ValueError("method_select_num_pcs must be one of 'FVE', 'AIC', 'BIC' or a positive integer.")
        if method_rho not in ["truncated", "ridge", "vanilla"]:
            raise ValueError("method_rho must be one of 'truncated', 'ridge', 'vanilla'.")
        if max_num_pcs is not None and (not isinstance(max_num_pcs, int) or max_num_pcs <= 0):
            raise ValueError("max_num_pcs must be a positive integer.")
        if max_num_pcs is None:
            self.max_num_pcs = min(num_samples - 2, self.num_points_reg_grid - 2)
        if not isinstance(fve_threshold, float) or fve_threshold <= 0 or fve_threshold > 1:
            raise ValueError("fve_threshold must be a float between 0 and 1.")
        if not isinstance(if_impute_scores, bool):
            raise ValueError("if_impute_scores must be a boolean value.")
        if not isinstance(if_shrinkage, bool):
            raise ValueError("if_shrinkage must be a boolean value.")
        if not isinstance(if_fit_eigen_values, bool):
            raise ValueError("if_fit_eigen_values must be a boolean value.")

    def fit(
        self,
        t: List[Union[np.ndarray, List[float]]],
        y: List[Union[np.ndarray, List[float]]],
        w: Optional[List[Union[np.ndarray, List[float]]]] = None,
        method_pcs: Literal["IN", "CE"] = "CE",
        method_select_num_pcs: Union[int, Literal["FVE", "AIC", "BIC"]] = "FVE",
        method_rho: Literal["trunc", "ridge", "vanilla"] = "vanilla",
        max_num_pcs: Optional[int] = None,
        if_impute_scores: bool = True,
        if_shrinkage: bool = False,
        if_fit_eigen_values: bool = False,
        fve_threshold: float = 0.99,
        reg_grid: Union[np.ndarray, List[float]] = None,
    ) -> "FunctionalPCA":
        """Fit the FPCA model: mean, covariance, eigenstructure, and scores.

        Parameters
        ----------
        t : list of array-like
            Time vectors per sample; each is 1D with shape (n_i,).
        y : list of array-like
            Observations per sample; each is 1D with shape (n_i,).
        w : list of array-like, optional
            Per-sample weights; each is 1D with shape (n_i,) or sample-level weights
            compatible with the flattening utility.
        method_pcs : {"IN", "CE"}, default="CE"
            Score computation method (In-sample or Conditional Expectation).
        method_select_num_pcs : int or {"FVE", "AIC", "BIC"}, default="FVE"
            Number of PCs selection strategy or fixed integer.
        method_rho : {"trunc", "ridge", "vanilla"}, default="vanilla"
            Rho strategy for CE (ignored for IN).
        max_num_pcs : int, optional
            Upper bound for PCs; inferred if None.
        if_impute_scores : bool, default=True
            Whether to compute scores during fit.
        if_shrinkage : bool, default=False
            Apply shrinkage for IN scores.
        if_fit_eigen_values : bool, default=False
            Fit eigenvalues by projection regression.
        fve_threshold : float, default=0.99
            FVE threshold used when method_select_num_pcs="FVE".
        reg_grid : array-like, optional
            Regular grid for smoothing; if None, created uniformly over observed range.

        Returns
        -------
        FunctionalPCA
            The fitted estimator with smoothed artifacts and scores.

        Notes
        -----
        - Requires at least two observations per sample to form covariance pairs.
        - Timing diagnostics are recorded in `elapsed_time_` when `verbose=True`.
        """
        init_start_time = time.time_ns()
        self.num_samples_ = len(y)
        self.__check_fit_params(
            self.num_samples_,
            method_pcs=method_pcs,
            method_select_num_pcs=method_select_num_pcs,
            method_rho=method_rho,
            max_num_pcs=max_num_pcs,
            if_impute_scores=if_impute_scores,
            if_shrinkage=if_shrinkage,
            if_fit_eigen_values=if_fit_eigen_values,
            fve_threshold=fve_threshold,
        )

        if len(y) <= 3:
            warnings.warn("The number of samples is less than or equal to 3. This may lead to unreliable results in functional PCA.")

        y_p, *_ = get_namespace_and_device(t[0], y[0])
        supported_dtype = supported_float_dtypes(y_p)
        tmp = check_array(y[0], ensure_2d=False, dtype=supported_dtype, force_all_finite=False)
        self._input_dtype = tmp.dtype
        self.t_ = []
        self.y_ = []
        for ti, yi in zip(t, y):
            ti = check_array(ti, ensure_2d=False, dtype=self._input_dtype)
            yi = check_array(yi, ensure_2d=False, dtype=self._input_dtype, force_all_finite=False)
            if ti.ndim != 1 or yi.ndim != 1:
                raise ValueError("Each element of t and y must be a 1D array.")
            if len(ti) != len(yi):
                raise ValueError("Each element of t and y must have the same length.")
            self.t_.append(ti)
            self.y_.append(yi)

        # Flatten and sort the data matrices
        self.flatten_func_data_ = flatten_and_sort_data_matrices(self.y_, self.t_, self._input_dtype, w)
        tt = self.flatten_func_data_.t
        obs_grid = self.flatten_func_data_.unique_tid

        if np.any(self.flatten_func_data_.sid_cnt < 2):
            raise ValueError("Each sample must have at least two observations for covariance calculation.")

        # create reg_grid_
        if reg_grid is not None:
            # Use custom grid
            reg_grid = check_array(reg_grid, ensure_2d=False, dtype=self._input_dtype)
            if reg_grid.ndim != 1:
                raise ValueError("reg_grid must be a 1D array")
            if len(reg_grid) < 2:
                raise ValueError("reg_grid must have at least 2 points")

            # Check if grid is within the range of input X
            if np.min(reg_grid) < tt[0] or np.max(reg_grid) > tt[-1]:
                raise ValueError(
                    f"reg_grid must be within the range of input X [{tt[0]:.6f}, {tt[-1]:.6f}]. "
                    f"Got reg_grid range [{np.min(reg_grid):.6f}, {np.max(reg_grid):.6f}]"
                )
        else:
            # Create uniform grid within the range of input X
            reg_grid = np.linspace(tt[0], tt[-1], self.num_points_reg_grid, dtype=self._input_dtype)
        init_time = (time.time_ns() - init_start_time) / 1e9

        # calculate the mean function
        start_time = time.time_ns()
        if self.user_params.t_mu is not None and self.user_params.mu is not None:
            t_mu = check_array(self.user_params.t_mu, ensure_2d=False, dtype=self._input_dtype)
            mu = check_array(self.user_params.mu, ensure_2d=False, dtype=self._input_dtype)
            if t_mu.ndim != 1 or mu.ndim != 1:
                raise ValueError("t_mu and mu must be 1D arrays.")
            if t_mu.size != mu.size:
                raise ValueError("t_mu and mu must have the same length.")
            mu_reg = interp1d(t_mu, mu, reg_grid, method="spline")
            mu_obs = interp1d(t_mu, mu, obs_grid, method="spline")
        elif self.mu_cov_params.estimate_method == "smooth":
            mean_func_model = Polyfit1DModel(
                kernel_type=self.mu_cov_params.kernel_type, interp_kind="spline", random_seed=self.mu_cov_params.random_seed
            )
            mean_func_model.fit(
                self.flatten_func_data_.t,
                self.flatten_func_data_.y,
                self.flatten_func_data_.w,
                bandwidth=self.mu_cov_params.bw_mu,
                reg_grid=reg_grid,
                bandwidth_selection_method=self.mu_cov_params.method_select_mu_bw,
                cv_folds=self.mu_cov_params.cv_folds_mu,
            )
            mu_reg = mean_func_model.fitted_values()
            mu_obs = interp1d(reg_grid, mu_reg, obs_grid, method="spline")
        elif self.mu_cov_params.estimate_method == "cross-sectional":
            mu_obs = (np.bincount(self.flatten_func_data_.tid, self.flatten_func_data_.y) / np.bincount(self.flatten_func_data_.tid)).astype(
                self._input_dtype, copy=False
            )
            mu_reg = interp1d(obs_grid, mu_obs, reg_grid, method="spline")
        mu_time = (time.time_ns() - start_time) / 1e9

        # calculate the covariance function
        start_time = time.time_ns()
        cov_obs = None
        cov_reg = None
        use_user_cov = False
        self.raw_cov_ = get_raw_cov(self.flatten_func_data_, mu_obs)
        if self.user_params.t_cov is not None and self.user_params.cov is not None:
            t_cov = check_array(self.user_params.t_cov, ensure_2d=False, dtype=self._input_dtype)
            cov = check_array(self.user_params.cov, ensure_2d=False, dtype=self._input_dtype)
            if t_cov.ndim != 1 or cov.ndim != 2:
                raise ValueError("t_cov must be a 1D array and cov must be a 2D array.")
            if t_cov.size != cov.shape[0] or t_cov.size != cov.shape[1]:
                raise ValueError("t_cov must match the dimensions of cov.")
            use_user_cov = True
            cov_obs = interp2d(t_cov, t_cov, cov, obs_grid, obs_grid, method="spline")
            cov_reg = interp2d(t_cov, t_cov, cov, reg_grid, reg_grid, method="spline")
        elif self.mu_cov_params.estimate_method == "smooth":
            cov_func_model = Polyfit2DModel(
                kernel_type=self.mu_cov_params.kernel_type, interp_kind="spline", random_seed=self.mu_cov_params.random_seed
            )
            cov_func_model.fit(
                self.raw_cov_[:, [1, 2]],
                self.raw_cov_[:, 4],
                sample_weight=self.raw_cov_[:, 3],
                bandwidth1=self.mu_cov_params.bw_cov,
                bandwidth2=self.mu_cov_params.bw_cov,
                reg_grid1=obs_grid,
                reg_grid2=obs_grid,
                bandwidth_selection_method=self.mu_cov_params.method_select_cov_bw,
                cv_folds=self.mu_cov_params.cv_folds_cov,
            )
            cov_obs = cov_func_model.fitted_values()
            cov_reg = interp2d(obs_grid, obs_grid, cov_obs, reg_grid, reg_grid, method="spline")
        elif self.mu_cov_params.estimate_method == "cross-sectional":
            cov_obs = get_covariance_matrix(self.raw_cov_, obs_grid)
            cov_reg = interp2d(obs_grid, obs_grid, cov_obs, reg_grid, reg_grid, method="spline")
        cov_time = (time.time_ns() - start_time) / 1e9

        start_time = time.time_ns()
        sigma2 = 0.0
        if self.assume_measurement_error and self.user_params.sigma2 is not None:
            sigma2 = float(self.user_params.sigma2)
        elif self.assume_measurement_error and self.mu_cov_params.estimate_method == "smooth":
            if use_user_cov:
                diag_obs_var = np.diagonal(get_covariance_matrix(self.raw_cov_, obs_grid))
                diag_reg_var = interp1d(obs_grid, diag_obs_var, reg_grid)
                sigma2 = np.average(diag_reg_var - np.diagonal(cov_reg))
            else:
                sigma2 = get_measurement_error_variance(
                    self.raw_cov_,
                    reg_grid,
                    cov_func_model.bandwidth1_,
                    self.mu_cov_params.kernel_type,
                )
        elif self.assume_measurement_error and self.mu_cov_params.estimate_method == "cross-sectional":
            # Use user-defined covariance or cross-sectional method
            diff_mask = (self.flatten_func_data_.tid[:-2] - self.flatten_func_data_.tid[2:] == 2) & (
                self.flatten_func_data_.sid[:-2] == self.flatten_func_data_.sid[2:]
            )
            diff_2nd_order = self.flatten_func_data_.y[:-2] - self.flatten_func_data_.y[2:]
            # Calculate sigma2 using the second-order difference method (6.0 = 4C2)
            sigma2 = np.average(diff_2nd_order[diff_mask] ** 2) / 6.0
            if not use_user_cov:
                np.fill_diagonal(cov_obs, cov_obs.diagonal() - sigma2)
        sigma2_time = (time.time_ns() - start_time) / 1e9

        # Create / update smoothed result containers (phi / fitted_cov filled later in fit_score)
        self.smoothed_model_result_obs_ = SmoothedModelResult(grid=obs_grid, mu=mu_obs, cov=cov_obs, grid_type="obs")
        self.smoothed_model_result_reg_ = SmoothedModelResult(grid=reg_grid, mu=mu_reg, cov=cov_reg, grid_type="reg")

        # Create model parameters
        start_time = time.time_ns()
        eig_lambda, eig_vector = get_eigen_analysis_results(self.smoothed_model_result_reg_.cov)
        self.fpca_model_params_ = FpcaModelParams(
            measurement_error_variance=sigma2, eigen_results={"eigenvalues": eig_lambda, "eigenvectors": eig_vector}
        )
        eigen_time = (time.time_ns() - start_time) / 1e9

        self.xi_, self.xi_var_, self.fitted_y_mat_, self.fitted_y_ = self.fit_score(
            method_pcs, method_select_num_pcs, method_rho, max_num_pcs, if_impute_scores, if_shrinkage, if_fit_eigen_values, fve_threshold
        )
        self.elapsed_time_ = {
            "initialization": init_time,
            "mu_estimation": mu_time,
            "cov_estimation": cov_time,
            "measurement_error_variance": sigma2_time,
            "eigen_decomposition": eigen_time,
            "fit_total_time": (time.time_ns() - init_start_time) / 1e9,
        }
        return self

    def fitted_values(self) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Return fitted functional values on grids.

        Returns
        -------
        fitted_y_mat : np.ndarray of shape (nt_obs, n_samples)
            Fitted values on the observation grid.
        fitted_y : List[np.ndarray]
            Fitted values at actually observed time points per subject.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If called before fitting.
        """
        check_is_fitted(self, ["fpca_model_params_", "fitted_y_"])
        return self.fitted_y_mat_, self.fitted_y_

    def predict(
        self,
        y: List[Union[np.ndarray, List[float]]],
        t: List[Union[np.ndarray, List[float]]],
        w: Optional[List[Union[np.ndarray, List[float]]]] = None,
        num_pcs: Optional[int] = None,
    ) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray, List[np.ndarray]]:
        """Predict scores and fitted curves for new observations.

        Parameters
        ----------
        y : list of array-like
            New observations per sample; each is 1D with shape (n_i,).
        t : list of array-like
            Matching time vectors per sample; each is 1D with shape (n_i,).
        w : list of array-like, optional
            Optional weights compatible with the flattening utility.
        num_pcs : int, optional
            Number of PCs to use; if None, uses the number selected during `fit`.

        Returns
        -------
        new_xi : np.ndarray of shape (n_samples, k)
            Predicted FPCA scores.
        new_xi_var : np.ndarray or List[np.ndarray]
            Predicted score variance summaries (method-dependent).
        new_fitted_y_mat : np.ndarray of shape (nt_obs, n_samples)
            Predicted values on the observation grid.
        new_fitted_y : List[np.ndarray]
            Predicted values at actually observed time points per subject.

        Notes
        -----
        - The method of scoring follows the fitted configuration (`method_pcs`).
        - LS/WLS pathways are not implemented and will raise if selected.
        """
        check_is_fitted(self, ["fpca_model_params_", "fitted_y_"])
        new_flatten_func_data = flatten_and_sort_data_matrices(y, t, self._input_dtype, w)

        start_time = time.time()
        new_xi_ = None
        new_xi_var_ = None
        new_fitted_y_mat_ = None
        new_fitted_y_ = None
        if self.fpca_model_params_.method_pcs == "CE":
            sigma2 = (
                self.fpca_model_params_.rho if self.fpca_model_params_.rho is not None else self.fpca_model_params_.measurement_error_variance
            )
            new_xi_, new_xi_var_, new_fitted_y_mat_, new_fitted_y_ = get_fpca_ce_score(
                new_flatten_func_data,
                self.smoothed_model_result_obs_.mu,
                num_pcs,
                self.fpca_model_params_.fpca_lambda,
                self.fpca_model_params_.fpca_phi["obs"],
                self.fpca_model_params_.fitted_covariance["obs"],
                sigma2,
            )
        elif self.fpca_model_params_.method_pcs == "IN":
            new_xi_, new_xi_var_, new_fitted_y_mat_, new_fitted_y_ = get_fpca_in_score(
                new_flatten_func_data,
                self.smoothed_model_result_obs_.mu,
                num_pcs,
                self.fpca_model_params_.fpca_lambda,
                self.fpca_model_params_.fpca_phi["obs"],
                self.fpca_model_params_.measurement_error_variance,
                self.fpca_model_params_.if_shrinkage,
            )
        elif self.fpca_model_params_.method_pcs == "LS":
            NotImplementedError("Least squares method is not implemented.")
        elif self.fpca_model_params_.method_pcs == "WLS":
            NotImplementedError("Weighted least squares method is not implemented.")
        self.elapsed_time_["prediction"] = time.time() - start_time
        return new_xi_, new_xi_var_, new_fitted_y_mat_, new_fitted_y_

    def fit_score(
        self,
        method_pcs: Literal["IN", "CE"] = "CE",
        method_select_num_pcs: Union[int, Literal["FVE", "AIC", "BIC"]] = "FVE",
        method_rho: Literal["trunc", "ridge", "vanilla"] = "vanilla",
        max_num_pcs: Optional[int] = None,
        if_impute_scores: bool = True,
        if_shrinkage: bool = False,
        if_fit_eigen_values: bool = False,
        fve_threshold: float = 0.99,
    ) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray, List[np.ndarray]]:
        """Compute principal component scores given smoothed artifacts.

        This method assumes `fit` has been called to produce smoothed mean/covariance
        and eigenstructure. It then selects the number of components and computes
        scores using the chosen method.

        Parameters
        ----------
        method_pcs : {"IN", "CE"}, default="CE"
            Score computation method (In-sample or Conditional Expectation).
        method_select_num_pcs : int or {"FVE", "AIC", "BIC"}, default="FVE"
            Selection strategy for number of components or fixed integer.
        method_rho : {"trunc", "ridge", "vanilla"}, default="vanilla"
            Rho strategy for CE (ignored for IN).
        max_num_pcs : int, optional
            Upper bound for PCs; inferred if None.
        if_impute_scores : bool, default=True
            Whether to compute scores.
        if_shrinkage : bool, default=False
            Apply shrinkage for IN scores.
        if_fit_eigen_values : bool, default=False
            Fit eigenvalues by projection regression.
        fve_threshold : float, default=0.99
            FVE threshold used when method_select_num_pcs="FVE".

        Returns
        -------
        xi : np.ndarray of shape (n_samples, k)
            Principal component scores.
        xi_var : np.ndarray or List[np.ndarray]
            Score variance summaries (method-dependent).
        fitted_y_mat : np.ndarray of shape (nt_obs, n_samples)
            Fitted values on the observation grid.
        fitted_y : List[np.ndarray]
            Fitted values at observed time points per subject.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If called before `fit`.
        ValueError
            If selection/scoring arguments are invalid.

        Notes
        -----
        - CE scoring may estimate `rho` depending on `method_rho`.
        - When `if_fit_eigen_values=True`, `eigenvalue_fit` is stored in `fpca_model_params_`.
        """
        check_is_fitted(self, ["smoothed_model_result_obs_", "smoothed_model_result_reg_"])
        self.__check_fit_params(
            num_samples=self.num_samples_,
            method_pcs=method_pcs,
            method_select_num_pcs=method_select_num_pcs,
            method_rho=method_rho,
            max_num_pcs=max_num_pcs,
            if_impute_scores=if_impute_scores,
            if_shrinkage=if_shrinkage,
            if_fit_eigen_values=if_fit_eigen_values,
            fve_threshold=fve_threshold,
        )

        param_name_list = [
            "method_pcs",
            "method_select_num_pcs",
            "method_rho",
            "max_num_pcs",
            "if_shrinkage",
            "if_fit_eigen_values",
            "fve_threshold",
        ]
        for param_name in param_name_list:
            setattr(self.fpca_model_params_, param_name, locals()[param_name])

        start_time = time.time_ns()
        num_pcs = None
        if isinstance(method_select_num_pcs, str):
            if method_select_num_pcs == "FVE":
                self.fpca_model_params_.select_num_pcs_criterion, num_pcs = select_num_pcs_fve(
                    self.fpca_model_params_.eigen_results["eig_lambda"],
                    self.fpca_model_params_.fve_threshold,
                    self.fpca_model_params_.max_num_pcs,
                )
            elif method_select_num_pcs in ["AIC", "BIC"]:
                # implement AIC/BIC-based function to choose number of principal components
                # self.fpca_model_params_.select_num_pcs_criterion, num_pcs = select_num_pcs_ic(eig_lambda, method_select_num_pcs)
                raise NotImplementedError("AIC/BIC-based selection method is not implemented yet.")
        elif isinstance(method_select_num_pcs, int):
            num_pcs = method_select_num_pcs
        else:
            raise ValueError("Invalid method_select_num_pcs. Must be one of ['FVE', 'AIC', 'BIC'] or an integer.")
        self.elapsed_time_["num_pcs_selection"] = (time.time_ns() - start_time) / 1e9

        obs_grid = self.smoothed_model_result_obs_.grid
        reg_grid = self.smoothed_model_result_reg_.grid

        fpca_lambda, fpca_phi_reg = get_fpca_phi(
            num_pcs,
            reg_grid,
            self.smoothed_model_result_reg_.mu,
            self.fpca_model_params_.eigen_results["eig_lambda"],
            self.fpca_model_params_.eigen_results["eig_vector"],
        )
        fpca_phi_obs = np.zeros((len(obs_grid), num_pcs), dtype=self._input_dtype)
        for i in range(num_pcs):
            fpca_phi_obs[:, i] = interp1d(reg_grid, fpca_phi_reg[:, i], obs_grid, method="spline")
        self.fpca_model_params_.fpca_lambda = fpca_lambda
        self.fpca_model_params_.fpca_phi = {"obs": fpca_phi_obs, "reg": fpca_phi_reg}

        fitted_cov_reg = fpca_phi_reg @ np.diag(fpca_lambda) @ fpca_phi_reg.T
        fitted_cov_obs = interp2d(reg_grid, reg_grid, fitted_cov_reg, obs_grid, obs_grid, method="spline")
        self.fpca_model_params_.fitted_covariance = {"obs": fitted_cov_obs, "reg": fitted_cov_reg}

        # select rho and calculate the functional principal component score
        start_time = time.time_ns()
        self.elapsed_time_["rho_estimate"] = 0.0
        rho = None
        self.xi = None
        self.xi_var = None
        if if_impute_scores:
            if method_pcs == "CE":
                rho_start_time = time.time_ns()
                if method_rho != "vanilla":
                    if self.user_params.rho is None:
                        rho = estimate_rho(
                            method_rho,
                            self.flatten_func_data_,
                            reg_grid,
                            self.smoothed_model_result_obs_.mu,
                            self.smoothed_model_result_reg_.mu,
                            fpca_lambda,
                            fpca_phi_obs,
                            fpca_phi_reg,
                            fitted_cov_obs,
                            self.fpca_model_params_.measurement_error_variance,
                        )
                        self.fpca_model_params_.rho = rho
                    else:
                        rho = self.user_params.rho
                self.elapsed_time_["rho_estimate"] = (time.time_ns() - rho_start_time) / 1e9

                sigma2 = rho if rho is not None else self.fpca_model_params_.measurement_error_variance
                self.xi_, self.xi_var_, self.fitted_y_mat_, self.fitted_y_ = get_fpca_ce_score(
                    self.flatten_func_data_, self.smoothed_model_result_obs_.mu, num_pcs, fpca_lambda, fpca_phi_obs, fitted_cov_obs, sigma2
                )
            elif method_pcs == "IN":
                self.xi_, self.xi_var_, self.fitted_y_mat_, self.fitted_y_ = get_fpca_in_score(
                    self.flatten_func_data_,
                    self.smoothed_model_result_obs_.mu,
                    num_pcs,
                    fpca_lambda,
                    fpca_phi_obs,
                    self.fpca_model_params_.measurement_error_variance,
                    self.fpca_model_params_.if_shrinkage,
                )
            elif method_pcs == "LS":
                NotImplementedError("Least squares method is not implemented.")
            elif method_pcs == "WLS":
                NotImplementedError("Weighted least squares method is not implemented.")
        self.elapsed_time_["score_computation"] = (time.time_ns() - start_time) / 1e9 - self.elapsed_time_["rho_estimate"]

        if if_fit_eigen_values:
            start_time = time.time_ns()
            self.fpca_model_params_.eigenvalue_fit = get_eigenvalue_fit(self.raw_cov_, obs_grid, fpca_phi_obs, num_pcs)
            self.elapsed_time_["eigenvalue_fit"] = (time.time_ns() - start_time) / 1e9

        return self.xi_, self.xi_var_, self.fitted_y_mat_, self.fitted_y_
