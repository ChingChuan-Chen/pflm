"""Functional Principal Component Analysis (FPCA) implementation."""

# Authors: Ching-Chuan Chen
# SPDX-License-Identifier: MIT

import warnings
from typing import List, Literal, Optional, Union

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils._array_api import get_namespace_and_device, supported_float_dtypes
from sklearn.utils.validation import check_array, check_is_fitted

from pflm.interp import interp1d, interp2d
from pflm.smooth import KernelType, Polyfit1DModel, Polyfit2DModel
from pflm.utils import (
    FlattenFunctionalData,
    FpcaModelResult,
    SmoothedModelResult,
    estimate_rho,
    flatten_and_sort_data_matrices,
    get_covariance_matrix,
    get_eigen_analysis_results,
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
        Whether to assume measurement error in the dataset.
    num_n_points_reg_grid : int, default=51
        Number of points in the regular grid for regression.
    user_params : FunctionalPCAUserDefinedParams, default=FunctionalPCAUserDefinedParams()
        User-defined parameters for mean and covariance functions.
    verbose : bool, default=False
        Whether to print diagnostic messages during fitting.

    Attributes
    ----------
    y_ : List[np.ndarray]
        List of functional data observations.
    t_ : List[np.ndarray]
        List of time points corresponding to the observations in `y_`.
    w_ : List[np.ndarray]
        List of weights for each observation. If None, equal weights are assumed.
    flatten_func_data_ : FlattenFunctionalData
        Flattened and sorted functional data. It includes `yy`, `ww`, `tt`, `tid`, `unique_tid`, `inverse_tid_idx`, `sid`, `unique_sid`, and `sid_cnt`.
    smoothed_model_result_obs_ : SmoothedModelResult
        The smoothed model result for the observation grid points.
    smoothed_model_result_reg_ : SmoothedModelResult
        The smoothed model result for the regular grid points.
    fpca_model_result_ : FpcaModelResult
        The functional PCA model result.
    fitted_y : List[np.ndarray]
        Fitted values of the functional data based on the estimated mean and principal component scores.
    measurement_error_variance_ : float
        Estimated measurement error variance at the regular grid points.

    See Also
    --------
    FPCASmoothingParams : Class for smoothing parameters in FPCA.
    FunctionalPCAUserDefinedParams : Class for user-defined parameters in FPCA.
    """

    def __init__(
        self,
        assume_measurement_error: bool = True,
        num_points_reg_grid: int = 51,
        mu_cov_params: FunctionalPCAMuCovParams = FunctionalPCAMuCovParams(),
        user_params: FunctionalPCAUserDefinedParams = FunctionalPCAUserDefinedParams(),
        verbose: bool = False,
    ) -> None:
        if not isinstance(assume_measurement_error, bool):
            raise ValueError("assume_measurement_error must be a boolean value.")
        if not isinstance(num_points_reg_grid, int) or num_points_reg_grid <= 0:
            raise ValueError("num_n_points_reg_grid must be a positive integer.")
        if not isinstance(mu_cov_params, FunctionalPCAMuCovParams):
            raise ValueError("mu_cov_params must be an instance of FunctionalPCAMuCovParams.")
        if not isinstance(user_params, FunctionalPCAUserDefinedParams):
            raise ValueError("user_params must be an instance of FunctionalPCAUserDefinedParams.")
        if not isinstance(verbose, bool):
            raise ValueError("verbose must be a boolean value.")

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
        impute_scores: bool = True,
        fit_eigen_values: bool = False,
        fve_threshold: float = 0.99,
    ):
        if method_pcs not in ["IN", "CE"]:
            raise ValueError("method_pcs must be either 'IN' (Numerical Integration) or 'CE' (Conditional Expectation).")
        if not isinstance(method_select_num_pcs, (int, str)):
            raise ValueError("method_select_num_pcs must be either a positive integer or one of 'FVE', 'AIC', 'BIC'.")
        if isinstance(method_select_num_pcs, int) and method_select_num_pcs <= 0:
            raise ValueError("If method_select_num_pcs is an integer, it must be a positive integer.")
        if isinstance(method_select_num_pcs, str) and method_select_num_pcs not in ["FVE", "AIC", "BIC"]:
            raise ValueError("method_select_num_pcs must be one of 'FVE', 'AIC', 'BIC' or a positive integer.")
        if method_rho not in ["trunc", "ridge", "vanilla"]:
            raise ValueError("method_rho must be one of 'trunc', 'ridge', 'vanilla'.")
        if max_num_pcs is not None and (not isinstance(max_num_pcs, int) or max_num_pcs <= 0):
            raise ValueError("max_num_pcs must be a positive integer.")
        if max_num_pcs is None:
            self.max_num_pcs = min(num_samples - 2, self.num_points_reg_grid - 2)
        if not isinstance(fve_threshold, float) or fve_threshold <= 0 or fve_threshold > 1:
            raise ValueError("fve_threshold must be a float between 0 and 1.")
        if not isinstance(impute_scores, bool):
            raise ValueError("impute_scores must be a boolean value.")
        if not isinstance(fit_eigen_values, bool):
            raise ValueError("fit_eigen_values must be a boolean value.")

        self.method_pcs_ = method_pcs
        self.method_select_num_pcs_ = method_select_num_pcs
        self.fve_threshold_ = fve_threshold
        self.max_num_pcs_ = max_num_pcs
        self.impute_scores_ = impute_scores
        self.fit_eigen_values_ = fit_eigen_values

    def fit(
        self,
        t: List[Union[np.ndarray, List[float]]],
        y: List[Union[np.ndarray, List[float]]],
        w: Optional[List[Union[np.ndarray, List[float]]]] = None,
        method_pcs: Literal["IN", "CE"] = "CE",
        method_select_num_pcs: Union[int, Literal["FVE", "AIC", "BIC"]] = "FVE",
        method_rho: Literal["trunc", "ridge", "vanilla"] = "vanilla",
        max_num_pcs: Optional[int] = None,
        impute_scores: bool = True,
        fit_eigen_values: bool = False,
        fve_threshold: float = 0.99,
        reg_grid: Union[np.ndarray, List[float]] = None,
    ) -> "FunctionalPCA":
        """
        Fit the functional PCA model to the data.

        Parameters
        ----------
        t : a list of array-like
            A 1D array of time points corresponding to the observations in `y`.
        y : a list of array-like
            A 2D array where each row corresponds to an observation at the time points in `t` which allow for missing values.
        w : a list of array-like, optional
            A 1D array of weights for each observation. If None, equal weights are assumed.
        method_pcs : {'IN', 'CE'}, default='CE'
            Method to compute principal component scores. 'IN' for Numerical Integration, 'CE' for Conditional Expectation.
        method_select_num_pcs : int or {"FVE", "AIC", "BIC"}, optional
            Method to select the number of principal component scores. If empty, it will be determined based on the explained variance.
            If an integer is provided, it specifies the number of principal component scores to use.
        method_rho : {'trunc', 'ridge', 'vanilla'}, default='vanilla'
            Method to estimate the regularization factor which is added to diagonal of covariance surface in estimating principal component
            scores. 'trunc' is using truncation of sigma2, 'ridge' is using rho as a ridge parameter, 'vanilla' is vanilla approach.
        max_num_pcs : int, optional
            Maximum number of principal component scores to consider. If None, it will be set to the minimum of
            (number of samples - 2, number of points in reg_grid - 2).
        impute_scores : bool, default=True
            Whether to impute missing scores in the functional data.
        fit_eigen_values: bool, default=False
            Whether to obtain a regression fit of the eigenvalues.
        fve_threshold : float, default=0.99
            Threshold for the explained variance when using 'FVE' method to select the number of principal component scores.
        reg_grid : array-like, optional
            Regular grid points for regression. If None, a default grid will be created.

        Returns
        -------
        FunctionalPCA
            The fitted model instance which will contain the estimated parameters such as mean function,
            covariance function, and principal component scores.
        """
        self.num_samples_ = len(y)
        self.__check_fit_params(
            self.num_samples_,
            method_pcs=method_pcs,
            method_select_num_pcs=method_select_num_pcs,
            method_rho=method_rho,
            max_num_pcs=max_num_pcs,
            impute_scores=impute_scores,
            fit_eigen_values=fit_eigen_values,
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

        # calculate the mean function
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

        # calculate the covariance function
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
            reg_obs = interp2d(t_cov, t_cov, cov, reg_grid, reg_grid, method="spline")
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
        self.measurement_error_variance_ = sigma2

        # Create / update smoothed result containers (phi / fitted_cov filled later in fit_score)
        self.smoothed_model_result_obs_ = SmoothedModelResult(
            grid=obs_grid,
            mu=mu_obs,
            cov=cov_obs,
            phi=None,
            fitted_cov=None,
            grid_type="obs",
        )
        self.smoothed_model_result_reg_ = SmoothedModelResult(
            grid=reg_grid,
            mu=mu_reg,
            cov=cov_reg,
            phi=None,
            fitted_cov=None,
            grid_type="reg",
        )

        self.fpca_model_result_ = self.fit_score(
            self.method_pcs_, self.method_select_num_pcs_, self.max_num_pcs_, self.impute_scores_, self.fit_eigen_values_, self.fve_threshold_
        )
        return self

    def __repr__(self):
        return (
            f"FunctionalPCA(assume_measurement_error={self.assume_measurement_error}, "
            f"num_points_reg_grid={self.num_points_reg_grid}, mu_cov_params={self.mu_cov_params}, "
            f"user_params={self.user_params}, verbose={self.verbose})"
        )

    def fitted_values(self) -> List[np.ndarray]:
        check_is_fitted(self, ["fpca_model_result_", "fitted_y_"])
        return self.fitted_y_

    def predict(self, t: List[Union[np.ndarray, List[float]]]) -> List[np.ndarray]:
        check_is_fitted(self, ["fpca_model_result_", "fitted_y_"])
        return np.array([])

    def fit_score(
        self,
        method_pcs: Literal["IN", "CE"] = "CE",
        method_select_num_pcs: Union[int, Literal["FVE", "AIC", "BIC"]] = "FVE",
        method_rho: Literal["trunc", "ridge", "vanilla"] = "vanilla",
        max_num_pcs: Optional[int] = None,
        impute_scores: bool = True,
        fit_eigen_values: bool = False,
        fve_threshold: float = 0.99,
    ) -> FpcaModelResult:
        """
        Fit the principal component scores for the functional data.

        The method computes the principal component scores based on the covariance function
        and selects the number of components based on the specified method.
        This method needs to be called after fitting the model with `fit()` which will ensure FunctionalPCA is ready for score computation.
        This method provides you an alternative way to compute the principal component scores in case you don't want to re-calculate
        the mean, covariance, and other parameters again.

        Parameters
        ----------
        method_pcs : {'IN', 'CE'}, default='CE'
            Method to compute principal components. 'IN' for Numerical Integration, 'CE' for Conditional Expectation.
        method_select_num_pcs : int or {"FVE", "AIC", "BIC"}, optional
            Method to select the number of principal components. If empty, it will be determined based on the explained variance.
            If an integer is provided, it specifies the number of principal components to use.
        method_rho : {'truncated', 'ridge', 'vanilla'}, default='vanilla'
            Method to estimate the regularization factor which is added to diagonal of covariance surface in estimating principal component
            scores. 'truncated' is using truncation of sigma2, 'ridge' is using rho as a ridge parameter, 'vanilla' is using vanilla approach.
        max_num_pcs : int, optional
            Maximum number of principal components to consider. If None, it will be set to the minimum of
            (number of samples - 2, number of points in reg_grid - 2).
        impute_scores : bool, default=True
            Whether to impute missing scores in the functional data.
        fit_eigen_values : bool, default=False
            Whether to obtain a regression fit of the eigenvalues.
        fve_threshold : float, default=0.99
            Threshold for the explained variance when using 'FVE' method to select the number of principal components.

        Returns
        -------
        fpca_model_result_ : FpcaModelResult
            A named tuple containing the results of the functional PCA model fitting.
            - xi: Principal component scores for the functional data.
            - xi_var: Variance of the principal component scores.
            - num_pcs: Number of principal components used.
            - fpca_lambda: Eigenvalues of the functional PCA model.
            - eigen_results: A dict containing the eigenvalue decomposition results.
            - rho: Regularization parameter used in the model.

        fitted_y : np.ndarray
            Fitted values for the functional data.
        """
        check_is_fitted(self, ["smoothed_model_result_obs_", "smoothed_model_result_reg_"])
        self.__check_fit_params(
            num_samples=self.num_samples_,
            method_pcs=method_pcs,
            method_select_num_pcs=method_select_num_pcs,
            method_rho=method_rho,
            max_num_pcs=max_num_pcs,
            impute_scores=impute_scores,
            fit_eigen_values=fit_eigen_values,
            fve_threshold=fve_threshold,
        )

        eig_lambda, eig_vector = get_eigen_analysis_results(self.smoothed_model_result_reg_.cov)
        if isinstance(method_select_num_pcs, str):
            if method_select_num_pcs == "FVE":
                cumulative_fve, num_pcs = select_num_pcs_fve(eig_lambda, self.fve_threshold_, self.max_num_pcs_)
            elif method_select_num_pcs in ["AIC", "BIC"]:
                # implement AIC/BIC-based function to choose number of principal components
                raise NotImplementedError("AIC/BIC-based selection method is not implemented yet.")
        elif isinstance(method_select_num_pcs, int):
            num_pcs = method_select_num_pcs
        else:
            raise ValueError("Invalid method_select_num_pcs. Must be one of ['FVE', 'AIC', 'BIC'] or an integer.")

        obs_grid = self.smoothed_model_result_obs_.grid
        reg_grid = self.smoothed_model_result_reg_.grid

        fpca_lambda, fpca_phi_reg = get_fpca_phi(num_pcs, reg_grid, self.smoothed_model_result_reg_.mu, eig_lambda, eig_vector)
        fitted_cov_reg = fpca_phi_reg @ np.diag(fpca_lambda) @ fpca_phi_reg.T
        fpca_phi_obs = np.zeros((len(obs_grid), num_pcs), dtype=self._input_dtype)
        for i in range(num_pcs):
            fpca_phi_obs[:, i] = interp1d(reg_grid, fpca_phi_reg[:, i], obs_grid, method="spline")
        fitted_cov_obs = interp2d(reg_grid, reg_grid, fitted_cov_reg, obs_grid, obs_grid, method="spline")

        # fill phi/fitted_cov
        self.smoothed_model_result_obs_.phi = fpca_phi_obs
        self.smoothed_model_result_obs_.fitted_cov = fitted_cov_obs
        self.smoothed_model_result_reg_.phi = fpca_phi_reg
        self.smoothed_model_result_reg_.fitted_cov = fitted_cov_reg

        # select rho and calculate the functional principal component score
        self.measurement_error_variance_origin_ = self.measurement_error_variance_
        if impute_scores:
            if method_pcs == "CE":
                if method_rho != "vanilla":
                    if self.user_params.rho is not None:
                        rho = self.user_params.rho
                    else:
                        rho = estimate_rho(
                            method_rho,
                            self.flatten_func_data_,
                            fitted_cov_obs,
                            fpca_lambda,
                            fpca_phi_obs,
                            self.measurement_error_variance_,
                        )
                    self.measurement_error_variance_ = rho

                xi, xi_var, self.fitted_y_ = get_fpca_ce_score(
                    self.flatten_func_data_,
                    fitted_cov_obs,
                    fpca_lambda,
                    fpca_phi_obs,
                    self.measurement_error_variance_,
                )
            elif method_pcs == "IN":
                xi, xi_var, self.fitted_y_ = get_fpca_in_score(
                    self.flatten_func_data_,
                    fitted_cov_obs,
                    fpca_lambda,
                    fpca_phi_obs,
                    self.measurement_error_variance_,
                )

        if fit_eigen_values:
            self.fit_eigen_values_ = np.zeros(num_pcs, dtype=self._input_dtype)

        # Create FPCA result object
        self.fpca_model_result_ = FpcaModelResult(
            xi=xi,
            xi_var=xi_var,
            num_pcs=num_pcs,
            fpca_lambda=fpca_lambda,
            eigen_results={"eigenvalues": eig_lambda, "eigenvectors": eig_vector},
            rho=rho
        )

        return self.fpca_model_result_, self.fitted_y_
