"""Functional Principal Component Analysis (FPCA) implementation."""

# Authors: Ching-Chuan Chen
# SPDX-License-Identifier: MIT

import warnings
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils._array_api import get_namespace_and_device, supported_float_dtypes
from sklearn.utils.validation import check_array, check_is_fitted

from pflm.interp import interp1d, interp2d
from pflm.smooth import KernelType, Polyfit1DModel, Polyfit2DModel
from pflm.utils.utility import flatten_and_sort_data_matrices, get_covariance_matrix, get_raw_cov


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
        if method_select_mu_bw not in ["GCV", "CV"]:
            raise ValueError("method_select_mu_bw must be either 'GCV' or 'CV'.")
        if method_select_cov_bw not in ["GCV", "CV"]:
            raise ValueError("method_select_cov_bw must be either 'GCV' or 'CV'.")
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
        Correlation parameter for the covariance function.
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
    sid_ : np.ndarray
        Sorted indices of the observations.
    tt_ : np.ndarray
        Sorted time points corresponding to the observations in `y_`.
    yy_ : np.ndarray
        Sorted functional data observations.
    ww_ : np.ndarray
        Weights corresponding to the sorted observations.
    tid_ : np.ndarray
        Indices of the regular grid points corresponding to the sorted time points.
    reg_grid_: np.ndarray
        Regular grid points used for regression.
    obs_grid_: np.ndarray
        Unique observation grid points derived from `tt_`.
    mu_ : np.ndarray
        Estimated mean function values at the observation grid points (obs_grid_).
    mu_dense_ : np.ndarray
        Estimated mean function values at the regular grid points (reg_grid_).
    cov_ : np.ndarray
        Estimated covariance function values at the observation grid points (obs_grid_).
    cov_dense_ : np.ndarray
        Estimated covariance function values at the regular grid points (reg_grid_).
    fpca_eigen_results_ : dict
        Dictionary containing eigenvalues and eigenvectors of the covariance function.
    num_pcs_ : int
        Number of principal components used in the model.
    xi : np.ndarray
        Principal component scores for the functional data.
    xi_var : np.ndarray
        Variance of the principal component scores.
    fitted_y : List[np.ndarray]
        Fitted values of the functional data based on the estimated mean and principal components.
    rho_ : float
        Estimated correlation function value at the regular grid points.
    sigma2_ : float
        Estimated variance function value at the regular grid points.

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
        max_num_pcs: Optional[int] = None,
        impute_scores: bool = True,
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
        if max_num_pcs is not None and (not isinstance(max_num_pcs, int) or max_num_pcs <= 0):
            raise ValueError("max_num_pcs must be a positive integer.")
        if max_num_pcs is None:
            self.max_num_pcs = min(num_samples - 2, self.num_points_reg_grid - 2)
        if not isinstance(fve_threshold, float) or fve_threshold <= 0 or fve_threshold > 1:
            raise ValueError("fve_threshold must be a float between 0 and 1.")
        if not isinstance(impute_scores, bool):
            raise ValueError("impute_scores must be a boolean value.")

        self.method_pcs_ = method_pcs
        self.method_select_num_pcs_ = method_select_num_pcs
        self.fve_threshold_ = fve_threshold
        self.max_num_pcs_ = max_num_pcs
        self.impute_scores_ = impute_scores

    def fit(
        self,
        t: List[Union[np.ndarray, List[float]]],
        y: List[Union[np.ndarray, List[float]]],
        *,
        w: Optional[List[Union[np.ndarray, List[float]]]] = None,
        method_pcs: Literal["IN", "CE"] = "CE",
        method_select_num_pcs: Union[int, Literal["FVE", "AIC", "BIC"]] = "FVE",
        max_num_pcs: Optional[int] = None,
        impute_scores: bool = True,
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
            Method to compute principal components. 'IN' for Numerical Integration, 'CE' for Conditional Expectation.
        method_select_num_pcs : int or {"FVE", "AIC", "BIC"}, optional
            Method to select the number of principal components. If empty, it will be determined based on the explained variance.
            If an integer is provided, it specifies the number of principal components to use.
        max_num_pcs : int, optional
            Maximum number of principal components to consider. If None, it will be set to the minimum of
            (number of samples - 2, number of points in reg_grid - 2).
        impute_scores : bool, default=True
            Whether to impute missing scores in the functional data.
        fve_threshold : float, default=0.99
            Threshold for the explained variance when using 'FVE' method to select the number of principal components.
        reg_grid : array-like, optional
            Regular grid points for regression. If None, a default grid will be created.

        Returns
        -------
        FunctionalPCA
            The fitted model instance which will contain the estimated parameters such as mean function,
            covariance function, and principal components scores.
        """
        self.num_samples_ = len(y)
        self.__check_fit_params(
            self.num_samples_,
            method_pcs=method_pcs,
            method_select_num_pcs=method_select_num_pcs,
            max_num_pcs=max_num_pcs,
            impute_scores=impute_scores,
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
        self.tt_, self.yy_, self.ww_, self.sid_ = flatten_and_sort_data_matrices(self.y_, self.t_, self._input_dtype, w)

        # get observation grid and time indices
        self.obs_grid_, self.obs_grid_idx_ = np.unique(self.tt_, return_inverse=True, sorted=True)
        self.tid_ = np.digitize(self.tt_, self.obs_grid_, right=True)

        # create reg_grid_
        if reg_grid is not None:
            # Use custom grid
            self.reg_grid_ = check_array(reg_grid, ensure_2d=False, dtype=self._input_dtype)
            if self.reg_grid_.ndim != 1:
                raise ValueError("reg_grid must be a 1D array")
            if len(self.reg_grid_) < 2:
                raise ValueError("reg_grid must have at least 2 points")

            # Check if grid is within the range of input X
            if np.min(self.reg_grid_) < self.tt_[0] or np.max(self.reg_grid_) > self.tt_[-1]:
                raise ValueError(
                    f"reg_grid must be within the range of input X [{self.tt_[0]:.6f}, {self.tt_[-1]:.6f}]. "
                    f"Got reg_grid range [{np.min(self.reg_grid_):.6f}, {np.max(self.reg_grid_):.6f}]"
                )
        else:
            # Create uniform grid within the range of input X
            self.reg_grid_ = np.linspace(self.tt_[0], self.tt_[-1], self.num_points_reg_grid, dtype=self._input_dtype)

        # calculate the mean function
        if self.user_params.t_mu is not None and self.user_params.mu is not None:
            t_mu = check_array(self.user_params.t_mu, ensure_2d=False, dtype=self._input_dtype)
            mu = check_array(self.user_params.mu, ensure_2d=False, dtype=self._input_dtype)
            if t_mu.ndim != 1 or mu.ndim != 1:
                raise ValueError("t_mu and mu must be 1D arrays.")
            if t_mu.size != mu.size:
                raise ValueError("t_mu and mu must have the same length.")
            self.mu_ = interp1d(t_mu, mu, self.reg_grid_, method="spline")
        elif self.mu_cov_params.estimate_method == "smooth":
            self.mean_func_fit_ = Polyfit1DModel(
                kernel_type=self.mu_cov_params.kernel_type, interp_kind="spline", random_seed=self.mu_cov_params.random_seed
            )
            self.mean_func_fit_.fit(
                self.tt_,
                self.yy_,
                self.ww_,
                bandwidth=self.mu_cov_params.bw_mu,
                reg_grid=self.reg_grid_,
                bandwidth_selection_method=self.mu_cov_params.method_select_mu_bw,
                cv_folds=self.mu_cov_params.cv_folds_mu,
            )
            self.mu_ = self.mean_func_fit_.fitted_values()
        elif self.mu_cov_params.estimate_method == "cross-sectional":
            self.mu_ = np.bincount(self.tid_, self.yy_) / np.bincount(self.tid_)

        # calculate the covariance function
        if self.user_params.t_cov is not None and self.user_params.cov is not None:
            t_cov = check_array(self.user_params.t_cov, ensure_2d=False, dtype=self._input_dtype)
            cov = check_array(self.user_params.cov, ensure_2d=False, dtype=self._input_dtype)
            if t_cov.ndim != 1 or cov.ndim != 2:
                raise ValueError("t_cov must be a 1D array and cov must be a 2D array.")
            if t_cov.size != cov.shape[0] or t_cov.size != cov.shape[1]:
                raise ValueError("t_cov must match the dimensions of cov.")
            self.cov_ = interp2d(t_cov, t_cov, cov, self.obs_grid_, self.obs_grid_, method="spline")
            self.smooth_cov_ = interp2d(t_cov, t_cov, cov, self.reg_grid_, self.reg_grid_, method="spline")
        elif self.mu_cov_params.estimate_method == "smooth":
            self.raw_cov_ = get_raw_cov(self.yy_, self.tt_, self.ww_, self.mu_, self.sid_, self.tid_)
            self.cov_func_fit_ = Polyfit2DModel(
                kernel_type=self.mu_cov_params.kernel_type, interp_kind="spline", random_seed=self.mu_cov_params.random_seed
            )
            self.cov_func_fit_.fit(
                self.raw_cov_[:, [1, 2]],
                self.raw_cov_[:, 4],
                sample_weight=self.raw_cov_[:, 3],
                bandwidth1=self.mu_cov_params.bw_cov,
                bandwidth2=self.mu_cov_params.bw_cov,
                reg_grid1=self.obs_grid_,
                reg_grid2=self.obs_grid_,
                bandwidth_selection_method=self.mu_cov_params.method_select_cov_bw,
                cv_folds=self.mu_cov_params.cv_folds_cov,
            )
            self.cov_ = self.cov_func_fit_.fitted_values()
            self.smooth_cov_ = interp2d(self.obs_grid_, self.obs_grid_, self.cov_, self.reg_grid_, self.reg_grid_, method="spline")
        elif self.mu_cov_params.estimate_method == "cross-sectional":
            self.raw_cov_ = get_raw_cov(self.yy_, self.tt_, self.ww_, self.mu_, self.sid_, self.tid_)
            self.cov_ = get_covariance_matrix(self.raw_cov_, self.obs_grid_)
            self.smooth_cov_ = interp2d(self.obs_grid_, self.obs_grid_, self.cov_, self.reg_grid_, self.reg_grid_, method="spline")

        self.xi, self.xi_var = self.fit_score(
            self.method_pcs_, self.method_select_num_pcs_, self.max_num_pcs_, self.impute_scores_, self.fve_threshold_
        )
        return self

    def fitted_values(self) -> List[np.ndarray]:
        check_is_fitted(self, ["fpca_eigen_results_", "xi", "xi_var"])
        return np.array([])

    def predict(self, t: List[Union[np.ndarray, List[float]]]) -> List[np.ndarray]:
        check_is_fitted(self, ["fpca_eigen_results_", "xi", "xi_var"])
        return np.array([])

    def fit_score(
        self,
        method_pcs: Literal["IN", "CE"] = "CE",
        method_select_num_pcs: Union[int, Literal["FVE", "AIC", "BIC"]] = "FVE",
        max_num_pcs: Optional[int] = None,
        impute_scores: bool = True,
        fve_threshold: float = 0.99,
    ) -> Tuple[np.ndarray, np.ndarray]:
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
        max_num_pcs : int, optional
            Maximum number of principal components to consider. If None, it will be set to the minimum of
            (number of samples - 2, number of points in reg_grid - 2).
        impute_scores : bool, default=True
            Whether to impute missing scores in the functional data.
        fve_threshold : float, default=0.99
            Threshold for the explained variance when using 'FVE' method to select the number of principal components.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            A tuple containing:
            - xi: Principal component scores for the functional data.
            - xi_var: Variance of the principal component scores.
        """
        check_is_fitted(self, ["fpca_eigen_results_"])
        self.__check_fit_params(
            num_samples=self.num_samples_,
            method_pcs=method_pcs,
            method_select_num_pcs=method_select_num_pcs,
            max_num_pcs=max_num_pcs,
            impute_scores=impute_scores,
            fve_threshold=fve_threshold,
        )
        return np.array([]), np.array([])
