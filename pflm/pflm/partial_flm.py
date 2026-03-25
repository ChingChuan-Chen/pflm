"""Partial Functional Linear Model (PFLM) implementation."""

# Authors: Ching-Chuan Chen
# SPDX-License-Identifier: MIT

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import numpy as np
from sklearn.base import BaseEstimator, MultiOutputMixin, RegressorMixin
from sklearn.utils._array_api import get_namespace_and_device, supported_float_dtypes
from sklearn.utils.validation import check_array, check_is_fitted

from pflm.fpca import FunctionalPCA, FunctionalPCAMuCovParams, FunctionalPCAUserDefinedParams
from pflm.pflm.utils import ElasticNet, LinearModelFamily


@dataclass
class FPCAConfig:
    """Configuration for one functional feature's FPCA.

    Parameters
    ----------
    assume_measurement_error : bool, default=True
        Whether to assume measurement error in the functional data.
        Forwarded to ``FunctionalPCA.__init__()``.
    num_points_reg_grid : int, default=51
        Number of points in the regular grid for FPCA.
        Forwarded to ``FunctionalPCA.__init__()``.
    mu_cov_params : FunctionalPCAMuCovParams, optional
        Parameters for mean and covariance estimation in FPCA.  Forwarded to ``FunctionalPCA.__init__()``.  If
        ``None``, the default ``FunctionalPCAMuCovParams()`` is used.
    user_params : FunctionalPCAUserDefinedParams, optional
        User-defined parameters for FPCA (e.g., user-defined mean or eigenfunctions).  Forwarded to
        ``FunctionalPCA.__init__()``.  If ``None``, the default ``FunctionalPCAUserDefinedParams()`` is used.
    verbose : bool, default=False
        Whether to print verbose output during FPCA fitting.
        Forwarded to ``FunctionalPCA.__init__()``.
    fit_params : dict
        Keyword arguments forwarded to ``FunctionalPCA.fit()``.  Supported keys: ``method_pcs``,
        ``method_select_num_pcs``, ``method_rho``, ``max_num_pcs``, ``if_impute_scores``, ``if_shrinkage``,
        ``if_fit_eigen_values``, ``fve_threshold``, ``reg_grid``.

    Examples
    --------
    Default configuration (all FPCA defaults):

    >>> cfg = FPCAConfig()

    Custom regular grid and FVE threshold:

    >>> cfg = FPCAConfig(
    ...     num_points_reg_grid=101,
    ...     fit_params=dict(fve_threshold=0.95, method_pcs="CE"),
    ... )

    Specify bandwidth parameters via ``mu_cov_params``:

    >>> from pflm.fpca import FunctionalPCAMuCovParams
    >>> cfg = FPCAConfig(
    ...     mu_cov_params=FunctionalPCAMuCovParams(bw_mu=0.5, bw_cov=0.5),
    ... )

    See Also
    --------
    FunctionalPCA
    FunctionalPCAMuCovParams
    FunctionalPCAUserDefinedParams
    """

    assume_measurement_error: bool = True
    num_points_reg_grid: int = 51
    mu_cov_params: Optional[FunctionalPCAMuCovParams] = None
    user_params: Optional[FunctionalPCAUserDefinedParams] = None
    verbose: bool = False
    fit_params: Dict = field(default_factory=dict)


class PartialFunctionalLinearModel(MultiOutputMixin, RegressorMixin, BaseEstimator):
    """Partial Functional Linear Model with elastic-net regularization.

    Combines scalar predictors with one or more functional predictors (via FPCA) and fits an ``ElasticNet`` on the
    concatenated design matrix ``[scalar_features | FPCA_scores_0 | ... | FPCA_scores_k]``.

    Parameters
    ----------
    family : LinearModelFamily, default=LinearModelFamily.GAUSSIAN
        Distribution family for the response variable.  Forwarded to ``ElasticNet`` via ``linear_opts``.
    linear_opts : dict, optional
        Keyword arguments forwarded to ``ElasticNet.__init__()``.
        Supported keys: ``alpha``, ``l1_ratio``, ``fit_intercept``,
        ``power``, ``max_iter``, ``rho``, ``abs_tol``, ``rel_tol``,
        ``min_iter``.
    fpca_configs : FPCAConfig or list of FPCAConfig, optional
        FPCA configuration.  If a single ``FPCAConfig``, it is shared across all functional features.  If a list,
        its length must equal the number of functional features passed to ``fit()``.  Each entry separately controls
        ``FunctionalPCA.__init__()`` and ``FunctionalPCA.fit()`` parameters.

    Attributes
    ----------
    linear_model_ : ElasticNet
        The fitted linear model.
    fpca_models_ : list of FunctionalPCA
        One fitted FPCA model per functional feature.
    n_functional_features_in_ : int
        Number of functional features seen during ``fit()``.
    n_scalar_features_in_ : int
        Number of scalar features seen during ``fit()``.
    fitted_values_ : np.ndarray of shape (n_samples,) or (n_samples, n_classes)
        Predicted values on the training data.

    Examples
    --------
    Fit a model with one functional feature and two scalar features:

    >>> import numpy as np
    >>> from pflm.fpca import FunctionalDataGenerator
    >>> from pflm.pflm.partial_flm import PartialFunctionalLinearModel
    >>> # Generate synthetic functional data
    >>> t = np.linspace(0.0, 10.0, 51)
    >>> gen = FunctionalDataGenerator(
    ...     t, lambda x: np.sin(x) * 0.5, lambda x: 1.0 + 0.2 * np.cos(x),
    ... )
    >>> y_list, t_list = gen.generate(n=50, seed=42)
    >>> scalar = np.random.default_rng(0).standard_normal((50, 2))
    >>> response = np.random.default_rng(1).standard_normal(50)
    >>> # Fit with default settings
    >>> model = PartialFunctionalLinearModel()
    >>> model.fit([t_list], [y_list], scalar, response)  # doctest: +ELLIPSIS
    PartialFunctionalLinearModel(...)
    >>> model.linear_model_.coef_.shape[0] == 2 + model.fpca_models_[0].num_pcs_
    True

    Customise ElasticNet and per-feature FPCA settings:

    >>> model = PartialFunctionalLinearModel(
    ...     linear_opts=dict(alpha=0.1, l1_ratio=0.8),
    ...     fpca_configs=FPCAConfig(
    ...         num_points_reg_grid=101,
    ...         fit_params=dict(fve_threshold=0.95),
    ...     ),
    ... )

    Use per-feature FPCA configs for two functional features:

    >>> model = PartialFunctionalLinearModel(
    ...     fpca_configs=[
    ...         FPCAConfig(fit_params=dict(method_pcs="IN")),
    ...         FPCAConfig(fit_params=dict(method_pcs="CE", fve_threshold=0.99)),
    ...     ],
    ... )

    See Also
    --------
    ElasticNet
    FunctionalPCA
    FPCAConfig
    """

    def __init__(
        self,
        family: LinearModelFamily = LinearModelFamily.GAUSSIAN,
        linear_opts: Optional[Dict] = None,
        fpca_configs: Optional[Union[FPCAConfig, List[FPCAConfig]]] = None,
    ):
        self.linear_opts = linear_opts or {}
        self.linear_opts["family"] = family
        self.fpca_configs = fpca_configs

    def _get_fpca_config(self, i: int) -> FPCAConfig:
        """Return the ``FPCAConfig`` for the *i*-th functional feature."""
        if self.fpca_configs is None:
            return FPCAConfig()
        if isinstance(self.fpca_configs, FPCAConfig):
            return self.fpca_configs
        return self.fpca_configs[i]

    def fit(
        self,
        functional_time: List[List[np.ndarray]],
        functional_features: List[List[np.ndarray]],
        scalar_features: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ):
        """Fit the partial functional linear model.

        Parameters
        ----------
        functional_time : list of list-of-ndarray
            ``functional_time[j]`` is the list of per-subject time vectors for the *j*-th functional feature.
        functional_features : list of list-of-ndarray
            ``functional_features[j]`` is the list of per-subject observation vectors for the *j*-th functional
            feature.
        scalar_features : ndarray of shape (n_samples, n_scalar_features)
            Scalar predictors.
        y : ndarray of shape (n_samples,)
            Response vector.  Will be cast to the functional features'
            dtype if necessary.
        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample.  Weights are normalized so that ``sum(w) == n``.

        Returns
        -------
        self : PartialFunctionalLinearModel
            Fitted estimator.

        Raises
        ------
        ValueError
            If the number of functional time lists does not match the number of functional feature lists, or if
            ``fpca_configs`` is a list whose length does not match.

        Notes
        -----
        The dtype resolution uses `sklearn.utils._array_api.get_namespace_and_device` and `supported_float_dtypes`
        to select a compatible floating dtype.
        """
        self.n_functional_features_in_ = len(functional_features)
        self.n_scalar_features_in_ = scalar_features.shape[1]
        if len(functional_time) != self.n_functional_features_in_:
            raise ValueError("Number of functional time lists does not match number of functional feature lists")
        if isinstance(self.fpca_configs, list) and len(self.fpca_configs) != self.n_functional_features_in_:
            raise ValueError(
                f"fpca_configs has {len(self.fpca_configs)} entries but {self.n_functional_features_in_} functional features were provided"
            )

        # -- Validate and store input data (scalar features, response, sample weight) --
        xp, *_ = get_namespace_and_device(scalar_features, y, sample_weight)
        scalar_features = check_array(scalar_features, ensure_2d=True, dtype=supported_float_dtypes(xp))
        y = check_array(y, ensure_2d=False, dtype=scalar_features.dtype)
        sample_weight = (
            check_array(sample_weight, ensure_2d=False, dtype=scalar_features.dtype)
            if sample_weight is not None
            else np.ones(y.shape[0], dtype=scalar_features.dtype)
        )
        self._input_dtype = scalar_features.dtype

        # -- Store fitted data (scalar features, response, sample weight) --
        self.scalar_features_ = scalar_features
        self.y_ = y
        self.weight_ = sample_weight

        # -- Validate functional features and times --
        self.functional_features_ = []
        self.functional_time_ = []
        for i in range(self.n_functional_features_in_):
            self.functional_features_.append(check_array(functional_features[i], ensure_2d=False, dtype=self._input_dtype))
            self.functional_time_.append(check_array(functional_time[i], ensure_2d=False, dtype=self._input_dtype))

        # -- preprocess weights --
        self.preprocessed_weight_ = self.weight_ * (self.weight_.shape[0] / self.weight_.sum())

        # --- Fit one FPCA per functional feature ---
        self.fpca_models_: List[FunctionalPCA] = []
        for i in range(self.n_functional_features_in_):
            cfg = self._get_fpca_config(i)
            fpca = FunctionalPCA(
                assume_measurement_error=cfg.assume_measurement_error,
                num_points_reg_grid=cfg.num_points_reg_grid,
                mu_cov_params=cfg.mu_cov_params or FunctionalPCAMuCovParams(),
                user_params=cfg.user_params or FunctionalPCAUserDefinedParams(),
                verbose=cfg.verbose,
            )
            fpca.fit(functional_time[i], functional_features[i], **cfg.fit_params)
            self.fpca_models_.append(fpca)

        # --- Assemble design matrix: scalar | scores_0 | scores_1 | ... ---
        feature_list = [scalar_features]
        for fpca in self.fpca_models_:
            feature_list.append(fpca.xi_)
        design_matrix = np.hstack(feature_list)

        # --- Fit ElasticNet ---
        self.linear_model_ = ElasticNet(**self.linear_opts)
        self.linear_model_.fit(design_matrix, y, sample_weight=self.preprocessed_weight_)

        # -- get fitted values --
        self.fitted_values_ = self.linear_model_.predict(design_matrix)
        return self

    def predict(
        self,
        new_functional_time: List[List[np.ndarray]],
        new_functional_features: List[List[np.ndarray]],
        new_scalar_features: np.ndarray,
    ) -> np.ndarray:
        """Predict using the fitted partial functional linear model.

        Parameters
        ----------
        new_functional_time : list of list-of-ndarray
            Time vectors for each functional feature (same structure as ``functional_time`` in ``fit()``).
        new_functional_features : list of list-of-ndarray
            Observations for each functional feature.
        new_scalar_features : np.ndarray of shape (n_samples, n_scalar_features)
            Scalar predictors.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,) or (n_samples, n_classes)
            Predicted values.  The transform depends on ``family``; see ``ElasticNet.predict`` for details.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If the model has not been fitted yet.
        ValueError
            If the number of new functional features or scalar features does not match the number seen during
            ``fit()``.
        """
        check_is_fitted(self, ["linear_model_", "fpca_models_"])
        if len(new_functional_features) != self.n_functional_features_in_:
            raise ValueError("Number of new functional features does not match the number seen during fit")
        if new_scalar_features.shape[1] != self.n_scalar_features_in_:
            raise ValueError("Number of new scalar features does not match the number seen during fit")

        new_scalar_features = check_array(new_scalar_features, ensure_2d=True, dtype=self._input_dtype)
        new_functional_time = [check_array(ft, ensure_2d=False, dtype=self._input_dtype) for ft in new_functional_time]
        new_functional_features = [check_array(ff, ensure_2d=False, dtype=self._input_dtype) for ff in new_functional_features]

        feature_list = [new_scalar_features]
        for i, fpca in enumerate(self.fpca_models_):
            new_xi, _, _, _ = fpca.predict(new_functional_features[i], new_functional_time[i])
            feature_list.append(new_xi)
        return self.linear_model_.predict(np.hstack(feature_list))

    def fitted_values(self) -> np.ndarray:
        """Return fitted values on the training data.

        Returns
        -------
        np.ndarray of shape (n_samples,) or (n_samples, n_classes)
            Predictions evaluated on the training data ``functional_features_`` and ``scalar_features_``.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If the model has not been fitted yet.
        """
        check_is_fitted(self, ["linear_model_", "fpca_models_", "fitted_values_"])
        return self.fitted_values_
