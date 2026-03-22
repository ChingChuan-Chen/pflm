import numpy as np
from enum import Enum
from sklearn.base import MultiOutputMixin, RegressorMixin
from typing import Tuple, Dict, Optional
from pflm.pflm.utils import fit_gaussian, fit_nongaussian


class LinearModelFamily(Enum):
    GAUSSIAN = 'gaussian'
    BINOMIAL = 'binomial'
    POISSON = 'poisson'


class ElasticNet(MultiOutputMixin, RegressorMixin):
    """
    ElasticNet Linear Model (contains Logistic and Poisson) based on ADMM.

    For the Gaussian family the objective is:

        min_w  1/(2n) ||y - Xw||_2^2 + alpha * ||w||_1 + (beta / 2) * ||w||_2^2

    where the penalty coefficients are derived from the constructor arguments:

        alpha  = self.alpha * self.l1_ratio          (L1 / sparsity penalty)
        beta   = self.alpha * (1 - self.l1_ratio)     (L2 / ridge penalty)

    For the Binomial (logistic) family the squared-error loss is replaced by
    the negative log-likelihood  -sum_i [ y_i eta_i + log(1 - sigma(eta_i)) ].

    For the Poisson family the loss becomes  -sum_i [ y_i eta_i - exp(eta_i) ].

    The solver uses ADMM (Alternating Direction Method of Multipliers);
    see ``fit_gaussian`` / ``fit_nongaussian`` for the detailed update formulae.

    parameters
    ----------
    alpha: float, optional (default=1.0). Constant that multiplies the penalty terms. Defaults to 1.0.
    l1_ratio: float, optional (default=0.5). The ElasticNet mixing parameter, with 0 <= l1_ratio <= 1. For l1_ratio = 0 the penalty is an L2 penalty.
        For l1_ratio = 1 it is an L1 penalty. For 0 < l1_ratio < 1, the penalty is a combination of L1 and L2.
    fit_intercept: bool, optional (default=True). If set to False, the model will not learn an intercept.
                   This works only if family is 'GAUSSIAN'.
    family: LinearModelFamily, optional (default=LinearModelFamily.GAUSSIAN). 'BINOMIAL' or 'POISSON' are available.
    max_iter: int, optional (default=1000). Maximum number of iterations.
    abs_tol: float, optional (default=1e-3).
             Tolerance for stopping criteria of the absolute difference between each iteration.
    rel_tol: float, optional (default=1e-4).
             Tolerance for stopping criteria of the relative difference between each iteration.
    min_iter: int, optional (default=3). Minimum number of iterations.

    Attributes
    ----------
    coef_: np.ndarray, shape = [n_features], the coefficients of the linear model.
    intercept_: float, the intercept of the model.
    n_iter: int, the number of iterations during fitting.
    fitted: bool, this attribute indicates whether the model has been fitted or not.

    Examples
    --------
    >>> import numpy as np
    >>> from pflm.linear_model import ElasticNet
    >>> X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    >>> y = np.dot(X, np.array([1, 2])) + 3
    >>> reg = ElasticNet().fit(X, y)
    >>> reg.coef_
    array([ 1.00401339, 1.9971606 ])
    """
    intercept_: float = 0.0
    coef_: Optional[np.ndarray] = None
    n_iter: Optional[int] = None
    fitted: bool = False

    def __init__(
        self, alpha: float = 1.0, l1_ratio: float = 0.5, fit_intercept: bool = True, family: LinearModelFamily = LinearModelFamily.GAUSSIAN,
        max_iter: int = 1000, rho: float = 1.0, abs_tol: float = 1e-4, rel_tol: float = 1e-5, min_iter: int = 3
    ):
        if alpha < 0:
            raise ValueError('alpha must be non-negative.')
        if not (0.0 <= l1_ratio <= 1.0):
            raise ValueError('l1_ratio must be between 0 and 1.')
        self.alpha: float = alpha
        self.l1_ratio: float = l1_ratio
        self.fit_intercept: bool = fit_intercept if family == LinearModelFamily.GAUSSIAN else False
        self.max_iter: int = max_iter
        self.family: LinearModelFamily = family
        if self.family not in [LinearModelFamily.GAUSSIAN, LinearModelFamily.BINOMIAL, LinearModelFamily.POISSON]:
            raise ValueError('Invalid family')
        self.family_code: Dict = {LinearModelFamily.BINOMIAL: 0, LinearModelFamily.POISSON: 1, LinearModelFamily.GAUSSIAN: 99}[self.family]
        self.rho: float = rho
        self.abs_tol: float = abs_tol
        self.rel_tol: float = rel_tol
        self.min_iter: int = min_iter

    @staticmethod
    def _preprocess_data(
        X, y, family: LinearModelFamily, fit_intercept: bool
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Preprocess the data for fitting the linear model.

        If family is 'gaussian' and fit_intercept is True, then center the data by subtracting the mean of each column of X and the mean of y.
        Otherwise, do not center the data.

          Centered_X = X - X_offset, Centered_Y = Y - Y_offset

        Parameters
        ----------
        X: {ndarray} of shape (n_samples, n_features). The training data.
        y: {ndarray} of shape (n_samples,) or (n_samples, n_targets). The target values.
        family: {LinearModelFamily}. The target distribution family.
        fit_intercept: {bool}. If set to False, the model will not learn the intercept.

        Returns
        -------
        X: {ndarray} of shape (n_samples, n_features). The normalized training data.
        y: {ndarray} of shape (n_samples,). Normalized target values which is only meaningful when family is 'gaussian'
           and fit_intercept is True.
        X_offset: {ndarray} of shape (n_features,). The mean per column of input X.
        y_offset: float or ndarray of shape (1,). The mean of input y.
        """
        # TODO: add weights
        n_features = X.shape[1]
        if fit_intercept:
            X_offset = np.mean(X, axis=0)
            X -= X_offset
        else:
            X_offset = np.zeros(n_features)

        if family == LinearModelFamily.GAUSSIAN and fit_intercept:
            y_offset = np.mean(y, axis=0)
            y -= y_offset
        else:
            y_offset = 0.0
        return X, y, X_offset, y_offset

    def fit(self, X, y, sample_weight=None):
        """
        Fit the model with L1 and L2 regularizations.

        Parameters
        ----------
        X: {array-like} of shape (n_samples, n_features)
                  Training data.
        y: array-like of shape (n_samples,) or (n_samples, n_targets)
                  Target values. Will be cast to X's dtype if necessary.
        sample_weight: array-like of shape (n_samples,), default=None
                              Individual weights for each sample.

        Returns
        -------
        self: object. Fitted model.
        """
        # TODO: add weights

        X, y, X_offset, y_offset = ElasticNet._preprocess_data(
            X.copy().astype(np.float64), y.copy().astype(np.float64), self.family, self.fit_intercept
        )

        # Map sklearn-style (alpha, l1_ratio) → solver-level (l1_reg, l2_reg):
        #   l1_reg (L1) = self.alpha * self.l1_ratio
        #   l2_reg (L2) = self.alpha * (1 - self.l1_ratio)
        l1_reg = self.alpha * self.l1_ratio
        l2_reg = self.alpha * (1.0 - self.l1_ratio)

        if self.family == LinearModelFamily.GAUSSIAN:
            coef_out, self.n_iter = fit_gaussian(
                X, y, l1_reg, l2_reg, self.rho, self.max_iter, self.abs_tol, self.rel_tol, self.min_iter
            )
        else:
            self.intercept_, coef_out, self.n_iter = fit_nongaussian(
                X, y, self.family_code, l1_reg, l2_reg, self.rho, self.max_iter, self.abs_tol, self.rel_tol, self.min_iter
            )

        self.coef_ = coef_out
        if self.family == LinearModelFamily.GAUSSIAN:
            if self.fit_intercept:
                self.intercept_ = y_offset - np.dot(X_offset, self.coef_)
            else:
                self.intercept_ = y_offset
        self.fitted = True
        return self

    def predict(self, new_X: np.ndarray) -> np.ndarray:
        """
        Predict using the linear model.

        Parameters
        ----------
        new_X: array-like, shape (n_samples, n_features) New samples.

        Returns
        -------
        C: array, shape (n_samples,) Predicted values.
            When family == 'gaussian', it returns intercept_ + new_X * b.
            When family == 'binomial', it returns 1 / (1 + exp(-new_X * b)).
            When family == 'poisson', it returns exp(new_X * b).
        """
        if not self.fitted:
            raise ValueError('Model not fitted yet.')
        if self.family == LinearModelFamily.GAUSSIAN:
            return np.dot(new_X.astype(np.float64), self.coef_) + self.intercept_
        elif self.family == LinearModelFamily.BINOMIAL:
            return 1.0 / (1.0 + np.exp(- self.intercept_ - np.dot(new_X.astype(np.float64), self.coef_)))
        elif self.family == LinearModelFamily.POISSON:
            return np.exp(self.intercept_ + np.dot(new_X.astype(np.float64), self.coef_))
