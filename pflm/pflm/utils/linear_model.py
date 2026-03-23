import numpy as np
from enum import Enum
from sklearn.base import MultiOutputMixin, RegressorMixin
from typing import Tuple, Dict, Optional
from pflm.pflm.utils import (
    fit_gaussian_f32, fit_gaussian_f64,
    fit_nongaussian_f32, fit_nongaussian_f64,
    fit_multinomial_f32, fit_multinomial_f64,
)


class LinearModelFamily(Enum):
    BINOMIAL = 1
    POISSON = 2
    GAMMA = 3
    MULTINOMIAL = 4
    TWEEDIE = 5
    GAUSSIAN = 99


class ElasticNet(MultiOutputMixin, RegressorMixin):
    """ElasticNet linear model solved via ADMM.

    Supports Gaussian, Binomial, Poisson, Gamma, Tweedie and Multinomial
    families.  For the Gaussian family the objective is:

    .. math::

        \\min_w \\frac{1}{2n} \\|y - Xw\\|_2^2
        + \\alpha \\|w\\|_1 + \\frac{\\beta}{2} \\|w\\|_2^2

    where ``alpha = self.alpha * self.l1_ratio`` (L1 penalty) and
    ``beta = self.alpha * (1 - self.l1_ratio)`` (L2 penalty).

    For non-Gaussian families the squared-error loss is replaced by the
    corresponding negative log-likelihood.  The solver uses ADMM
    (Alternating Direction Method of Multipliers); see
    ``fit_gaussian`` / ``fit_nongaussian`` / ``fit_multinomial`` for the
    detailed update formulae.

    Parameters
    ----------
    alpha : float, default=1.0
        Constant that multiplies the penalty terms.
    l1_ratio : float, default=0.5
        ElasticNet mixing parameter with ``0 <= l1_ratio <= 1``.
        ``l1_ratio=0`` corresponds to an L2 penalty, ``l1_ratio=1`` to L1.
    fit_intercept : bool, default=True
        Whether to fit an intercept.  Only effective when
        ``family=GAUSSIAN``; other families always include an intercept.
    family : LinearModelFamily, default=LinearModelFamily.GAUSSIAN
        Distribution family.  One of ``GAUSSIAN``, ``BINOMIAL``,
        ``POISSON``, ``GAMMA``, ``TWEEDIE``, or ``MULTINOMIAL``.
    power : float, default=1.5
        Tweedie variance power parameter.  Only used when
        ``family=TWEEDIE``.  Must not be 0, 1, or 2.
    max_iter : int, default=1000
        Maximum number of ADMM iterations.
    rho : float, default=1.0
        ADMM augmented-Lagrangian parameter.
    abs_tol : float, default=1e-4
        Absolute tolerance for the ADMM convergence criterion.
    rel_tol : float, default=1e-5
        Relative tolerance for the ADMM convergence criterion.
    min_iter : int, default=3
        Minimum number of ADMM iterations before convergence is checked.

    Attributes
    ----------
    coef_ : np.ndarray of shape (n_features,) or (n_classes, n_features)
        Fitted coefficients.
    intercept_ : float or np.ndarray of shape (n_classes,)
        Fitted intercept(s).
    n_iter : int
        Number of ADMM iterations performed during fitting.
    fitted : bool
        Whether the model has been fitted.

    Examples
    --------
    >>> import numpy as np
    >>> from pflm.pflm.utils import ElasticNet
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
        power: float = 1.5, max_iter: int = 1000, rho: float = 1.0, abs_tol: float = 1e-4, rel_tol: float = 1e-5, min_iter: int = 3
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
        _valid = {LinearModelFamily.GAUSSIAN, LinearModelFamily.BINOMIAL, LinearModelFamily.POISSON,
                  LinearModelFamily.GAMMA, LinearModelFamily.TWEEDIE, LinearModelFamily.MULTINOMIAL}
        if self.family not in _valid:
            raise ValueError('Invalid family')
        self.power: float = power
        self.rho: float = rho
        self.abs_tol: float = abs_tol
        self.rel_tol: float = rel_tol
        self.min_iter: int = min_iter

    @staticmethod
    def preprocess_data(
        X: np.ndarray, y: np.ndarray, family: LinearModelFamily, fit_intercept: bool,
        sample_weight: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Centre data for the Gaussian-with-intercept case.

        When ``family`` is ``GAUSSIAN`` and ``fit_intercept`` is ``True``,
        subtract the (optionally weighted) column means from *X* and the
        mean from *y*.  Otherwise leave the data unchanged.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data (modified in-place).
        y : np.ndarray of shape (n_samples,)
            Target values (modified in-place when centred).
        family : LinearModelFamily
            Distribution family.
        fit_intercept : bool
            Whether the model includes an intercept.
        sample_weight : np.ndarray of shape (n_samples,), optional
            Per-sample weights used to compute weighted means.

        Returns
        -------
        X : np.ndarray of shape (n_samples, n_features)
            Centred training data.
        y : np.ndarray of shape (n_samples,)
            Centred target values (only meaningful for Gaussian with
            intercept).
        X_offset : np.ndarray of shape (n_features,)
            Column means subtracted from *X*.
        y_offset : np.ndarray of shape (1,)
            Mean subtracted from *y*.
        """
        n_features = X.shape[1]
        if fit_intercept:
            X_offset = np.average(X, weights=sample_weight, axis=0).astype(X.dtype)
            X -= X_offset
        else:
            X_offset = np.zeros(n_features, dtype=X.dtype)

        if family == LinearModelFamily.GAUSSIAN and fit_intercept:
            y_offset = np.average(y, weights=sample_weight, axis=0).astype(X.dtype)
            y -= y_offset
        else:
            y_offset = np.zeros(1, dtype=X.dtype)
        return X, y, X_offset, y_offset

    def fit(self, X, y, sample_weight=None):
        """Fit the model with L1 and L2 regularization.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.  Will be cast to *X*'s dtype if necessary.
        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample.  Weights are normalized
            so that ``sum(w) == n``.

        Returns
        -------
        self : ElasticNet
            Fitted estimator.

        Raises
        ------
        ValueError
            If ``sample_weight`` contains negative values, is all zeros,
            or has the wrong length.  Also raised when *y* violates the
            constraints of the chosen ``family``.
        """

        # Store input dtype for later use
        self._input_dtype = X.dtype
        n = X.shape[0]

        # ---------- Validate y for the chosen family ----------
        y_arr = np.asarray(y).ravel()
        if self.family == LinearModelFamily.BINOMIAL:
            unique = np.unique(y_arr)
            if not np.all(np.isin(unique, [0, 1])):
                raise ValueError('For BINOMIAL family, y must contain only 0 and 1.')
        elif self.family == LinearModelFamily.POISSON:
            if np.any(y_arr < 0):
                raise ValueError('For POISSON family, y must be non-negative.')
        elif self.family == LinearModelFamily.GAMMA:
            if np.any(y_arr <= 0):
                raise ValueError('For GAMMA family, y must be strictly positive.')
        elif self.family == LinearModelFamily.TWEEDIE:
            if np.any(y_arr < 0):
                raise ValueError('For TWEEDIE family, y must be non-negative.')
        elif self.family == LinearModelFamily.MULTINOMIAL:
            if np.any(y_arr < 0) or not np.all(y_arr == np.floor(y_arr)):
                raise ValueError('For MULTINOMIAL family, y must be non-negative integers.')
            if len(np.unique(y_arr)) < 2:
                raise ValueError('For MULTINOMIAL family, y must contain at least 2 classes.')

        # Validate and normalise sample weights (sum → n)
        if sample_weight is not None:
            sw = np.asarray(sample_weight, dtype=self._input_dtype).ravel()
            if sw.shape[0] != n:
                raise ValueError('sample_weight must have the same length as y.')
            if np.any(sw < 0):
                raise ValueError('sample_weight cannot contain negative values.')
            if np.sum(sw) == 0:
                raise ValueError('At least one sample_weight must be positive.')
            sw = sw * (n / sw.sum())
        else:
            sw = np.ones(n, dtype=self._input_dtype)

        # Preprocess the data (center if fit_intercept is True and family is 'gaussian')
        X_, y_, X_offset, y_offset = ElasticNet.preprocess_data(
            X.copy().astype(self._input_dtype), y.copy().astype(self._input_dtype),
            self.family, self.fit_intercept, sample_weight=sw
        )

        # Map sklearn-style (alpha, l1_ratio) → solver-level (l1_reg, l2_reg):
        #   l1_reg (L1) = self.alpha * self.l1_ratio
        #   l2_reg (L2) = self.alpha * (1 - self.l1_ratio)
        l1_reg = self.alpha * self.l1_ratio
        l2_reg = self.alpha * (1.0 - self.l1_ratio)

        if self.family == LinearModelFamily.GAUSSIAN:
            _fit = fit_gaussian_f32 if self._input_dtype == np.float32 else fit_gaussian_f64
            coef_out, self.n_iter = _fit(X_, y_, sw, l1_reg, l2_reg, self.rho, self.max_iter, self.abs_tol, self.rel_tol, self.min_iter)
        elif self.family == LinearModelFamily.MULTINOMIAL:
            _fit = fit_multinomial_f32 if self._input_dtype == np.float32 else fit_multinomial_f64
            n_classes = int(y_.max()) + 1
            self.intercept_, coef_out, self.n_iter = _fit(
                X_, y_, sw, n_classes, l1_reg, l2_reg, self.rho, self.max_iter, self.abs_tol, self.rel_tol, self.min_iter
            )
        else:
            _fit = fit_nongaussian_f32 if self._input_dtype == np.float32 else fit_nongaussian_f64
            self.intercept_, coef_out, self.n_iter = _fit(
                X_, y_, sw, self.family.value, self.power, l1_reg, l2_reg, self.rho, self.max_iter, self.abs_tol, self.rel_tol, self.min_iter
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
        """Predict using the fitted linear model.

        Parameters
        ----------
        new_X : np.ndarray of shape (n_samples, n_features)
            New samples.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,) or (n_samples, n_classes)
            Predicted values.  The transform depends on ``family``:

            - ``GAUSSIAN``: ``intercept + X @ coef``
            - ``BINOMIAL``: ``sigmoid(intercept + X @ coef)``
            - ``POISSON`` / ``GAMMA`` / ``TWEEDIE``: ``exp(intercept + X @ coef)``
            - ``MULTINOMIAL``: softmax probabilities of shape
              ``(n_samples, n_classes)``

        Raises
        ------
        ValueError
            If the model has not been fitted yet.
        """
        if not self.fitted:
            raise ValueError('Model not fitted yet.')
        Xf = new_X.astype(np.float64)
        if self.family == LinearModelFamily.GAUSSIAN:
            return np.dot(Xf, self.coef_) + self.intercept_
        elif self.family == LinearModelFamily.BINOMIAL:
            return 1.0 / (1.0 + np.exp(-self.intercept_ - np.dot(Xf, self.coef_)))
        elif self.family in (LinearModelFamily.POISSON, LinearModelFamily.GAMMA, LinearModelFamily.TWEEDIE):
            return np.exp(self.intercept_ + np.dot(Xf, self.coef_))
        elif self.family == LinearModelFamily.MULTINOMIAL:
            eta = self.intercept_ + Xf @ self.coef_.T  # (n, K)
            eta -= eta.max(axis=1, keepdims=True)
            e = np.exp(eta)
            return e / e.sum(axis=1, keepdims=True)
