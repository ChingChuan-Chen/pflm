import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.exceptions import ConvergenceWarning, NotFittedError
from sklearn.linear_model import ElasticNet as SklearnElasticNet
from sklearn.linear_model import GammaRegressor as SklearnGammaRegressor
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.linear_model import PoissonRegressor as SklearnPoissonRegressor
from sklearn.linear_model import TweedieRegressor as SklearnTweedieRegressor

from pflm.pflm.utils import ElasticNet, LinearModelFamily


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def _make_regression_data(dtype, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(240, 8)).astype(dtype)
    coef = np.array([1.6, -2.2, 0.0, 0.0, 0.9, -0.4, 0.0, 1.1], dtype=dtype)
    y = (1.2 + x @ coef + rng.normal(scale=0.08, size=x.shape[0])).astype(dtype)
    return x, y


def _fit_sklearn_reference(estimator, x, y, sample_weight=None):
    """Fit a sklearn reference estimator while silencing known upstream warnings."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", message="'penalty' was deprecated.*", category=FutureWarning)
        estimator.fit(x, y, sample_weight=sample_weight)
    return estimator


def _fit_both_models(alpha: float, l1_ratio: float, fit_intercept: bool, dtype):
    x, y = _make_regression_data(dtype=dtype, seed=42)

    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=fit_intercept, family=LinearModelFamily.GAUSSIAN)
    model.fit(x, y)

    tol = 1e-6 if np.dtype(dtype) == np.float32 else 1e-8
    sk_model = SklearnElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        fit_intercept=fit_intercept,
        max_iter=20000,
        tol=tol,
        random_state=0,
        selection="cyclic",
    )
    _fit_sklearn_reference(sk_model, x, y)
    return model, sk_model, x


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_elastic_net_matches_sklearn_with_intercept(dtype):
    model, sk_model, x = _fit_both_models(alpha=0.08, l1_ratio=0.7, fit_intercept=True, dtype=dtype)

    assert_allclose(model.coef_, sk_model.coef_, rtol=8e-2, atol=3e-2)
    assert_allclose(model.intercept_, sk_model.intercept_, rtol=2e-2, atol=2e-2)
    assert_allclose(model.predict(x), sk_model.predict(x), rtol=2e-2, atol=5e-2)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_elastic_net_matches_sklearn_without_intercept(dtype):
    model, sk_model, x = _fit_both_models(alpha=0.05, l1_ratio=0.35, fit_intercept=False, dtype=dtype)

    assert_allclose(model.coef_, sk_model.coef_, rtol=8e-2, atol=3e-2)
    assert_allclose(model.intercept_, 0.0, atol=1e-12)
    assert_allclose(model.predict(x), sk_model.predict(x), rtol=2e-2, atol=5e-2)


# ---------------------------------------------------------------------------
# Non-Gaussian (Binomial / Poisson) tests
# ---------------------------------------------------------------------------
#
# Our ADMM formulation uses an UNSCALED negative log-likelihood:
#     min_w  NLL(w) + l1_reg * ||w[1:]||_1 + (l2_reg/2) * ||w[1:]||_2^2
#
# sklearn LogisticRegression averages the loss by n:
#     min_w  (1/n) NLL(w) + (l1_ratio/C) * ||w||_1 + ((1-l1_ratio)/(2C)) * ||w||_2^2
#
# Dividing our objective by n doesn't change the minimizer, so the mapping is:
#     C_sklearn = n / alpha
#
# sklearn PoissonRegressor (L2 only):
#     min_w  (1/n) * Poisson_NLL(w) + (alpha_sk/2) * ||w||_2^2
# Mapping:  alpha_sk = alpha_our / n   (with l1_ratio = 0)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_binomial_elastic_net_matches_sklearn(dtype):
    """Compare Binomial (logistic) ElasticNet against sklearn LogisticRegression."""
    n, p = 300, 6
    alpha = 0.05
    l1_ratio = 0.5
    rng = np.random.default_rng(42)

    x = rng.normal(size=(n, p)).astype(dtype)
    true_coef = np.array([1.0, -0.5, 0.0, 0.8, 0.0, -0.3], dtype=dtype)
    eta = x @ true_coef + 0.5  # true intercept = 0.5
    prob = 1.0 / (1.0 + np.exp(-eta))
    y = (rng.random(n) < prob).astype(dtype)

    # Our model
    model = ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        family=LinearModelFamily.BINOMIAL,
        max_iter=3000,
        abs_tol=1e-6,
        rel_tol=1e-6,
        min_iter=5,
        rho=1.0,
    )
    model.fit(x, y)

    # sklearn — C = n / alpha
    sk_model = SklearnLogisticRegression(
        penalty="elasticnet",
        solver="saga",
        C=n / alpha,
        l1_ratio=l1_ratio,
        max_iter=20000,
        tol=1e-6 if np.dtype(dtype) == np.float32 else 1e-8,
        fit_intercept=True,
        random_state=0,
    )
    _fit_sklearn_reference(sk_model, x, y)

    assert_allclose(model.coef_, sk_model.coef_.ravel(), rtol=0.15, atol=0.08)
    assert_allclose(model.intercept_, sk_model.intercept_[0], rtol=0.15, atol=0.08)
    assert_allclose(model.predict(x), sk_model.predict_proba(x)[:, 1], rtol=0.1, atol=0.06)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_poisson_l2_matches_sklearn(dtype):
    """Compare Poisson ElasticNet (L2-only) against sklearn PoissonRegressor."""
    n, p = 300, 5
    alpha_our = 0.5
    l1_ratio = 0.0  # pure L2 — PoissonRegressor only supports L2
    rng = np.random.default_rng(123)

    x = rng.normal(size=(n, p)).astype(dtype) * 0.5
    true_coef = np.array([0.4, -0.3, 0.2, 0.0, -0.1], dtype=dtype)
    eta = x @ true_coef + 0.3  # true intercept = 0.3
    y = rng.poisson(lam=np.exp(eta)).astype(dtype)

    # Our model
    model = ElasticNet(
        alpha=alpha_our,
        l1_ratio=l1_ratio,
        family=LinearModelFamily.POISSON,
        max_iter=3000,
        abs_tol=1e-6,
        rel_tol=1e-6,
        min_iter=5,
        rho=1.0,
    )
    model.fit(x, y)

    # sklearn — alpha_sk = alpha_our / n
    sk_model = SklearnPoissonRegressor(
        alpha=alpha_our / n,
        max_iter=5000,
        tol=1e-10,
        fit_intercept=True,
    )
    sk_model.fit(x, y)

    assert_allclose(model.coef_, sk_model.coef_, rtol=0.15, atol=0.08)
    assert_allclose(model.intercept_, sk_model.intercept_, rtol=0.15, atol=0.08)
    assert_allclose(model.predict(x), sk_model.predict(x), rtol=0.1, atol=0.15)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_gamma_l2_matches_sklearn(dtype):
    """Compare Gamma ElasticNet (L2-only) against sklearn GammaRegressor."""
    n, p = 300, 5
    alpha_our = 0.5
    l1_ratio = 0.0
    rng = np.random.default_rng(77)

    x = rng.normal(size=(n, p)).astype(dtype) * 0.3
    true_coef = np.array([0.3, -0.2, 0.15, 0.0, -0.1], dtype=dtype)
    eta = x @ true_coef + 0.5
    mu = np.exp(eta)
    y = rng.gamma(shape=5.0, scale=mu / 5.0).astype(dtype)

    model = ElasticNet(
        alpha=alpha_our,
        l1_ratio=l1_ratio,
        family=LinearModelFamily.GAMMA,
        max_iter=3000,
        abs_tol=1e-6,
        rel_tol=1e-6,
        min_iter=5,
        rho=1.0,
    )
    model.fit(x, y)

    sk_model = SklearnGammaRegressor(
        alpha=alpha_our / n,
        max_iter=5000,
        tol=1e-10,
        fit_intercept=True,
    )
    sk_model.fit(x, y)

    assert_allclose(model.coef_, sk_model.coef_, rtol=0.15, atol=0.08)
    assert_allclose(model.intercept_, sk_model.intercept_, rtol=0.15, atol=0.08)
    assert_allclose(model.predict(x), sk_model.predict(x), rtol=0.1, atol=0.15)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_tweedie_l2_matches_sklearn(dtype):
    """Compare Tweedie ElasticNet (L2-only, power=1.5) against sklearn TweedieRegressor."""
    n, p = 300, 5
    alpha_our = 0.5
    l1_ratio = 0.0
    tw_power = 1.5
    rng = np.random.default_rng(99)

    x = rng.normal(size=(n, p)).astype(dtype) * 0.3
    true_coef = np.array([0.3, -0.2, 0.1, 0.0, -0.05], dtype=dtype)
    eta = x @ true_coef + 0.5
    mu = np.exp(eta)
    # simulate compound Poisson-Gamma (Tweedie p=1.5) via Poisson-sum-of-Gammas
    y = np.zeros(n, dtype=dtype)
    for i in range(n):
        cnt = rng.poisson(lam=mu[i] ** (2 - tw_power) / (2 - tw_power))
        if cnt > 0:
            y[i] = rng.gamma(
                shape=(2 - tw_power) / (tw_power - 1), scale=mu[i] ** (tw_power - 1) / (2 - tw_power) * (tw_power - 1), size=cnt
            ).sum()

    model = ElasticNet(
        alpha=alpha_our,
        l1_ratio=l1_ratio,
        family=LinearModelFamily.TWEEDIE,
        power=tw_power,
        max_iter=3000,
        abs_tol=1e-6,
        rel_tol=1e-6,
        min_iter=5,
        rho=1.0,
    )
    model.fit(x, y)

    sk_model = SklearnTweedieRegressor(
        power=tw_power,
        alpha=alpha_our / n,
        max_iter=5000,
        tol=1e-10,
        fit_intercept=True,
    )
    sk_model.fit(x, y)

    assert_allclose(model.coef_, sk_model.coef_, rtol=0.2, atol=0.1)
    assert_allclose(model.intercept_, sk_model.intercept_, rtol=0.2, atol=0.1)
    assert_allclose(model.predict(x), sk_model.predict(x), rtol=0.15, atol=0.2)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_multinomial_matches_sklearn(dtype):
    """Compare Multinomial ElasticNet against sklearn LogisticRegression(multi_class='multinomial')."""
    n, p, K = 400, 6, 3
    alpha = 0.02
    l1_ratio = 0.5
    rng = np.random.default_rng(55)

    x = rng.normal(size=(n, p)).astype(dtype)
    true_W = np.array(
        [
            [1.0, -0.5, 0.0, 0.8, 0.0, -0.3],
            [-0.5, 1.0, 0.3, 0.0, -0.4, 0.0],
            [0.0, 0.0, -0.5, -0.3, 0.5, 0.2],
        ],
        dtype=dtype,
    )
    eta = x @ true_W.T + np.array([0.3, -0.2, 0.1])
    eta -= eta.max(axis=1, keepdims=True)
    prob = np.exp(eta) / np.exp(eta).sum(axis=1, keepdims=True)
    y_int = np.array([rng.choice(K, p=prob[i]) for i in range(n)])
    y = y_int.astype(dtype)

    model = ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        family=LinearModelFamily.MULTINOMIAL,
        max_iter=3000,
        abs_tol=1e-6,
        rel_tol=1e-6,
        min_iter=5,
        rho=1.0,
    )
    model.fit(x, y)

    sk_model = SklearnLogisticRegression(
        penalty="elasticnet",
        solver="saga",
        C=n / alpha,
        l1_ratio=l1_ratio,
        max_iter=20000,
        tol=1e-6 if np.dtype(dtype) == np.float32 else 1e-8,
        fit_intercept=True,
        random_state=0,
    )
    _fit_sklearn_reference(sk_model, x, y_int)

    assert model.coef_.shape == sk_model.coef_.shape
    assert_allclose(model.predict(x), sk_model.predict_proba(x), rtol=0.15, atol=0.1)


# ---------------------------------------------------------------------------
# Sample-weighted tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_weighted_gaussian_matches_sklearn(dtype):
    """Gaussian ElasticNet with sample_weight vs sklearn."""
    n, p = 240, 8
    rng = np.random.default_rng(42)
    x = rng.normal(size=(n, p)).astype(dtype)
    coef_true = np.array([1.6, -2.2, 0.0, 0.0, 0.9, -0.4, 0.0, 1.1], dtype=dtype)
    y = (1.2 + x @ coef_true + rng.normal(scale=0.08, size=n)).astype(dtype)
    sw = rng.uniform(0.2, 3.0, size=n).astype(dtype)

    alpha, l1_ratio = 0.08, 0.7
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=True, family=LinearModelFamily.GAUSSIAN)
    model.fit(x, y, sample_weight=sw)

    sk_model = SklearnElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        fit_intercept=True,
        max_iter=20000,
        tol=1e-6 if np.dtype(dtype) == np.float32 else 1e-8,
        selection="cyclic",
        random_state=0,
    )
    _fit_sklearn_reference(sk_model, x, y, sample_weight=sw)

    assert_allclose(model.coef_, sk_model.coef_, rtol=8e-2, atol=3e-2)
    assert_allclose(model.intercept_, sk_model.intercept_, rtol=2e-2, atol=2e-2)
    assert_allclose(model.predict(x), sk_model.predict(x), rtol=2e-2, atol=5e-2)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_weighted_binomial_matches_sklearn(dtype):
    """Binomial ElasticNet with sample_weight vs sklearn LogisticRegression."""
    n, p = 300, 6
    alpha = 0.05
    l1_ratio = 0.5
    rng = np.random.default_rng(42)

    x = rng.normal(size=(n, p)).astype(dtype)
    true_coef = np.array([1.0, -0.5, 0.0, 0.8, 0.0, -0.3], dtype=dtype)
    eta = x @ true_coef + 0.5
    prob = 1.0 / (1.0 + np.exp(-eta))
    y = (rng.random(n) < prob).astype(dtype)
    sw = rng.uniform(0.5, 2.0, size=n).astype(dtype)

    model = ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        family=LinearModelFamily.BINOMIAL,
        max_iter=3000,
        abs_tol=1e-6,
        rel_tol=1e-6,
        min_iter=5,
        rho=1.0,
    )
    model.fit(x, y, sample_weight=sw)

    sk_model = SklearnLogisticRegression(
        penalty="elasticnet",
        solver="saga",
        C=n / alpha,
        l1_ratio=l1_ratio,
        max_iter=20000,
        tol=1e-6 if np.dtype(dtype) == np.float32 else 1e-8,
        fit_intercept=True,
        random_state=0,
    )
    _fit_sklearn_reference(sk_model, x, y, sample_weight=sw)

    assert_allclose(model.coef_, sk_model.coef_.ravel(), rtol=0.15, atol=0.08)
    assert_allclose(model.intercept_, sk_model.intercept_[0], rtol=0.15, atol=0.08)
    assert_allclose(model.predict(x), sk_model.predict_proba(x)[:, 1], rtol=0.1, atol=0.06)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_weighted_poisson_l2_matches_sklearn(dtype):
    """Poisson ElasticNet (L2-only) with sample_weight vs sklearn PoissonRegressor."""
    n, p = 300, 5
    alpha_our = 0.5
    l1_ratio = 0.0
    rng = np.random.default_rng(123)

    x = rng.normal(size=(n, p)).astype(dtype) * 0.5
    true_coef = np.array([0.4, -0.3, 0.2, 0.0, -0.1], dtype=dtype)
    eta = x @ true_coef + 0.3
    y = rng.poisson(lam=np.exp(eta)).astype(dtype)
    sw = rng.uniform(0.5, 2.0, size=n).astype(dtype)

    model = ElasticNet(
        alpha=alpha_our,
        l1_ratio=l1_ratio,
        family=LinearModelFamily.POISSON,
        max_iter=3000,
        abs_tol=1e-6,
        rel_tol=1e-6,
        min_iter=5,
        rho=1.0,
    )
    model.fit(x, y, sample_weight=sw)

    sk_model = SklearnPoissonRegressor(
        alpha=alpha_our / n,
        max_iter=5000,
        tol=1e-10,
        fit_intercept=True,
    )
    _fit_sklearn_reference(sk_model, x, y, sample_weight=sw)

    assert_allclose(model.coef_, sk_model.coef_, rtol=0.15, atol=0.08)
    assert_allclose(model.intercept_, sk_model.intercept_, rtol=0.15, atol=0.08)
    assert_allclose(model.predict(x), sk_model.predict(x), rtol=0.1, atol=0.15)


def test_negative_sample_weight_raises():
    """sample_weight with negative values should raise ValueError."""
    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y = np.array([1.0, 2.0, 3.0])
    sw = np.array([1.0, -1.0, 1.0])
    model = ElasticNet(family=LinearModelFamily.GAUSSIAN)
    with pytest.raises(ValueError, match="negative"):
        model.fit(x, y, sample_weight=sw)


def test_incorrect_length_sample_weight_raises():
    """sample_weight with incorrect length should raise ValueError."""
    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y = np.array([1.0, 2.0, 3.0])
    sw = np.array([1.0, 2.0])
    model = ElasticNet(family=LinearModelFamily.GAUSSIAN)
    with pytest.raises(ValueError, match="length"):
        model.fit(x, y, sample_weight=sw)


def test_all_zero_sample_weight_raises():
    """sample_weight of all zeros should raise ValueError."""
    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y = np.array([1.0, 2.0, 3.0])
    sw = np.array([0.0, 0.0, 0.0])
    model = ElasticNet(family=LinearModelFamily.GAUSSIAN)
    with pytest.raises(ValueError, match="positive"):
        model.fit(x, y, sample_weight=sw)


def test_negative_alpha_raises():
    with pytest.raises(ValueError, match="alpha must be non-negative"):
        ElasticNet(alpha=-0.1)


@pytest.mark.parametrize("l1_ratio", [-0.1, 1.1])
def test_invalid_l1_ratio_raises(l1_ratio):
    with pytest.raises(ValueError, match="l1_ratio must be between 0 and 1"):
        ElasticNet(l1_ratio=l1_ratio)


def test_invalid_family_raises():
    with pytest.raises(ValueError, match="Invalid family"):
        ElasticNet(family="not_a_family")


def test_predict_before_fit_raises():
    """Calling predict on an unfitted model should raise NotFittedError."""
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    model = ElasticNet(family=LinearModelFamily.GAUSSIAN)
    with pytest.raises(NotFittedError):
        model.predict(x)


# ---------------------------------------------------------------------------
# y-validation tests per family
# ---------------------------------------------------------------------------


def test_binomial_y_not_binary_raises():
    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y = np.array([0.0, 1.0, 2.0])
    model = ElasticNet(family=LinearModelFamily.BINOMIAL)
    with pytest.raises(ValueError, match="BINOMIAL.*0 and 1"):
        model.fit(x, y)


def test_poisson_y_negative_raises():
    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y = np.array([1.0, -1.0, 2.0])
    model = ElasticNet(family=LinearModelFamily.POISSON)
    with pytest.raises(ValueError, match="POISSON.*non-negative"):
        model.fit(x, y)


def test_gamma_y_zero_raises():
    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y = np.array([1.0, 0.0, 2.0])
    model = ElasticNet(family=LinearModelFamily.GAMMA)
    with pytest.raises(ValueError, match="GAMMA.*strictly positive"):
        model.fit(x, y)


def test_gamma_y_negative_raises():
    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y = np.array([1.0, -0.5, 2.0])
    model = ElasticNet(family=LinearModelFamily.GAMMA)
    with pytest.raises(ValueError, match="GAMMA.*strictly positive"):
        model.fit(x, y)


def test_tweedie_y_negative_raises():
    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y = np.array([1.0, -1.0, 2.0])
    model = ElasticNet(family=LinearModelFamily.TWEEDIE)
    with pytest.raises(ValueError, match="TWEEDIE.*non-negative"):
        model.fit(x, y)


def test_multinomial_y_non_integer_raises():
    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y = np.array([0.0, 1.5, 1.0])
    model = ElasticNet(family=LinearModelFamily.MULTINOMIAL)
    with pytest.raises(ValueError, match="MULTINOMIAL.*non-negative integers"):
        model.fit(x, y)


def test_multinomial_y_single_class_raises():
    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y = np.array([0.0, 0.0, 0.0])
    model = ElasticNet(family=LinearModelFamily.MULTINOMIAL)
    with pytest.raises(ValueError, match="MULTINOMIAL.*at least 2 classes"):
        model.fit(x, y)


def test_multinomial_y_negative_raises():
    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y = np.array([0.0, -1.0, 1.0])
    model = ElasticNet(family=LinearModelFamily.MULTINOMIAL)
    with pytest.raises(ValueError, match="MULTINOMIAL.*non-negative integers"):
        model.fit(x, y)


# ---------------------------------------------------------------------------
# fitted_values tests
# ---------------------------------------------------------------------------


def test_fitted_values_before_fit_raises():
    model = ElasticNet()
    with pytest.raises(NotFittedError):
        model.fitted_values()


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_fitted_values_gaussian_matches_sklearn(dtype):
    """fitted_values() should equal sklearn predict(X_train) for Gaussian."""
    model, sk_model, x = _fit_both_models(alpha=0.08, l1_ratio=0.7, fit_intercept=True, dtype=dtype)
    assert_allclose(model.fitted_values(), sk_model.predict(x), rtol=2e-2, atol=5e-2)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_fitted_values_equals_predict_on_train(dtype):
    """fitted_values() must be identical to predict(X_) for all families."""
    x, y = _make_regression_data(dtype=dtype, seed=42)
    model = ElasticNet(alpha=0.08, l1_ratio=0.7, fit_intercept=True).fit(x, y)
    assert_allclose(model.fitted_values(), model.predict(model.X_), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_fitted_values_binomial(dtype):
    """fitted_values() for Binomial must equal predict(X_)."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 4)).astype(dtype)
    y = (x @ np.array([1, -1, 0.5, 0], dtype=dtype) > 0).astype(dtype)
    model = ElasticNet(alpha=0.05, l1_ratio=0.5, family=LinearModelFamily.BINOMIAL).fit(x, y)
    assert_allclose(model.fitted_values(), model.predict(model.X_), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_fitted_values_multinomial(dtype):
    """fitted_values() for Multinomial must equal predict(X_)."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=(200, 4)).astype(dtype)
    y = (np.argmax(x[:, :3], axis=1)).astype(dtype)
    model = ElasticNet(alpha=0.01, l1_ratio=0.5, family=LinearModelFamily.MULTINOMIAL).fit(x, y)
    assert_allclose(model.fitted_values(), model.predict(model.X_), rtol=0, atol=0)
