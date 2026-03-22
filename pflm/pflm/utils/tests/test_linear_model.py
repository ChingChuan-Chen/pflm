import numpy as np
from numpy.testing import assert_allclose
from sklearn.linear_model import ElasticNet as SklearnElasticNet

from pflm.pflm.utils import LinearModelFamily, ElasticNet


def _make_regression_data(seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(240, 8))
    coef = np.array([1.6, -2.2, 0.0, 0.0, 0.9, -0.4, 0.0, 1.1])
    y = 1.2 + x @ coef + rng.normal(scale=0.08, size=x.shape[0])
    return x, y


def _fit_both_models(alpha: float, l1_ratio: float, fit_intercept: bool):
    x, y = _make_regression_data(seed=42)

    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=fit_intercept, family=LinearModelFamily.GAUSSIAN)
    model.fit(x, y)

    sk_model = SklearnElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        fit_intercept=fit_intercept,
        max_iter=4000,
        tol=1e-8,
        random_state=0,
        selection="cyclic",
    )
    sk_model.fit(x, y)
    return model, sk_model, x


def test_elastic_net_matches_sklearn_with_intercept():
    model, sk_model, x = _fit_both_models(alpha=0.08, l1_ratio=0.7, fit_intercept=True)

    assert_allclose(model.coef_, sk_model.coef_, rtol=8e-2, atol=3e-2)
    assert_allclose(model.intercept_, sk_model.intercept_, rtol=2e-2, atol=2e-2)
    assert_allclose(model.predict(x), sk_model.predict(x), rtol=2e-2, atol=5e-2)


def test_elastic_net_matches_sklearn_without_intercept():
    model, sk_model, x = _fit_both_models(alpha=0.05, l1_ratio=0.35, fit_intercept=False)

    assert_allclose(model.coef_, sk_model.coef_, rtol=8e-2, atol=3e-2)
    assert_allclose(model.intercept_, 0.0, atol=1e-12)
    assert_allclose(model.predict(x), sk_model.predict(x), rtol=2e-2, atol=5e-2)
