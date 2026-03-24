import pytest
import numpy as np
from numpy.testing import assert_allclose
from sklearn.exceptions import NotFittedError

from pflm.fpca import FunctionalDataGenerator, FunctionalPCAMuCovParams
from pflm.pflm.partial_flm import (
    FPCAConfig,
    PartialFunctionalLinearModel,
)
from pflm.pflm.utils import LinearModelFamily

# Suppress numerical-noise eigenvalue warnings from FPCA on synthetic data
pytestmark = pytest.mark.filterwarnings(
    "ignore:Eigenvalues contain.*:UserWarning"
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _generate_functional_data(n: int, nt: int = 51, seed: int = 42):
    """Generate one functional feature via FunctionalDataGenerator."""
    t = np.linspace(0.0, 10.0, nt)
    gen = FunctionalDataGenerator(
        t,
        mean_func=lambda x: np.sin(x) * 0.5,
        var_func=lambda x: 1.0 + 0.2 * np.cos(x),
    )
    y_list, t_list = gen.generate(n=n, seed=seed)
    return y_list, t_list


def _make_pflm_data(n: int = 60, n_scalar: int = 3, seed: int = 0):
    """Build a full PFLM dataset: 1 functional feature + scalar features + response."""
    y_list, t_list = _generate_functional_data(n, seed=seed)
    rng = np.random.default_rng(seed + 1)
    scalar = rng.standard_normal((n, n_scalar))
    response = rng.standard_normal(n)
    return t_list, y_list, scalar, response


# ---------------------------------------------------------------------------
# Smoke / integration tests
# ---------------------------------------------------------------------------

def test_pflm_gaussian_fit_predict():
    """Basic fit + predict round-trip for Gaussian family."""
    t_list, y_list, scalar, response = _make_pflm_data(n=60)

    model = PartialFunctionalLinearModel(
        family=LinearModelFamily.GAUSSIAN,
        linear_opts=dict(alpha=0.1, l1_ratio=0.5),
    )
    model.fit([t_list], [y_list], scalar, response)

    # Attributes set after fit
    assert model.n_functional_features_in_ == 1
    assert model.n_scalar_features_in_ == scalar.shape[1]
    assert len(model.fpca_models_) == 1
    n_pcs = model.fpca_models_[0].num_pcs_
    assert model.linear_model_.coef_.shape[0] == scalar.shape[1] + n_pcs

    # fitted_values shape
    fv = model.fitted_values()
    assert fv.shape == (60,)

    # predict on same data
    y_pred = model.predict([t_list], [y_list], scalar)
    assert y_pred.shape == (60,)


def test_pflm_gaussian_with_sample_weight():
    """Gaussian PFLM with sample_weight."""
    t_list, y_list, scalar, response = _make_pflm_data(n=60)
    rng = np.random.default_rng(99)
    w = rng.uniform(0.5, 2.0, size=60)

    model = PartialFunctionalLinearModel(
        family=LinearModelFamily.GAUSSIAN,
        linear_opts=dict(alpha=0.1, l1_ratio=0.5),
    )
    model.fit([t_list], [y_list], scalar, response, sample_weight=w)
    assert model.fitted_values().shape == (60,)


def test_pflm_binomial():
    """Binomial family PFLM."""
    t_list, y_list, scalar, _ = _make_pflm_data(n=80)
    rng = np.random.default_rng(7)
    response = rng.integers(0, 2, size=80).astype(np.float64)

    model = PartialFunctionalLinearModel(
        family=LinearModelFamily.BINOMIAL,
        linear_opts=dict(alpha=0.1, l1_ratio=0.5),
    )
    model.fit([t_list], [y_list], scalar, response)
    y_pred = model.predict([t_list], [y_list], scalar)
    assert y_pred.shape == (80,)
    assert np.all((y_pred >= 0) & (y_pred <= 1))


def test_pflm_poisson():
    """Poisson family PFLM."""
    t_list, y_list, scalar, _ = _make_pflm_data(n=80)
    rng = np.random.default_rng(8)
    response = rng.poisson(lam=3, size=80).astype(np.float64)

    model = PartialFunctionalLinearModel(
        family=LinearModelFamily.POISSON,
        linear_opts=dict(alpha=0.1, l1_ratio=0.5),
    )
    model.fit([t_list], [y_list], scalar, response)
    y_pred = model.predict([t_list], [y_list], scalar)
    assert y_pred.shape == (80,)
    assert np.all(y_pred > 0)


def test_pflm_multinomial():
    """Multinomial family PFLM."""
    t_list, y_list, scalar, _ = _make_pflm_data(n=120, n_scalar=3)
    rng = np.random.default_rng(9)
    response = rng.integers(0, 3, size=120).astype(np.float64)

    model = PartialFunctionalLinearModel(
        family=LinearModelFamily.MULTINOMIAL,
        linear_opts=dict(alpha=0.05, l1_ratio=0.5),
    )
    model.fit([t_list], [y_list], scalar, response)
    y_pred = model.predict([t_list], [y_list], scalar)
    assert y_pred.shape == (120, 3)
    assert_allclose(y_pred.sum(axis=1), 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Multiple functional features
# ---------------------------------------------------------------------------

def test_pflm_two_functional_features():
    """Fit with two independent functional features."""
    n = 60
    y1, t1 = _generate_functional_data(n, seed=10)
    y2, t2 = _generate_functional_data(n, seed=20)
    rng = np.random.default_rng(30)
    scalar = rng.standard_normal((n, 2))
    response = rng.standard_normal(n)

    model = PartialFunctionalLinearModel(
        linear_opts=dict(alpha=0.1, l1_ratio=0.5),
    )
    model.fit([t1, t2], [y1, y2], scalar, response)
    assert model.n_functional_features_in_ == 2
    assert len(model.fpca_models_) == 2
    total_features = scalar.shape[1] + sum(f.num_pcs_ for f in model.fpca_models_)
    assert model.linear_model_.coef_.shape[0] == total_features

    y_pred = model.predict([t1, t2], [y1, y2], scalar)
    assert y_pred.shape == (n,)


# ---------------------------------------------------------------------------
# FPCAConfig tests
# ---------------------------------------------------------------------------

def test_pflm_shared_fpca_config():
    """A single FPCAConfig is shared across all functional features."""
    n = 60
    y1, t1 = _generate_functional_data(n, seed=10)
    y2, t2 = _generate_functional_data(n, seed=20)
    rng = np.random.default_rng(30)
    scalar = rng.standard_normal((n, 2))
    response = rng.standard_normal(n)

    cfg = FPCAConfig(num_points_reg_grid=31, fit_params=dict(fve_threshold=0.95))
    model = PartialFunctionalLinearModel(
        linear_opts=dict(alpha=0.1, l1_ratio=0.5),
        fpca_configs=cfg,
    )
    model.fit([t1, t2], [y1, y2], scalar, response)
    assert model.n_functional_features_in_ == 2


def test_pflm_per_feature_fpca_config():
    """Per-feature FPCAConfig list must match the number of functional features."""
    n = 60
    y1, t1 = _generate_functional_data(n, seed=10)
    y2, t2 = _generate_functional_data(n, seed=20)
    rng = np.random.default_rng(30)
    scalar = rng.standard_normal((n, 2))
    response = rng.standard_normal(n)

    cfgs = [
        FPCAConfig(fit_params=dict(method_pcs="IN")),
        FPCAConfig(fit_params=dict(method_pcs="CE", fve_threshold=0.95)),
    ]
    model = PartialFunctionalLinearModel(
        linear_opts=dict(alpha=0.1, l1_ratio=0.5),
        fpca_configs=cfgs,
    )
    model.fit([t1, t2], [y1, y2], scalar, response)
    assert model.n_functional_features_in_ == 2


# ---------------------------------------------------------------------------
# Validation / error tests
# ---------------------------------------------------------------------------

def test_pflm_predict_before_fit_raises():
    """Calling predict before fit should raise NotFittedError."""
    model = PartialFunctionalLinearModel()
    with pytest.raises(NotFittedError):
        model.predict([[]], [[]], np.zeros((1, 2)))


def test_pflm_fitted_values_before_fit_raises():
    """Calling fitted_values before fit should raise NotFittedError."""
    model = PartialFunctionalLinearModel()
    with pytest.raises(NotFittedError):
        model.fitted_values()


def test_pflm_mismatched_functional_time_raises():
    """functional_time length != functional_features length should raise."""
    t_list, y_list, scalar, response = _make_pflm_data(n=60)
    model = PartialFunctionalLinearModel()
    with pytest.raises(ValueError, match="does not match"):
        model.fit([t_list, t_list], [y_list], scalar, response)


def test_pflm_mismatched_fpca_configs_length_raises():
    """fpca_configs list length != functional features count should raise."""
    t_list, y_list, scalar, response = _make_pflm_data(n=60)
    model = PartialFunctionalLinearModel(
        fpca_configs=[FPCAConfig(), FPCAConfig()],
    )
    with pytest.raises(ValueError, match="fpca_configs has"):
        model.fit([t_list], [y_list], scalar, response)


def test_pflm_predict_wrong_num_functional_raises():
    """predict with wrong number of functional features should raise."""
    t_list, y_list, scalar, response = _make_pflm_data(n=60)
    model = PartialFunctionalLinearModel(
        linear_opts=dict(alpha=0.1, l1_ratio=0.5),
    )
    model.fit([t_list], [y_list], scalar, response)
    with pytest.raises(ValueError, match="does not match"):
        model.predict([t_list, t_list], [y_list, y_list], scalar)


def test_pflm_predict_wrong_num_scalar_raises():
    """predict with wrong number of scalar features should raise."""
    t_list, y_list, scalar, response = _make_pflm_data(n=60, n_scalar=3)
    model = PartialFunctionalLinearModel(
        linear_opts=dict(alpha=0.1, l1_ratio=0.5),
    )
    model.fit([t_list], [y_list], scalar, response)
    bad_scalar = np.zeros((60, 5))
    with pytest.raises(ValueError, match="does not match"):
        model.predict([t_list], [y_list], bad_scalar)


# ---------------------------------------------------------------------------
# fitted_values consistency
# ---------------------------------------------------------------------------

def test_pflm_fitted_values_matches_predict():
    """fitted_values() should match predict() on the training data."""
    t_list, y_list, scalar, response = _make_pflm_data(n=60)
    model = PartialFunctionalLinearModel(
        linear_opts=dict(alpha=0.1, l1_ratio=0.5),
    )
    model.fit([t_list], [y_list], scalar, response)
    fv = model.fitted_values()
    pred = model.predict([t_list], [y_list], scalar)
    # predict recomputes FPCA scores, so allow small tolerance
    assert_allclose(fv, pred, rtol=1e-4, atol=1e-6)
