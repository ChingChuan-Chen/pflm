import numpy as np
import pytest
from pflm import FunctionalDataGenerator


def test_functional_data_generator():
    n = 20
    t = np.linspace(0, 10, 5)
    fdg = FunctionalDataGenerator(t, lambda x: np.sin(0.4 * x), lambda x: 2.0 * np.log(x + 5.5))
    true_y = fdg.generate(n, 100)
    y = FunctionalDataGenerator.make_missing(true_y, 2, 101)

    assert y.shape == (n, 5)
    assert np.isnan(true_y).sum() == 0  # Ensure no NaNs in the original data
    assert np.isnan(y).sum() == 40  # 2 missing values per sample
    assert np.all(np.isfinite(true_y))  # Ensure no NaNs in the original data
    assert fdg.get_num_fpc() == 5
    assert fdg.get_fpca_phi().shape == (5, 5)


def test_fdg_variation_prop_thresh_invalid():
    t = np.linspace(0, 1, 5)
    with pytest.raises(ValueError):
        FunctionalDataGenerator(t, np.sin, np.abs, variation_prop_thresh=1.0)
    with pytest.raises(ValueError):
        FunctionalDataGenerator(t, np.sin, np.abs, variation_prop_thresh=0.0)
    with pytest.raises(ValueError):
        FunctionalDataGenerator(t, np.sin, np.abs, variation_prop_thresh=-0.1)


def test_fdg_make_missing_invalid():
    y = np.ones((3, 4))
    # missing_number < 1
    with pytest.raises(ValueError):
        FunctionalDataGenerator.make_missing(y, 0)
    # missing_number >= ncol
    with pytest.raises(ValueError):
        FunctionalDataGenerator.make_missing(y, 4)
    # y 含 NaN
    y_nan = y.copy()
    y_nan[0, 0] = np.nan
    with pytest.raises(ValueError):
        FunctionalDataGenerator.make_missing(y_nan, 1)


def test_fdg_make_missing_seed_none():
    y = np.ones((2, 3))
    out1 = FunctionalDataGenerator.make_missing(y, 1)
    out2 = FunctionalDataGenerator.make_missing(y, 1)
    assert out1.shape == y.shape
    assert out2.shape == y.shape
    assert np.all(np.isnan(out1).sum(axis=1) == 1)
    assert np.all(np.isnan(out2).sum(axis=1) == 1)


def test_fdg_generate_seed_none_and_custom_corr():
    t = np.linspace(0, 1, 4)
    fdg = FunctionalDataGenerator(t, lambda x: np.zeros_like(x), lambda x: np.ones_like(x), corr_func=lambda x: np.exp(-x))
    y = fdg.generate(3)
    assert y.shape == (3, 4)
    # lazy property
    assert fdg.get_num_fpc() > 0
    assert fdg.get_fpca_phi().shape[1] == fdg.get_num_fpc()


def test_fdg_generate_and_lazy():
    t = np.linspace(0, 1, 3)
    fdg = FunctionalDataGenerator(t, lambda x: x, lambda x: np.ones_like(x))
    # call get_fpca_phi before generate
    phi = fdg.get_fpca_phi()
    y = fdg.generate(2)
    assert y.shape == (2, 3)


def test_fdg_generate_and_lazy_2():
    t = np.linspace(0, 1, 3)
    fdg = FunctionalDataGenerator(t, lambda x: x, lambda x: np.ones_like(x))
    # call get_num_fpc before generate
    n_fpc = fdg.get_num_fpc()
    y2 = fdg.generate(2)
    assert y2.shape == (2, 3)
