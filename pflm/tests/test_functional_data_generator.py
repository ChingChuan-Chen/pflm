import numpy as np
import pytest
from numpy.testing import assert_allclose

from pflm.functional_data_generator import FunctionalDataGenerator


def test_functional_data_generator_happy_path():
    n = 20
    nt = 7
    num_missing = 2
    t = np.linspace(0, 10, nt)
    fdg = FunctionalDataGenerator(t, lambda x: np.sin(0.4 * x), lambda x: 2.0 * np.log(x + 5.5))
    y_origin, t_origin = fdg.generate(n, 100)
    y, t = FunctionalDataGenerator.make_missing(y_origin, t_origin, num_missing, 101)

    assert len(y) == n
    assert len(t) == n
    for i in range(n):
        assert len(y_origin[i]) == nt
        assert len(t_origin[i]) == nt
        assert np.isnan(y_origin[i]).sum() == 0
        assert np.isnan(t_origin[i]).sum() == 0
        assert len(y[i]) == nt - num_missing
        assert len(t[i]) == nt - num_missing
        assert np.isnan(y[i]).sum() == 0
        assert np.isnan(t[i]).sum() == 0
    assert fdg.get_num_pcs() == nt
    assert fdg.get_fpca_phi().shape == (nt, nt)


@pytest.mark.parametrize("num_pcs", [1, 2, 3, 4, 5, 6, 7])
def test_functional_data_generator_specific_num_pcs(num_pcs):
    n = 20
    nt = 7
    num_missing = 2
    fdg = FunctionalDataGenerator(np.linspace(0, 10, nt), lambda x: np.sin(0.4 * x), lambda x: 2.0 * np.log(x + 5.5), num_pcs=num_pcs)
    y_origin, t_origin = fdg.generate(n, 100)
    y, t = FunctionalDataGenerator.make_missing(y_origin, t_origin, num_missing, 101)

    assert len(y) == n
    assert len(t) == n
    for i in range(n):
        assert len(y_origin[i]) == nt
        assert len(t_origin[i]) == nt
        assert np.isnan(y_origin[i]).sum() == 0
        assert np.isnan(t_origin[i]).sum() == 0
        assert len(y[i]) == nt - num_missing
        assert len(t[i]) == nt - num_missing
        assert np.isnan(y[i]).sum() == 0
        assert np.isnan(t[i]).sum() == 0
    assert fdg.get_num_pcs() == num_pcs
    assert fdg.get_fpca_phi().shape == (nt, num_pcs)


@pytest.mark.parametrize("num_pcs", [0, -1, 8, 10])
def test_fdg_num_pcs_invalid(num_pcs):
    t = np.linspace(0, 1, 5)
    with pytest.raises(ValueError, match="num_pcs must be a positive integer between 1 and length of t."):
        FunctionalDataGenerator(t, np.sin, np.abs, num_pcs=num_pcs)


@pytest.mark.parametrize("num_pcs", [float("nan"), "string", [2.5], {"a": 2.5}])
def test_fdg_num_pcs_invalid_types(num_pcs):
    t = np.linspace(0, 1, 5)
    with pytest.raises(ValueError, match="num_pcs must be an integer."):
        FunctionalDataGenerator(t, np.sin, np.abs, num_pcs=num_pcs)


def test_fdg_variation_prop_thresh_invalid():
    t = np.linspace(0, 1, 5)
    with pytest.raises(ValueError):
        FunctionalDataGenerator(t, np.sin, np.abs, variation_prop_thresh=1.0)
    with pytest.raises(ValueError):
        FunctionalDataGenerator(t, np.sin, np.abs, variation_prop_thresh=0.0)
    with pytest.raises(ValueError):
        FunctionalDataGenerator(t, np.sin, np.abs, variation_prop_thresh=-0.1)


def __build_basic_data():
    n = 20
    nt = 7
    y = []
    t = []
    for i in range(n):
        y.append(np.random.normal(size=nt))
        t.append(np.linspace(0, 1, nt))
    return y, t


def test_fdg_make_missing_invalid():
    y, t = __build_basic_data()
    # missing_number < 1
    with pytest.raises(ValueError, match="missing_number must be between 1 and the length of"):
        FunctionalDataGenerator.make_missing(y, t, 0)
    # missing_number >= number of columns
    with pytest.raises(ValueError, match="missing_number must be between 1 and the length of"):
        FunctionalDataGenerator.make_missing(y, t, 8)
    # y contains NaN
    y_nan = y.copy()
    y_nan[0][0] = np.nan
    with pytest.raises(ValueError, match="y contains NaN values"):
        FunctionalDataGenerator.make_missing(y_nan, t, 1)


def test_fdg_generate_and_lazy_phi():
    fdg = FunctionalDataGenerator(np.linspace(0, 1, 3), lambda x: x, lambda x: np.ones_like(x))
    # call get_fpca_phi before generate
    phi = fdg.get_fpca_phi()
    y, t = fdg.generate(2)
    assert len(y) == 2
    assert len(t) == 2
    assert y[0].shape == (3,)
    assert t[0].shape == (3,)


def test_fdg_generate_and_lazy_num_pcs():
    fdg = FunctionalDataGenerator(np.linspace(0, 1, 3), lambda x: x, lambda x: np.ones_like(x))
    # call get_num_fpc before generate
    num_pcs = fdg.get_num_pcs()
    y, t = fdg.generate(2)
    assert len(y) == 2
    assert len(t) == 2
    assert y[0].shape == (3,)
    assert t[0].shape == (3,)


def test_fdg_generate_same_seed():
    y, t = __build_basic_data()
    fdg = FunctionalDataGenerator(t[0], lambda x: np.sin(0.4 * x), lambda x: 2.0 * np.log(x + 5.5))
    y1, t1 = fdg.generate(2, seed=123)
    y2, t2 = fdg.generate(2, seed=123)
    for i in range(2):
        assert_allclose(y1[i], y2[i])
        assert_allclose(t1[i], t2[i])
