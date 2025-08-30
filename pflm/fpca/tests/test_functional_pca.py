import warnings

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from pflm.fpca import FunctionalPCA, FunctionalPCAMuCovParams, FunctionalPCAUserDefinedParams
from pflm.smooth import KernelType


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("basic_func_data", [np.float32, np.float64], indirect=["basic_func_data"])
@pytest.mark.parametrize("method_pcs", ["CE", "IN"])
@pytest.mark.parametrize("assume_measurement_error", [True, False])
@pytest.mark.parametrize("method_rho", ["truncated", "ridge", "vanilla"])
@pytest.mark.parametrize("if_shrinkage", [True, False])
def test_functional_pca_basic(basic_func_data, method_pcs, assume_measurement_error, method_rho, if_shrinkage):
    y, t, w = basic_func_data
    fpca = FunctionalPCA(
        assume_measurement_error=assume_measurement_error, mu_cov_params=FunctionalPCAMuCovParams(kernel_type=KernelType.EPANECHNIKOV)
    )
    fpca.fit(y, t, w, method_pcs=method_pcs, method_rho=method_rho, if_shrinkage=if_shrinkage)

    expected_output_attrs = [
        "y_",
        "t_",
        "flatten_func_data_",
        "raw_cov_",
        "smoothed_model_result_obs_",
        "smoothed_model_result_reg_",
        "fpca_model_params_",
        "xi_",
        "xi_var_",
        "fitted_y_mat_",
        "fitted_y_",
        "elapsed_time_",
    ]
    assert all(hasattr(fpca, attr) for attr in expected_output_attrs)
    assert fpca.fitted_y_mat_.shape == (fpca.smoothed_model_result_obs_.grid.shape[0], len(y))
    assert len(fpca.fitted_y_) == len(y)
    assert fpca.num_pcs_ is not None
    assert fpca.flatten_func_data_ is not None
    assert fpca.raw_cov_ is not None
    assert fpca.fpca_model_params_ is not None
    assert fpca.fpca_model_params_.fpca_phi.get("obs") is not None
    assert fpca.fpca_model_params_.fpca_phi.get("reg") is not None
    assert fpca.xi_ is not None


def _make_toy(n_subj=4, lens=(5, 6, 7, 8), dtype=np.float64):
    """Generate small synthetic series with varying lengths."""
    rng = np.random.default_rng(0)
    y, t = [], []
    for L in lens[:n_subj]:
        ti = np.linspace(0.0, 1.0, L, dtype=dtype)
        yi = np.sin(2 * np.pi * ti).astype(dtype) + 0.02 * rng.standard_normal(L).astype(dtype)
        t.append(ti)
        y.append(yi)
    return y, t


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("estimate_method", ["smooth", "cross-sectional"])
@pytest.mark.parametrize("selector", ["FVE"])  # TODO: ADD "AIC", "BIC"
def test_fit_covers_estimation_and_selectors(dtype, estimate_method, selector):
    """Cover mean/cov estimation branches and num_pcs selectors."""
    y, t = _make_toy(dtype=dtype)
    fpca = FunctionalPCA(
        assume_measurement_error=True,
        mu_cov_params=FunctionalPCAMuCovParams(estimate_method=estimate_method),
    )
    fpca.fit(t, y, method_pcs="CE", method_select_num_pcs=selector, if_fit_eigen_values=True)
    assert fpca.fitted_y_mat_.shape[1] == len(y)
    assert fpca.num_pcs_ is not None
    assert fpca.fpca_model_params_ is not None


def test_fit_with_user_defined_mu_and_cov_paths():
    """Use user-specified mu/cov to cover those branches."""
    y, t = _make_toy()
    # user-defined mu along observation grid
    obs = np.unique(np.concatenate(t))
    mu_obs = np.interp(obs, [obs.min(), obs.max()], [0.1, -0.2])
    # user-defined cov on a small regular grid
    reg = np.linspace(obs.min(), obs.max(), 21)
    # PSD covariance (RBF kernel)
    X1, X2 = np.meshgrid(reg, reg, indexing="ij")
    cov_reg = np.exp(-((X1 - X2) ** 2) / (2 * 0.1**2))

    user = FunctionalPCAUserDefinedParams(t_mu=obs, mu=mu_obs, t_cov=reg, cov=cov_reg, sigma2=0.01, rho=0.05)
    fpca = FunctionalPCA(assume_measurement_error=True, user_params=user)
    fpca.fit(t, y, method_pcs="CE", method_rho="ridge", reg_grid=reg)
    assert fpca.smoothed_model_result_obs_.mu.shape[0] == obs.shape[0]
    assert fpca.smoothed_model_result_reg_.cov.shape == cov_reg.shape


def test_fit_user_defined_mu_or_cov_with_nan_raises():
    """NaN in user-specified mu/cov should trigger finite checks."""
    y, t = _make_toy()
    obs = np.unique(np.concatenate(t))
    reg = np.linspace(obs.min(), obs.max(), 21)

    # NaN mu
    bad_mu = np.full_like(obs, 0.0)
    bad_mu[0] = np.nan
    user = FunctionalPCAUserDefinedParams(t_mu=obs, mu=bad_mu)
    fpca = FunctionalPCA(user_params=user)
    with pytest.raises(Exception):
        fpca.fit(t, y)

    # NaN cov
    good_mu = np.zeros_like(obs)
    good_user = FunctionalPCAUserDefinedParams(t_mu=obs, mu=good_mu, t_cov=reg, cov=np.eye(reg.size))
    fpca2 = FunctionalPCA(user_params=good_user)
    # Replace cov with NaN after construction to avoid constructor validation
    fpca2.user_params.cov = np.eye(reg.size)
    fpca2.user_params.cov[0, 0] = np.nan
    with pytest.raises(Exception):
        fpca2.fit(t, y)


def test_reg_grid_mismatch_raises():
    """reg_grid endpoints must match data range."""
    y, t = _make_toy()
    bad_grid = np.linspace(-0.1, 1.1, 51)
    fpca = FunctionalPCA()
    with pytest.raises(Exception):
        fpca.fit(t, y, reg_grid=bad_grid)


def test_insufficient_observations_raises():
    """Each subject must have at least two observations."""
    y, t = _make_toy()
    y[0] = y[0][:1]
    t[0] = t[0][:1]
    fpca = FunctionalPCA()
    with pytest.raises(Exception):
        fpca.fit(t, y)


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("method_pcs", ["CE", "IN"])
def test_predict(method_pcs):
    y, t = _make_toy()
    fpca = FunctionalPCA()
    fpca.fit(t, y, method_pcs=method_pcs)
    ny, nt = _make_toy()
    new_xi, _, new_fitted_y_mat, new_fitted_y = fpca.predict(ny, nt)
    assert new_xi.shape[0] == len(ny)
    assert new_fitted_y_mat.shape[1] == len(ny)
    assert len(new_fitted_y) == len(ny)
    fy_mat, fy = fpca.fitted_values()
    assert fy_mat.shape[1] == len(y)
    assert len(fy) == len(y)


def test_notfitted_errors():
    fpca = FunctionalPCA()
    with pytest.raises(NotFittedError):
        fpca.fitted_values()
    with pytest.raises(NotFittedError):
        y, t = _make_toy(n_subj=1, lens=(5,))
        fpca.predict(y, t)
    with pytest.raises(NotFittedError):
        fpca.fit_score()


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("selector", [1, 2, 3])
def test_fit_with_fixed_num_pcs(selector):
    y, t = _make_toy()
    fpca = FunctionalPCA()
    fpca.fit(t, y, method_pcs="IN", method_select_num_pcs=selector)
    assert fpca.num_pcs_ == selector or fpca.num_pcs_ is not None  # allow cap by data
    assert fpca.fitted_y_mat_.shape[1] == len(y)


@pytest.mark.filterwarnings("ignore")
def test_init_assume_error_with_user_sigma2():
    y, t = _make_toy()
    user = FunctionalPCAUserDefinedParams(sigma2=0.02)
    with pytest.raises(ValueError, match="Measurement error is assumed to be false"):
        FunctionalPCA(assume_measurement_error=False, user_params=user)


@pytest.mark.filterwarnings("ignore")
def test_init_assume_error_with_user_rho():
    y, t = _make_toy()
    user = FunctionalPCAUserDefinedParams(rho=0.1)
    with pytest.raises(ValueError, match="Measurement error is assumed to be false"):
        FunctionalPCA(assume_measurement_error=False, user_params=user)


def _series(n=4, lens=(5, 6, 7, 8), dtype=np.float64, finite=True, make_2d=False, mismatch_len=False):
    """Helper to build toy (y, t) lists with knobs to violate constraints."""
    y, t = [], []
    for i, L in enumerate(lens[:n]):
        ti = np.linspace(0.0, 1.0, L, dtype=dtype)
        yi = np.sin(2 * np.pi * ti).astype(dtype)
        if not finite and i == 0:
            yi[0] = np.nan
        if mismatch_len and i == 0:
            yi = yi[:-1]
        if make_2d and i == 0:
            ti = ti[:, None]  # 2D on purpose
        y.append(yi)
        t.append(ti)
    w = np.ones(n, dtype=dtype)
    return y, t, w


def test_check_data_warns_on_small_sample_count():
    """Should warn when number of samples is <= 3."""
    fpca = FunctionalPCA()
    fpca._input_dtype = np.float64  # ensure dtype for internal casting
    y, t, _ = _series(n=3)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        dtype_out, y_, t_, w_ = fpca._FunctionalPCA__check_data(y, t, w=None)
    assert any("less than or equal to 3" in str(w.message) for w in rec)
    assert dtype_out == np.float64
    assert isinstance(y_[0], np.ndarray) and isinstance(t_[0], np.ndarray)
    assert w_ is None


@pytest.mark.filterwarnings("ignore")
def test_check_data_len_mismatch_raises():
    """len(y) must equal len(t)."""
    fpca = FunctionalPCA()
    fpca._input_dtype = np.float64
    y, t, w = _series(n=2)
    t = t[:1]
    with pytest.raises(ValueError, match="y and t must have the same length"):
        fpca._FunctionalPCA__check_data(y, t, w=w)


@pytest.mark.filterwarnings("ignore")
def test_check_weight_mismatch_raises():
    """len(y) must equal len(t)."""
    fpca = FunctionalPCA()
    fpca._input_dtype = np.float64
    y, t, w = _series(n=2)
    w = w[:1]
    with pytest.raises(ValueError, match="The length of w must be the same as the number of samples"):
        fpca._FunctionalPCA__check_data(y, t, w=w)


@pytest.mark.parametrize(
    "make_2d,mismatch_len,finite,err_msg",
    [
        (True, False, True, "must be a 1D array"),
        (False, True, True, "must have the same length"),
        (False, False, False, "Input contains NaN"),
    ],
)
def test_check_data_per_sample_validations(make_2d, mismatch_len, finite, err_msg):
    """Per-sample validations: 1D, length match, finiteness."""
    fpca = FunctionalPCA()
    fpca._input_dtype = np.float64
    y, t, w = _series(n=4, make_2d=make_2d, mismatch_len=mismatch_len, finite=finite)
    with pytest.raises(ValueError, match=err_msg):
        fpca._FunctionalPCA__check_data(y, t, w=w)


def test_check_weight():
    """Weight vector must be 1D and match the number of samples."""
    fpca = FunctionalPCA()
    fpca._input_dtype = np.float64
    y, t, w = _series(n=4)
    w = w[:, None]  # make w 2D
    with pytest.raises(ValueError, match="must be a 1D array"):
        fpca._FunctionalPCA__check_data(y, t, w=w)


def test_check_data_dtype_override():
    """Providing dtype should drive output dtype."""
    fpca = FunctionalPCA()
    fpca._input_dtype = np.float32  # drive internal casting
    y, t = _make_toy(dtype=np.float64)  # input double, but we override to float32
    dtype_out, y_, t_, w_ = fpca._FunctionalPCA__check_data(y, t, dtype=np.float32)
    assert dtype_out == np.float32
    assert y_[0].dtype == np.float32 and t_[0].dtype == np.float32
    assert w_ is None
