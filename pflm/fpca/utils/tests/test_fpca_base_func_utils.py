import math

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pflm.fpca.utils import get_eigen_analysis_results, get_fpca_phi, select_num_pcs_fve, select_num_pcs_ic
from pflm.fpca.utils.log_lik import get_log_likelihood_f32, get_log_likelihood_f64
from pflm.utils import trapz


@pytest.mark.parametrize("dtype, order", [(np.float32, "C"), (np.float64, "C"), (np.float32, "F"), (np.float64, "F")])
def test_get_eigen_analysis_results_happy_path(dtype, order):
    A = np.array([[1.1, 1, 1], [1, 1.1, 1], [1, 1, 1.1]], dtype=dtype, order=order)
    eig_lambda, eig_vector = get_eigen_analysis_results(A)
    expected_eig_lambda = np.array([3.1, 0.1, 0.1], dtype=dtype)
    expected_first_eig_vector = np.array([1.0 / math.sqrt(3)] * 3, dtype=dtype)
    assert_allclose(eig_lambda, expected_eig_lambda, rtol=1e-5, atol=1e-5)
    if np.all(eig_vector[:, 0] >= 0):
        assert_allclose(eig_vector[:, 0], expected_first_eig_vector, rtol=1e-5, atol=1e-5)
    else:
        assert_allclose(eig_vector[:, 0], -expected_first_eig_vector, rtol=1e-5, atol=1e-5)


def _make_spd_matrix(n: int, dtype=np.float64):
    A = np.arange(1, n * n + 1, dtype=dtype).reshape(n, n)
    M = A @ A.T / n
    np.fill_diagonal(M, np.diag(M) + np.linspace(0, n - 1, n))
    return M.astype(dtype, copy=False)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_get_eigen_analysis_results_spd(dtype):
    reg_cov = _make_spd_matrix(3, dtype=dtype)
    eig_lambda, eig_vector = get_eigen_analysis_results(reg_cov, is_upper_triangular=False)
    assert eig_lambda.ndim == 1
    # Eigenvalues should be strictly decreasing (allow tiny numerical jitter)
    assert np.all(np.diff(eig_lambda) <= 1e-10)
    # Reconstruct and compare
    recon = eig_vector @ np.diag(eig_lambda) @ eig_vector.T
    assert_allclose(recon, reg_cov, rtol=1e-5, atol=1e-5)


def test_get_eigen_analysis_results_upper_triangular_path():
    reg_cov = _make_spd_matrix(4)
    lam_u, vec_u = get_eigen_analysis_results(reg_cov, is_upper_triangular=True)
    lam_l, vec_l = get_eigen_analysis_results(reg_cov, is_upper_triangular=False)
    assert_allclose(lam_u, lam_l)
    # Columns may differ by sign
    for i in range(vec_u.shape[1]):
        assert_allclose(np.abs(vec_u[:, i]), np.abs(vec_l[:, i]), rtol=1e-5, atol=1e-5)


def test_get_eigen_analysis_results_zero_matrix_all_filtered():
    Z = np.zeros((3, 3), dtype=np.float64)
    eig_lambda, eig_vector = get_eigen_analysis_results(Z)
    # All eigenvalues zero -> filtered out
    assert eig_lambda.size == 0
    assert eig_vector.shape == (3, 0)


def test_get_eigen_analysis_results_rank1_matrix():
    ones_matrix = np.ones((3, 3), dtype=np.float64)
    with pytest.warns(UserWarning, match="Eigenvalues contain NaN or negative values."):
        eig_lambda, eig_vector = get_eigen_analysis_results(ones_matrix)
    assert eig_lambda.size == 1
    assert eig_vector.shape == (3, 1)
    assert_allclose(np.abs(eig_vector[:, 0]), np.array([1, 1, 1]) / np.sqrt(3))


def test_get_eigen_analysis_results_lapack_fail(monkeypatch):
    import pflm.fpca.utils.fpca_base_func_utils as fh

    def fake_syevd(jobz, uplo, eig_vector, eig_lambda, n):
        # mimic LAPACK failure by returning non‑zero info
        return 1

    # patch both float64 / float32 variants
    monkeypatch.setattr(fh, "_syevd_memview_f64", fake_syevd, raising=True)
    monkeypatch.setattr(fh, "_syevd_memview_f32", fake_syevd, raising=True)

    A64 = np.eye(3, dtype=np.float64)
    with pytest.warns(UserWarning, match="LAPACK syevd failed"):
        lam, vec = fh.get_eigen_analysis_results(A64)
    assert lam is None and vec is None

    A32 = np.eye(3, dtype=np.float32)
    with pytest.warns(UserWarning, match="LAPACK syevd failed"):
        lam2, vec2 = fh.get_eigen_analysis_results(A32)
    assert lam2 is None and vec2 is None


def test_select_num_pcs_fve_basic():
    eig_lambda = np.array([4.0, 3.0, 2.0, 1.0])
    cumulative_fve, num_pcs = select_num_pcs_fve(eig_lambda, fve_threshold=0.75)
    assert num_pcs == 3
    assert_allclose(cumulative_fve, np.array([0.4, 0.7, 0.9, 1.0]))


def test_select_num_pcs_fve_threshold_and_max_components():
    eig_lambda = np.array([5.0, 1.0, 1.0, 1.0])
    _, n_low = select_num_pcs_fve(eig_lambda, fve_threshold=0.2)
    _, n_high = select_num_pcs_fve(eig_lambda, fve_threshold=0.999)
    _, n_trunc = select_num_pcs_fve(eig_lambda, fve_threshold=0.9, max_components=2)
    assert (n_low, n_high, n_trunc) == (1, 4, 2)


@pytest.mark.parametrize("criterion", ["AIC", "BIC"])
def test_select_num_pcs_ic_returns_finite_and_valid_k(criterion):
    rng = np.random.default_rng(0)
    obs_grid = np.linspace(0.0, 1.0, 8)
    reg_grid = np.linspace(0.0, 1.0, 21)

    t = [obs_grid.copy() for _ in range(6)]
    y = []
    for i in range(6):
        base = np.sin(2 * np.pi * obs_grid) + 0.2 * np.cos(np.pi * obs_grid)
        y.append((base + 0.03 * rng.standard_normal(obs_grid.size) + 0.01 * i).astype(np.float64))

    y_mat = np.vstack(y)
    mu_obs = np.mean(y_mat, axis=0)
    cov_obs = np.cov(y_mat, rowvar=False)
    cov_row_interp = np.vstack([np.interp(reg_grid, obs_grid, row) for row in cov_obs])
    cov_reg = np.vstack([np.interp(reg_grid, obs_grid, col) for col in cov_row_interp.T]).T
    cov_reg = (cov_reg + cov_reg.T) / 2.0
    lam_tmp, vec_tmp = np.linalg.eigh(cov_reg)
    cov_reg = vec_tmp @ np.diag(np.clip(lam_tmp, 1e-10, None)) @ vec_tmp.T

    eig_lambda, eig_vector = get_eigen_analysis_results(cov_reg)
    ic_values, selected_k = select_num_pcs_ic(
        criterion,
        y,
        t,
        obs_grid,
        reg_grid,
        np.interp(reg_grid, obs_grid, mu_obs),
        mu_obs,
        eig_lambda,
        eig_vector,
        max_components=min(5, eig_lambda.size),
        measurement_error_variance=1e-4,
        rho=None,
    )

    assert ic_values.ndim == 1
    assert ic_values.size >= selected_k
    assert np.all(np.isfinite(ic_values))
    assert 1 <= selected_k <= min(5, eig_lambda.size)


def test_select_num_pcs_ic_invalid_criterion_raises():
    with pytest.raises(ValueError, match="criterion must be either 'AIC' or 'BIC'"):
        select_num_pcs_ic(
            "XXX",
            [np.array([0.0, 1.0], dtype=np.float64)],
            [np.array([0.0, 1.0], dtype=np.float64)],
            np.array([0.0, 1.0], dtype=np.float64),
            np.array([0.0, 0.5, 1.0], dtype=np.float64),
            np.array([0.0, 0.0, 0.0], dtype=np.float64),
            np.array([0.0, 0.0], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
            np.array([[1.0], [0.0], [0.0]], dtype=np.float64),
        )


def test_select_num_pcs_ic_no_components_raises():
    with pytest.raises(ValueError, match="No available principal components"):
        select_num_pcs_ic(
            "AIC",
            [np.array([0.0, 1.0], dtype=np.float64)],
            [np.array([0.0, 1.0], dtype=np.float64)],
            np.array([0.0, 1.0], dtype=np.float64),
            np.array([0.0, 0.5, 1.0], dtype=np.float64),
            np.array([0.0, 0.0, 0.0], dtype=np.float64),
            np.array([0.0, 0.0], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.empty((3, 0), dtype=np.float64),
            max_components=0,
        )


def test_select_num_pcs_ic_handles_rho_and_negative_sigma2_branches():
    obs_grid = np.linspace(0.0, 1.0, 4)
    reg_grid = np.linspace(0.0, 1.0, 5)
    y = [np.array([0.0, 0.1, -0.1, 0.0], dtype=np.float64) for _ in range(3)]
    t = [obs_grid.copy() for _ in range(3)]
    reg_mu = np.zeros_like(reg_grid)
    mu_obs = np.zeros_like(obs_grid)
    eig_lambda = np.array([1.0, 0.2], dtype=np.float64)
    eig_vector = np.column_stack([np.ones(reg_grid.size), np.linspace(-1, 1, reg_grid.size)]).astype(np.float64)

    # Covers branch: sigma2 <= rho -> sigma2 = rho
    ic_values_rho, k_rho = select_num_pcs_ic(
        "BIC",
        y,
        t,
        obs_grid,
        reg_grid,
        reg_mu,
        mu_obs,
        eig_lambda,
        eig_vector,
        max_components=2,
        measurement_error_variance=0.0,
        rho=0.05,
    )
    assert np.all(np.isfinite(ic_values_rho))
    assert 1 <= k_rho <= 2

    # Covers branch: sigma2 < 0 -> sigma2 = 0 and then fallback jitter path
    ic_values_neg, k_neg = select_num_pcs_ic(
        "AIC",
        y,
        t,
        obs_grid,
        reg_grid,
        reg_mu,
        mu_obs,
        eig_lambda,
        eig_vector,
        max_components=2,
        measurement_error_variance=-1.0,
        rho=None,
    )
    assert np.all(np.isfinite(ic_values_neg))
    assert 1 <= k_neg <= 2


@pytest.mark.parametrize("num_pcs", [1, 2, 3])
def test_get_fpca_phi_shapes_and_scaling(num_pcs):
    reg_cov = _make_spd_matrix(3)
    eig_lambda, eig_vector = get_eigen_analysis_results(reg_cov)
    reg_grid = np.array([0.0, 0.5, 1.0])
    reg_mu = np.array([0.2, -0.1, 0.3])
    fpca_lambda, fpca_phi = get_fpca_phi(num_pcs, reg_grid, reg_mu, eig_lambda, eig_vector)
    assert fpca_lambda.shape == (num_pcs,)
    assert fpca_phi.shape == (reg_grid.size, num_pcs)
    # Lambda scaling by grid spacing (0.5)
    assert_allclose(fpca_lambda, eig_lambda[:num_pcs] * (reg_grid[1] - reg_grid[0]))
    # L2 norm ~ 1 for each component
    energy = trapz((fpca_phi**2).T, reg_grid)
    assert_allclose(energy, np.ones_like(energy), rtol=1e-5, atol=1e-5)
    # Sign alignment: inner product with mean >= 0
    signs = np.sum(fpca_phi * reg_mu.reshape(-1, 1), axis=0)
    assert np.all(signs >= -1e-12)


def test_get_fpca_phi_sign_consistency():
    nt = 4
    reg_cov = _make_spd_matrix(nt)
    eig_lambda, eig_vector = get_eigen_analysis_results(reg_cov)
    reg_grid = np.linspace(0, 1, nt)
    reg_mu = np.linspace(1, 2, nt)
    fpca_lambda, fpca_phi = get_fpca_phi(2, reg_grid, reg_mu, eig_lambda, eig_vector)
    # Inner products should be non-negative
    assert np.all(fpca_phi.T @ reg_mu >= -1e-12)
    assert fpca_lambda.shape == (2,)


def _reference_loglik(yy: np.ndarray, tid: np.ndarray, mu: np.ndarray, sigma_y: np.ndarray, sid_cnt: np.ndarray) -> float:
    loglik = 0.0
    start = 0
    for cnt in sid_cnt:
        end = start + int(cnt)
        tid_i = tid[start:end]
        resid = yy[start:end] - mu[tid_i]
        sigma_i = sigma_y[np.ix_(tid_i, tid_i)]
        sign, logdet = np.linalg.slogdet(sigma_i)
        assert sign > 0
        solved = np.linalg.solve(sigma_i, resid)
        loglik += float(logdet + resid @ solved)
        start = end
    return float(loglik)


@pytest.mark.parametrize("dtype, cy_func", [(np.float64, get_log_likelihood_f64), (np.float32, get_log_likelihood_f32)])
def test_get_log_likelihood_matches_numpy_reference(dtype, cy_func):
    yy = np.array([0.1, -0.2, 0.3, -0.1, 0.2], dtype=dtype)
    tt = np.array([0.0, 0.5, 0.0, 0.5, 1.0], dtype=dtype)
    tid = np.array([0, 2, 1, 3, 4], dtype=np.int64)
    mu = np.array([0.0, 0.05, -0.05, 0.02, -0.01], dtype=dtype)

    base = np.array(
        [
            [1.2, 0.1, 0.05, 0.02, 0.01],
            [0.1, 1.1, 0.03, 0.04, 0.02],
            [0.05, 0.03, 1.3, 0.06, 0.03],
            [0.02, 0.04, 0.06, 1.25, 0.05],
            [0.01, 0.02, 0.03, 0.05, 1.15],
        ],
        dtype=np.float64,
    )
    sigma_y = np.array((base + base.T) / 2.0, dtype=dtype)

    fpca_lambda = np.array([1.0, 0.5], dtype=dtype)
    lambda_phi = np.array(
        [
            [0.2, 0.1],
            [0.3, -0.1],
            [0.1, 0.2],
            [-0.2, 0.1],
            [0.05, -0.15],
        ],
        dtype=dtype,
    )
    unique_sid = np.array([0, 1], dtype=np.int64)
    sid_cnt = np.array([2, 3], dtype=np.int64)

    got = cy_func(yy, tt, tid, mu, sigma_y, fpca_lambda, lambda_phi, unique_sid, sid_cnt)
    expected = _reference_loglik(yy.astype(np.float64), tid, mu.astype(np.float64), sigma_y.astype(np.float64), sid_cnt)
    tol = 1e-10 if dtype == np.float64 else 1e-4
    assert_allclose(got, expected, rtol=tol, atol=tol)


@pytest.mark.parametrize("dtype, cy_func", [(np.float64, get_log_likelihood_f64), (np.float32, get_log_likelihood_f32)])
def test_get_log_likelihood_returns_nan_for_singular(dtype, cy_func):
    yy = np.array([0.1, -0.2], dtype=dtype)
    tt = np.array([0.0, 1.0], dtype=dtype)
    tid = np.array([0, 1], dtype=np.int64)
    mu = np.zeros(2, dtype=dtype)
    sigma_y = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=dtype)  # singular
    fpca_lambda = np.array([1.0], dtype=dtype)
    lambda_phi = np.array([[0.1], [0.2]], dtype=dtype)
    unique_sid = np.array([0], dtype=np.int64)
    sid_cnt = np.array([2], dtype=np.int64)

    got = cy_func(yy, tt, tid, mu, sigma_y, fpca_lambda, lambda_phi, unique_sid, sid_cnt)
    assert np.isnan(got)


@pytest.mark.parametrize("dtype, cy_func", [(np.float64, get_log_likelihood_f64), (np.float32, get_log_likelihood_f32)])
def test_get_log_likelihood_small_sample_no_inf(dtype, cy_func):
    # Two subjects, one observation each (small-sample corner case).
    yy = np.array([0.2, -0.1], dtype=dtype)
    tt = np.array([0.0, 1.0], dtype=dtype)
    tid = np.array([0, 1], dtype=np.int64)
    mu = np.array([0.0, 0.0], dtype=dtype)
    sigma_y = np.array([[0.8, 0.0], [0.0, 0.6]], dtype=dtype)
    fpca_lambda = np.array([0.5], dtype=dtype)
    lambda_phi = np.array([[0.2], [-0.1]], dtype=dtype)
    unique_sid = np.array([0, 1], dtype=np.int64)
    sid_cnt = np.array([1, 1], dtype=np.int64)

    got = cy_func(yy, tt, tid, mu, sigma_y, fpca_lambda, lambda_phi, unique_sid, sid_cnt)
    assert np.isfinite(got)
    assert not np.isinf(got)


@pytest.mark.parametrize("dtype, cy_func", [(np.float64, get_log_likelihood_f64), (np.float32, get_log_likelihood_f32)])
def test_get_log_likelihood_tiny_spd_no_inf(dtype, cy_func):
    # Tiny but positive-definite covariance should still avoid +/-inf in normal ranges.
    tiny = np.array(np.finfo(dtype).tiny, dtype=dtype)
    yy = np.array([tiny * 10], dtype=dtype)
    tt = np.array([0.0], dtype=dtype)
    tid = np.array([0], dtype=np.int64)
    mu = np.array([0.0], dtype=dtype)
    sigma_y = np.array([[tiny * 100]], dtype=dtype)
    fpca_lambda = np.array([tiny * 50], dtype=dtype)
    lambda_phi = np.array([[tiny * 5]], dtype=dtype)
    unique_sid = np.array([0], dtype=np.int64)
    sid_cnt = np.array([1], dtype=np.int64)

    got = cy_func(yy, tt, tid, mu, sigma_y, fpca_lambda, lambda_phi, unique_sid, sid_cnt)
    assert np.isfinite(got)
    assert not np.isinf(got)
