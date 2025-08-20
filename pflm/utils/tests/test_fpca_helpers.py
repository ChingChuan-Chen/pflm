import math

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pflm.utils.covariance_utils import get_covariance_matrix, get_raw_cov
from pflm.utils.fpca_helpers import (
    estimate_rho,
    get_eigen_analysis_results,
    get_fpca_ce_score,
    get_fpca_phi,
    select_num_pcs_fve,
)
from pflm.utils.utility import flatten_and_sort_data_matrices, trapz


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_get_eigen_analysis_results_happy_path(dtype):
    A = np.array([[1.1, 1, 1], [1, 1.1, 1], [1, 1, 1.1]], dtype=dtype)
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
    import pflm.utils.fpca_helpers as fh

    def fake_syevd(eig_vector, eig_lambda, uplo, n, lwork):
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


def _build_flatten_data(dtype):
    y = [np.array([1.0, 2.0, 2.0], dtype=dtype), np.array([3.0, 4.0], dtype=dtype), np.array([4.0, 5.0], dtype=dtype)]
    t = [np.array([0.1, 0.2, 0.3], dtype=dtype), np.array([0.2, 0.3], dtype=dtype), np.array([0.1, 0.3], dtype=dtype)]
    w = np.array([1.0, 2.0, 2.0], dtype=dtype)
    ffd = flatten_and_sort_data_matrices(y, t, dtype, w)
    mu = (np.bincount(ffd.tid, ffd.y) / np.bincount(ffd.tid)).astype(dtype, copy=False)
    return ffd, mu


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_fpca_ce_score_happy_path(dtype):
    ffd, mu = _build_flatten_data(dtype)
    num_samples = len(ffd.sid_cnt)
    raw_cov = get_raw_cov(ffd, mu)
    obs_cov = get_covariance_matrix(raw_cov, ffd.unique_tid)
    with pytest.warns(UserWarning, match="Eigenvalues contain NaN or negative values."):
        eig_lambda, eig_vector = get_eigen_analysis_results(obs_cov)
    assert np.sum(eig_lambda > 0) >= 2
    num_pcs = 2
    fpca_lambda, fpca_phi = get_fpca_phi(num_pcs, ffd.unique_tid, mu, eig_lambda, eig_vector)
    lambda_mat = np.diag(fpca_lambda)
    fitted_cov = fpca_phi @ (fpca_phi @ lambda_mat).T
    sigma2 = dtype(0.3)

    xi, xi_var, yhat_mat, yhat = get_fpca_ce_score(ffd, num_pcs, mu, fitted_cov, fpca_lambda, fpca_phi, sigma2)
    sigma_y = fitted_cov + np.eye(fitted_cov.shape[0], dtype=dtype) * sigma2

    # manual expected
    xi_expected = np.zeros((ffd.unique_sid.size, num_pcs), dtype=dtype)
    xi_var_expected = []
    for i, sid_val in enumerate(ffd.unique_sid):
        mask = ffd.sid == sid_val
        tid_i = ffd.tid[mask]
        yi = ffd.y[mask]
        sigma_lambda_phi = np.linalg.solve(sigma_y[np.ix_(tid_i, tid_i)], (fpca_phi @ lambda_mat)[tid_i, :])
        xi_expected[i] = sigma_lambda_phi.T @ (yi - mu[tid_i])
        xi_var_expected.append(lambda_mat - (fpca_phi @ lambda_mat)[tid_i, :].T @ sigma_lambda_phi)

    assert xi.shape == xi_expected.shape
    assert_allclose(xi, xi_expected, rtol=1e-5, atol=1e-5)
    assert len(xi_var) == num_samples
    for i in range(num_samples):
        assert xi_var[i].shape == (num_pcs, num_pcs)
        assert_allclose(xi_var[i], xi_var_expected[i], rtol=1e-5, atol=1e-5)
    assert yhat_mat.shape == (mu.size, num_samples)
    yhat_mat_expected = mu + fpca_phi @ xi.T
    assert_allclose(yhat_mat, yhat_mat_expected, rtol=1e-5, atol=1e-5)
    assert len(yhat) == num_samples
    assert all(yi_cnt == len(yhat_i) for yi_cnt, yhat_i in zip(ffd.sid_cnt, yhat))


def test_get_fpca_ce_score_too_many_num_pcs():
    ffd, mu = _build_flatten_data(np.float64)
    fitted_cov = np.eye(ffd.unique_tid.size, dtype=np.float64)  # dummy covariance
    fpca_phi = np.array([[0.8, 0.1], [0.6, 0.2], [0.1, 0.3]])
    with pytest.raises(ValueError, match="num_pcs must be less than or equal to the number of eigenvalues"):
        get_fpca_ce_score(ffd, 3, mu, fitted_cov, np.array([[2.0, 1.0]]), fpca_phi, 0.5)


def test_get_fpca_ce_score_fitted_cov_shape_mismatch():
    ffd, mu = _build_flatten_data(np.float64)
    fitted_cov = np.eye(ffd.unique_tid.size + 1, dtype=np.float64)  # dummy covariance
    fpca_phi = np.array([[0.8, 0.1], [0.6, 0.2], [0.1, 0.3]])
    with pytest.raises(ValueError, match="fitted_cov must have shape"):
        get_fpca_ce_score(ffd, 2, mu, fitted_cov, np.array([[2.0, 1.0]]), fpca_phi, 0.5)


def test_get_fpca_ce_score_fpca_phi_shape_mismatch():
    ffd, mu = _build_flatten_data(np.float64)
    fitted_cov = np.eye(ffd.unique_tid.size, dtype=np.float64)  # dummy covariance
    bad_phi = np.array([[0.8, 0.1], [0.6, 0.2]])
    with pytest.raises(ValueError, match="fpca_phi must have shape"):
        get_fpca_ce_score(ffd, 2, mu, fitted_cov, np.array([[2.0, 1.0]]), bad_phi, 0.5)


@pytest.mark.parametrize("method_rho,dtype", [("ridge", np.float64), ("ridge", np.float32), ("truncated", np.float64), ("truncated", np.float32)])
def test_estimate_rho_happy_path(method_rho, dtype):
    ffd, mu = _build_flatten_data(dtype)
    raw_cov = get_raw_cov(ffd, mu)
    obs_cov = get_covariance_matrix(raw_cov, ffd.unique_tid)
    with pytest.warns(UserWarning, match="Eigenvalues contain NaN or negative values."):
        eig_lambda, eig_vector = get_eigen_analysis_results(obs_cov)
    assert np.sum(eig_lambda > 0) >= 2
    num_pcs = 2
    fpca_lambda, fpca_phi = get_fpca_phi(num_pcs, ffd.unique_tid, mu, eig_lambda, eig_vector)
    lambda_mat = np.diag(fpca_lambda)
    fitted_cov = fpca_phi @ (fpca_phi @ lambda_mat).T
    sigma2 = dtype(0.3)

    rho_estimate = estimate_rho(method_rho, ffd, mu, fitted_cov, fpca_lambda, fpca_phi, sigma2)
    print(f"Estimated rho: {rho_estimate}")
    assert np.issubdtype(rho_estimate.dtype, np.floating)
    assert rho_estimate > 0
    if method_rho == "truncated":
        assert np.allclose(rho_estimate, 2.640398, rtol=1e-5, atol=1e-5)
