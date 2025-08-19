import math

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pflm.utils.utility import flatten_and_sort_data_matrices
from pflm.utils.covariance_utils import get_covariance_matrix, get_raw_cov
from pflm.utils.fpca_helpers import (get_eigen_analysis_results, get_fpca_phi,
                                     select_num_pcs_fve, get_fpca_ce_score)
from pflm.utils.utility import trapz


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_get_eigen_analysis_results_happy_path(dtype):
    A = np.array([[1.1, 1, 1], [1, 1.1, 1], [1, 1, 1.1]], dtype=dtype)
    eig_lambda, eig_vector = get_eigen_analysis_results(A)
    expected_eig_lambda = np.array([3.1, 0.1, 0.1], dtype=dtype)
    expected_first_eig_vector = np.array([1.0 / math.sqrt(3), 1.0 / math.sqrt(3), 1.0 / math.sqrt(3)], dtype=dtype)
    assert_allclose(eig_lambda, expected_eig_lambda, rtol=1e-5, atol=1e-5)
    if np.all(eig_vector[:, 0] >= 0):
        assert_allclose(eig_vector[:, 0], expected_first_eig_vector, rtol=1e-5, atol=1e-5)
    else:
        assert_allclose(eig_vector[:, 0], -expected_first_eig_vector, rtol=1e-5, atol=1e-5)


def _make_spd_matrix(n: int, dtype=np.float64):
    """Create a symmetric positive definite matrix."""
    A = np.arange(1, n * n + 1, dtype=dtype).reshape(n, n)
    M = A @ A.T / n
    np.fill_diagonal(M, np.diag(M) + np.linspace(0, n - 1, n))
    print(M)
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
    assert np.allclose(recon, reg_cov, rtol=1e-5, atol=1e-5)


def test_get_eigen_analysis_results_upper_triangular_path():
    reg_cov = _make_spd_matrix(4)
    lam_u, vec_u = get_eigen_analysis_results(reg_cov, is_upper_triangular=True)
    lam_l, vec_l = get_eigen_analysis_results(reg_cov, is_upper_triangular=False)
    assert_allclose(lam_u, lam_l)
    # Columns may differ by sign
    for i in range(vec_u.shape[1]):
        if np.all(vec_l[:, i] * vec_u[:, i] >= 0):
            assert_allclose(vec_u[:, i], vec_l[:, i])
        else:
            assert_allclose(vec_u[:, i], -vec_l[:, i])


def test_get_eigen_analysis_results_zero_matrix_all_filtered():
    Z = np.zeros((3, 3), dtype=np.float64)
    eig_lambda, eig_vector = get_eigen_analysis_results(Z)
    # All eigenvalues zero -> filtered out
    assert eig_lambda.size == 0
    assert eig_vector.shape[0] == 3
    assert eig_vector.shape[1] == 0


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
    assert n_low == 1
    assert n_high == 4
    assert n_trunc == 2


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


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_fpca_ce_score_happy_path(dtype):
    y = [np.array([1.0, 2.0, 2.0], dtype=dtype), np.array([3.0, 4.0], dtype=dtype), np.array([4.0, 5.0], dtype=dtype)]
    t = [np.array([0.1, 0.2, 0.3], dtype=dtype), np.array([0.2, 0.3], dtype=dtype), np.array([0.1, 0.3], dtype=dtype)]
    w = np.array([1.0, 2.0, 2.0], dtype=dtype)
    yy, tt, ww, sid = flatten_and_sort_data_matrices(y, t, dtype, w)
    obs_grid = np.unique(tt, sorted=True)
    tid = np.digitize(tt, obs_grid, right=True)
    mu = (np.bincount(tid, yy) / np.bincount(tid)).astype(dtype, copy=False)
    raw_cov = get_raw_cov(yy, tt, ww, mu, tid, sid)
    obs_cov = get_covariance_matrix(raw_cov, obs_grid)
    with pytest.warns(UserWarning, match="Eigenvalues contain NaN or negative values."):
        eig_lambda, eig_vector = get_eigen_analysis_results(obs_cov)
    num_pcs = 2
    fpca_lambda, fpca_phi = get_fpca_phi(num_pcs, obs_grid, mu, eig_lambda, eig_vector)
    lambda_mat = np.diag(fpca_lambda)
    lambda_phi = fpca_phi @ np.diag(fpca_lambda)
    fitted_cov = fpca_phi @ lambda_phi.T
    sigma2 = 0.3

    xi, xi_var = get_fpca_ce_score(yy, tt, tid, sid, mu, fitted_cov, fpca_lambda, fpca_phi, sigma2)
    sigma_y = fitted_cov + np.eye(fitted_cov.shape[0]) * sigma2

    xi_expected = np.zeros((3, num_pcs), dtype=dtype)
    xi_var_expected = []
    for i, yi in enumerate(y):
        tid_mask = tid[sid == i]
        sigma_lambda_phi = np.linalg.solve(sigma_y[tid_mask][:, tid_mask], lambda_phi[tid_mask, :])
        xi_expected[i] = sigma_lambda_phi.T @ (yi - mu[tid_mask])
        xi_var_expected.append(lambda_mat - lambda_phi[tid_mask, :].T @ sigma_lambda_phi)

    assert xi.shape == (3, 2)
    assert_allclose(xi, xi_expected, rtol=1e-5, atol=1e-5)
    assert len(xi_var) == 3
    for i in range(len(y)):
        assert xi_var[i].shape == (num_pcs, num_pcs)
        assert_allclose(xi_var[i], xi_var_expected[i], rtol=1e-5, atol=1e-5)


def _valid_fpca_ce_inputs():
    # Construct a minimal consistent valid set (2 time points, 2 samples, each sample has 2 obs)
    yy = np.array([1.0, 2.0, 3.0, 4.0])
    tt = np.array([0.1, 0.2, 0.1, 0.2])
    tid = np.array([0, 1, 0, 1], dtype=np.int64)
    sid = np.array([0, 0, 1, 1], dtype=np.int64)
    mu = np.array([2.0, 3.0])
    fitted_cov = np.array([[1.0, 0.2], [0.2, 1.1]])
    fpca_lambda = np.array([0.5])
    fpca_phi = np.array([[0.8], [0.6]])  # shape (2,1)
    sigma2 = 0.1
    return yy, tt, tid, sid, mu, fitted_cov, fpca_lambda, fpca_phi, sigma2


def test_get_fpca_ce_score_invalid_yy_shape():
    yy, tt, tid, sid, mu, fitted_cov, fpca_lambda, fpca_phi, sigma2 = _valid_fpca_ce_inputs()
    yy2 = yy.reshape(-1, 1)
    with pytest.raises(ValueError, match="yy and tt must be 1D arrays"):
        get_fpca_ce_score(yy2, tt, tid, sid, mu, fitted_cov, fpca_lambda, fpca_phi, sigma2)


def test_get_fpca_ce_score_length_mismatch():
    yy, tt, tid, sid, mu, fitted_cov, fpca_lambda, fpca_phi, sigma2 = _valid_fpca_ce_inputs()
    with pytest.raises(ValueError, match="yy and tt must have the same length"):
        get_fpca_ce_score(yy[:-1], tt, tid[:-1], sid[:-1], mu, fitted_cov, fpca_lambda, fpca_phi, sigma2)


def test_get_fpca_ce_score_mu_length_mismatch():
    yy, tt, tid, sid, mu, fitted_cov, fpca_lambda, fpca_phi, sigma2 = _valid_fpca_ce_inputs()
    mu_bad = np.array([2.0])  # wrong length
    with pytest.raises(ValueError, match="The length of mu must match"):
        get_fpca_ce_score(yy, tt, tid, sid, mu_bad, fitted_cov, fpca_lambda, fpca_phi, sigma2)


def test_get_fpca_ce_score_sid_shape_mismatch():
    yy, tt, tid, sid, mu, fitted_cov, fpca_lambda, fpca_phi, sigma2 = _valid_fpca_ce_inputs()
    sid2 = sid.reshape(-1, 1)
    with pytest.raises(ValueError, match="sid must be a 1D array"):
        get_fpca_ce_score(yy, tt, tid, sid2, mu, fitted_cov, fpca_lambda, fpca_phi, sigma2)


def test_get_fpca_ce_score_sid_not_sorted():
    yy, tt, tid, sid, mu, fitted_cov, fpca_lambda, fpca_phi, sigma2 = _valid_fpca_ce_inputs()
    sid_unsorted = np.array([0, 1, 0, 1], dtype=np.int64)
    with pytest.raises(ValueError, match="must be sorted in ascending order"):
        get_fpca_ce_score(yy, tt, tid, sid_unsorted, mu, fitted_cov, fpca_lambda, fpca_phi, sigma2)


def test_get_fpca_ce_score_sample_too_few_obs():
    yy, tt, tid, sid, mu, fitted_cov, fpca_lambda, fpca_phi, sigma2 = _valid_fpca_ce_inputs()
    # Make sample 0 only one observation
    sid_bad = np.array([0, 1, 1, 1], dtype=np.int64)
    with pytest.raises(ValueError, match="at least two observations"):
        get_fpca_ce_score(yy, tt, tid, sid_bad, mu, fitted_cov, fpca_lambda, fpca_phi, sigma2)


def test_get_fpca_ce_score_fpca_phi_shape_mismatch():
    yy, tt, tid, sid, mu, fitted_cov, fpca_lambda, fpca_phi, sigma2 = _valid_fpca_ce_inputs()
    bad_phi = np.array([[0.8, 0.1], [0.6, 0.2]])  # shape (2,2) but num_pcs = 1
    with pytest.raises(ValueError, match="fpca_phi must have shape"):
        get_fpca_ce_score(yy, tt, tid, sid, mu, fitted_cov, fpca_lambda, bad_phi, sigma2)
