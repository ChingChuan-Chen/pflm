import math

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pflm.fpca.utils import get_eigen_analysis_results, get_fpca_phi, select_num_pcs_fve
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
