import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pflm.fpca.utils import (
    estimate_rho,
    get_covariance_matrix,
    get_eigen_analysis_results,
    get_eigenvalue_fit,
    get_fpca_ce_score,
    get_fpca_in_score,
    get_fpca_phi,
    get_raw_cov,
)
from pflm.utils import trapz


def get_phi_cov(ffd, mu):
    raw_cov = get_raw_cov(ffd, mu)
    obs_cov = get_covariance_matrix(raw_cov, ffd.unique_tid)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eig_lambda, eig_vector = get_eigen_analysis_results(obs_cov)
    num_pcs = 2
    fpca_lambda, fpca_phi = get_fpca_phi(num_pcs, ffd.unique_tid, mu, eig_lambda, eig_vector)
    fitted_cov = fpca_phi @ (fpca_phi @ np.diag(fpca_lambda)).T
    return num_pcs, fpca_lambda, fpca_phi, fitted_cov


@pytest.mark.parametrize(
    "flatten_data, dtype, order",
    [(np.float32, np.float32, "F"), (np.float64, np.float64, "F"), (np.float32, np.float32, "C"), (np.float64, np.float64, "C")],
    indirect=["flatten_data"],
)
def test_fpca_ce_score_happy_path(flatten_data, dtype, order):
    ffd, mu = flatten_data
    num_samples = ffd.sid_cnt.size
    num_pcs, fpca_lambda, fpca_phi, fitted_cov = get_phi_cov(ffd, mu)
    lambda_mat = np.diag(fpca_lambda)
    fpca_phi = np.ascontiguousarray(fpca_phi) if order == "C" else np.asfortranarray(fpca_phi)
    sigma2 = dtype(0.3)

    xi, xi_var, yhat_mat, yhat = get_fpca_ce_score(ffd, mu, num_pcs, fpca_lambda, fpca_phi, fitted_cov, sigma2)
    sigma_y = fitted_cov + np.eye(fitted_cov.shape[0], dtype=dtype) * sigma2

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
    yhat_mat_expected = mu.reshape(-1, 1) + fpca_phi @ xi_expected.T
    assert_allclose(yhat_mat, yhat_mat_expected, rtol=1e-5, atol=1e-5)
    assert len(yhat) == num_samples
    assert all(yi_cnt == len(yhat_i) for yi_cnt, yhat_i in zip(ffd.sid_cnt, yhat))


@pytest.mark.parametrize("flatten_data", [np.float64], indirect=True)
def test_get_fpca_ce_score_too_many_num_pcs(flatten_data):
    ffd, mu = flatten_data
    fitted_cov = np.eye(ffd.unique_tid.size, dtype=np.float64)  # dummy covariance
    fpca_phi = np.array([[0.8, 0.1], [0.6, 0.2], [0.1, 0.3]])
    with pytest.raises(ValueError, match="num_pcs must be less than or equal to the number of eigenvalues"):
        get_fpca_ce_score(ffd, mu, 3, np.array([[2.0, 1.0]]), fpca_phi, fitted_cov, 0.5)


@pytest.mark.parametrize("flatten_data", [np.float64], indirect=True)
def test_get_fpca_ce_score_fitted_cov_shape_mismatch(flatten_data):
    ffd, mu = flatten_data
    fitted_cov = np.eye(ffd.unique_tid.size + 1, dtype=np.float64)  # dummy covariance
    fpca_phi = np.array([[0.8, 0.1], [0.6, 0.2], [0.1, 0.3]])
    with pytest.raises(ValueError, match="fitted_cov must have shape"):
        get_fpca_ce_score(ffd, mu, 2, np.array([[2.0, 1.0]]), fpca_phi, fitted_cov, 0.5)


@pytest.mark.parametrize("flatten_data", [np.float64], indirect=True)
def test_get_fpca_ce_score_fpca_phi_shape_mismatch(flatten_data):
    ffd, mu = flatten_data
    fitted_cov = np.eye(ffd.unique_tid.size, dtype=np.float64)  # dummy covariance
    bad_phi = np.array([[0.8, 0.1], [0.6, 0.2]])
    with pytest.raises(ValueError, match="fpca_phi must have shape"):
        get_fpca_ce_score(ffd, mu, 2, np.array([[2.0, 1.0]]), bad_phi, fitted_cov, 0.5)


@pytest.mark.parametrize(
    "flatten_data, dtype, method_rho, order",
    [
        (np.float64, np.float64, "ridge", "F"),
        (np.float64, np.float64, "ridge", "C"),
        (np.float32, np.float32, "ridge", "F"),
        (np.float32, np.float32, "ridge", "C"),
        (np.float64, np.float64, "truncated", "F"),
        (np.float64, np.float64, "truncated", "C"),
        (np.float32, np.float32, "truncated", "F"),
        (np.float32, np.float32, "truncated", "C"),
    ],
    indirect=["flatten_data"],
)
def test_estimate_rho_happy_path(flatten_data, dtype, method_rho, order):
    ffd, mu = flatten_data
    _, fpca_lambda, fpca_phi, fitted_cov = get_phi_cov(ffd, mu)
    fpca_phi = np.ascontiguousarray(fpca_phi) if order == "C" else np.asfortranarray(fpca_phi)
    sigma2 = dtype(2)

    rho_estimate = estimate_rho(method_rho, ffd, ffd.unique_tid, mu, mu, fpca_lambda, fpca_phi, fpca_phi, fitted_cov, sigma2)
    assert np.issubdtype(rho_estimate.dtype, np.floating)
    assert rho_estimate > 0.0

    expected_rho = {"truncated": 0.009893764, "ridge": 0.000007581265}.get(method_rho, None)
    if not (method_rho == "ridge" and dtype == np.float32):
        # not validate results in float32 which is less accurate when solving an inverse with small sigma2
        assert_allclose(rho_estimate, expected_rho, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "flatten_data, dtype, if_shrinkage, order",
    [
        (np.float32, np.float32, False, "F"),
        (np.float64, np.float64, False, "F"),
        (np.float32, np.float32, False, "C"),
        (np.float64, np.float64, False, "C"),
        (np.float32, np.float32, True, "F"),
        (np.float64, np.float64, True, "F"),
        (np.float32, np.float32, True, "C"),
        (np.float64, np.float64, True, "C"),
    ],
    indirect=["flatten_data"],
)
def test_get_fpca_in_score_happy_path(flatten_data, dtype, if_shrinkage, order):
    ffd, mu = flatten_data
    num_samples = ffd.sid_cnt.size
    num_pcs, fpca_lambda, fpca_phi, _ = get_phi_cov(ffd, mu)
    fpca_phi = np.ascontiguousarray(fpca_phi) if order == "C" else np.asfortranarray(fpca_phi)
    sigma2 = dtype(0.3)

    expected_xi = np.zeros((num_samples, num_pcs), dtype=dtype)
    t_range = ffd.unique_tid[-1] - ffd.unique_tid[0]
    for i, sid_val in enumerate(ffd.unique_sid):
        mask = ffd.sid == sid_val
        tid_i = ffd.tid[mask]
        yi = ffd.y[mask]
        temp = np.zeros((num_pcs, ffd.sid_cnt[i]), dtype=dtype)
        for j in range(num_pcs):
            temp[j, :] = fpca_phi[tid_i, j] * (yi - mu[tid_i])
        expected_xi[i, :] = trapz(temp, ffd.t[mask])
        if if_shrinkage:
            for j in range(num_pcs):
                expected_xi[i, j] *= fpca_lambda[j] / (fpca_lambda[j] + t_range * sigma2 / ffd.sid_cnt[i])

    xi, xi_var, yhat_mat, yhat = get_fpca_in_score(ffd, mu, 2, fpca_lambda, fpca_phi, sigma2, if_shrinkage)

    assert xi.shape == (num_samples, 2)
    assert_allclose(xi, expected_xi, rtol=1e-5, atol=1e-5)
    assert len(xi_var) == num_samples
    for i in range(num_samples):
        assert xi_var[i].shape == (num_pcs, num_pcs)
    assert yhat_mat.shape == (mu.size, num_samples)
    yhat_mat_expected = mu.reshape(-1, 1) + fpca_phi @ xi.T
    assert_allclose(yhat_mat, yhat_mat_expected, rtol=1e-5, atol=1e-5)
    assert len(yhat) == num_samples
    assert all(yi_cnt == len(yhat_i) for yi_cnt, yhat_i in zip(ffd.sid_cnt, yhat))


@pytest.mark.parametrize("flatten_data", [np.float64], indirect=True)
def test_get_fpca_in_score_too_many_num_pcs(flatten_data):
    ffd, mu = flatten_data
    fpca_phi = np.array([[0.8, 0.1], [0.6, 0.2], [0.1, 0.3]])
    with pytest.raises(ValueError, match="num_pcs must be less than or equal to the number of eigenvalues"):
        get_fpca_in_score(ffd, mu, 3, np.array([2.0, 1.0]), fpca_phi, 0.5, False)


@pytest.mark.parametrize("flatten_data", [np.float64], indirect=True)
def test_get_fpca_in_score_fpca_phi_shape_mismatch(flatten_data):
    ffd, mu = flatten_data
    bad_phi = np.array([[0.8, 0.1], [0.6, 0.2]])
    with pytest.raises(ValueError, match="fpca_phi must have shape"):
        get_fpca_in_score(ffd, mu, 2, np.array([[2.0, 1.0]]), bad_phi, 0.5, False)


@pytest.mark.parametrize(
    "flatten_data,bad_value",
    [(np.float64, None), (np.float64, 0), (np.float64, 1.0), (np.float64, "string"), (np.float64, np.nan)],
    indirect=["flatten_data"],
)
def test_get_fpca_in_score_fpca_bad_if_shrinkage(flatten_data, bad_value):
    ffd, mu = flatten_data
    fpca_phi = np.array([[0.8, 0.1], [0.6, 0.2], [0.1, 0.3]])
    with pytest.raises(ValueError, match="if_shrinkage must be a boolean"):
        get_fpca_in_score(ffd, mu, 2, np.array([[2.0, 1.0]]), fpca_phi, 0.5, bad_value)


@pytest.mark.parametrize("raw_cov,dtype", [(np.float64, np.float64), (np.float32, np.float32)], indirect=["raw_cov"])
def test_get_eigenvalue_fit_happy_path(raw_cov, dtype):
    raw_cov_unit_weight = raw_cov.copy()
    raw_cov_unit_weight[:, 3] = 1.0  # set all weights to 1.0
    obs_grid = np.array([0.1, 0.2, 0.3], dtype=dtype)
    fpca_phi_obs = np.array([[0.8, 0.1], [0.6, 0.2], [0.1, 0.3]], dtype=dtype)

    # case for unit weights
    ev_fit_vals_unit_weight = get_eigenvalue_fit(raw_cov_unit_weight, obs_grid, fpca_phi_obs, 2)
    expected_ev_fit_vals_unit_weight = np.array([1.481559983044, 18.935989826198], dtype=dtype)
    assert ev_fit_vals_unit_weight.shape == (2,)
    assert_allclose(ev_fit_vals_unit_weight, expected_ev_fit_vals_unit_weight, rtol=1e-5, atol=1e-5)
    # make sure raw_cov is not changed
    assert_allclose(raw_cov_unit_weight[:3, 1], np.array([0.1, 0.1, 0.1], dtype=dtype), rtol=1e-5, atol=1e-5)
    assert_allclose(raw_cov_unit_weight[:3, 2], np.array([0.1, 0.2, 0.3], dtype=dtype), rtol=1e-5, atol=1e-5)
    assert_allclose(raw_cov_unit_weight[:2, 4], np.array([2.25, 0.75], dtype=dtype), rtol=1e-5, atol=1e-5)

    # case for non-unit weights
    ev_fit_vals = get_eigenvalue_fit(raw_cov, obs_grid, fpca_phi_obs, 2)
    expected_ev_fit_vals = np.array([1.916509202883, 16.338442158304], dtype=dtype)
    assert ev_fit_vals.shape == (2,)
    assert_allclose(ev_fit_vals, expected_ev_fit_vals, rtol=1e-5, atol=1e-5)
    # make sure raw_cov is not changed
    assert_allclose(raw_cov[:3, 1], np.array([0.1, 0.1, 0.1], dtype=dtype), rtol=1e-5, atol=1e-5)
    assert_allclose(raw_cov[:3, 2], np.array([0.1, 0.2, 0.3], dtype=dtype), rtol=1e-5, atol=1e-5)
    assert_allclose(raw_cov[:2, 4], np.array([2.25, 0.75], dtype=dtype), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("raw_cov", [np.float64], indirect=True)
def test_get_eigenvalue_fit_mismatch_shape(raw_cov):
    obs_grid = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    fpca_phi_obs = np.array([[0.8, 0.1], [0.6, 0.2], [0.1, 0.3]], dtype=np.float64)
    with pytest.raises(ValueError, match="fpca_phi_obs must have shape"):
        get_eigenvalue_fit(raw_cov, obs_grid, fpca_phi_obs.T, 2)
