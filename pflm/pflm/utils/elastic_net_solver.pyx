import numpy as np
cimport numpy as np

def fit_gaussian(
    np.ndarray[np.float64_t, ndim=2] x,
    np.ndarray[np.float64_t] y,
    np.float64_t l1_reg,
    np.float64_t l2_reg,
    np.float64_t rho = 1.0,
    int max_iter = 1000,
    np.float64_t abs_tol = 1e-3,
    np.float64_t rel_tol = 1e-4,
    int min_iter = 3
):
    """Fit Gaussian ElasticNet via ADMM.

    Solves:
        min_w  1/(2n) ||y - Xw||_2^2 + l1_reg * ||w||_1 + (l2_reg / 2) * ||w||_2^2

    ADMM updates (with augmented-Lagrangian parameter rho):
        z^{k+1}   = (X^T X / n + rho I)^{-1} (X^T y / n + rho (w^k - tau^k))
        w^{k+1}   = S_{l1_reg/(rho+l2_reg)}( (z^{k+1} + tau^k) * rho / (rho + l2_reg) )
        tau^{k+1} = tau^k + z^{k+1} - w^{k+1}

    where S_t(x) = sign(x) max(|x| - t, 0) is the soft-thresholding operator.

    Parameters
    ----------
    x : ndarray of shape (n, p). Design matrix.
    y : ndarray of shape (n,). Response vector.
    l1_reg : float. L1 penalty coefficient (sparsity).
    l2_reg : float. L2 penalty coefficient (ridge).
    rho : float. ADMM augmented-Lagrangian parameter.
    """
    cdef size_t n = x.shape[0], p = x.shape[1]
    cdef np.ndarray[np.float64_t, ndim=2] xtx = np.matmul(x.T, x)
    cdef np.ndarray[np.float64_t, ndim=2] xtx_plus_rho_inv = np.linalg.inv(xtx/n + rho * np.identity(p))
    cdef np.ndarray[np.float64_t] xty = np.matmul(x.T, y) / n

    cdef np.ndarray[np.float64_t] coef = np.zeros((p,)), z = np.zeros((p,)), tau = np.zeros((p,))
    cdef np.ndarray[np.float64_t] z_new = np.zeros((p,)), temp = np.zeros((p,))

    cdef int iter = 0, j
    cdef np.float64_t rho_plus_l2_reg = rho + l2_reg, threshold = l1_reg / rho_plus_l2_reg
    cdef np.float64_t sqrt_n = np.sqrt(n), eps_pri, eps_dual
    cdef bint converged = False
    while ((iter < max_iter) and (not converged)) or (iter <= min_iter):
        iter += 1
        z_new = np.matmul(xtx_plus_rho_inv, (xty + rho * (coef - tau)))

        temp = (z_new + tau) * rho / rho_plus_l2_reg
        for j in range(p):
            if temp[j] > threshold:
                coef[j] = temp[j] - threshold
            elif temp[j] < -threshold:
                coef[j] = temp[j] + threshold
            else:
                coef[j] = 0.0
        tau += z_new - coef

        eps_pri = sqrt_n * abs_tol + rel_tol * np.maximum(np.linalg.norm(coef), np.linalg.norm(z_new))
        eps_dual = sqrt_n * abs_tol + rel_tol * rho * np.linalg.norm(tau)
        converged = (np.linalg.norm(coef - z_new) < eps_pri) and (rho * np.linalg.norm(z_new-z) < eps_dual)

        z = z_new
    return coef, iter

cdef double MTHRESH = -30.0
cdef double THRESH = 30.0

cdef logistic_loss(
    np.ndarray[np.float64_t] z,
    np.ndarray[np.float64_t, ndim=2] x,
    np.ndarray[np.float64_t] y,
    np.ndarray[np.float64_t] coef,
    np.ndarray[np.float64_t] tau,
    np.float64_t rho,
    np.ndarray[np.float64_t] xty,
):
    cdef size_t n = x.shape[0], p = x.shape[1]
    cdef np.ndarray[np.float64_t] eta = np.matmul(x, z)
    cdef np.ndarray[np.float64_t] phat = 1 /(1 + np.exp(-np.clip(eta, MTHRESH, THRESH)))
    cdef np.float64_t obj = -np.dot(y, eta) - np.sum(np.log(1.0-phat)) + rho / 2.0 * np.linalg.norm(z - coef + tau)**2
    cdef np.ndarray[np.float64_t] grad = np.matmul(x.T, phat) - xty + rho * (z - coef + tau)
    cdef np.ndarray[np.float64_t, ndim=2] xw = x * np.sqrt(phat * (1 - phat)).reshape((n, 1))
    cdef np.ndarray[np.float64_t, ndim=2] hess = np.matmul(xw.T, xw) + rho * np.identity(p)
    return obj/n, grad/n, hess/n

cdef poisson_loss(
    np.ndarray[np.float64_t] z,
    np.ndarray[np.float64_t, ndim=2] x,
    np.ndarray[np.float64_t] y,
    np.ndarray[np.float64_t] coef,
    np.ndarray[np.float64_t] tau,
    np.float64_t rho,
    np.ndarray[np.float64_t] xty,
):
    cdef size_t n = x.shape[0], p = x.shape[1]
    cdef np.ndarray[np.float64_t] eta = np.matmul(x, z)
    cdef np.ndarray[np.float64_t] yhat = np.exp(np.minimum(eta, THRESH))
    cdef np.float64_t obj = -np.dot(y, eta) + np.sum(yhat) + rho / 2.0 * np.linalg.norm(z - coef + tau)**2
    cdef np.ndarray[np.float64_t] grad = np.matmul(x.T, yhat) - xty + rho * (z - coef + tau)
    cdef np.ndarray[np.float64_t, ndim=2] xw = x * np.sqrt(yhat).reshape((n, 1))
    cdef np.ndarray[np.float64_t, ndim=2] hess = np.matmul(xw.T, xw) + rho * np.identity(p)
    return obj/n, grad/n, hess/n

cdef update_coef(
    int family_code,
    np.ndarray[np.float64_t] z,
    np.ndarray[np.float64_t, ndim=2] x,
    np.ndarray[np.float64_t] y,
    np.ndarray[np.float64_t] coef,
    np.ndarray[np.float64_t] tau,
    np.float64_t rho,
    np.ndarray[np.float64_t] xty,
    int max_iter,
    np.float64_t tol
):
    cdef int iter = 0
    cdef np.float64_t obj, obj_old = 9999.0
    cdef np.ndarray[np.float64_t] grad, z_old = z.copy()
    cdef np.ndarray[np.float64_t, ndim=2] hess
    while True:
        iter += 1
        if family_code == 0:
            obj, grad, hess = logistic_loss(z_old, x, y, coef, tau, rho, xty)
        elif family_code == 1:
            obj, grad, hess = poisson_loss(z_old, x, y, coef, tau, rho, xty)
        z_new = z_old - np.linalg.solve(hess, grad)
        if (np.abs(obj - obj_old)/obj < tol) or (iter > max_iter):
            break
        z_old = z_new
        obj_old = obj
    return z_new

def fit_nongaussian(
    np.ndarray[np.float64_t, ndim=2] x,
    np.ndarray[np.float64_t] y,
    int family_code,
    np.float64_t l1_reg,
    np.float64_t l2_reg,
    np.float64_t rho = 1.0,
    int max_iter = 1000,
    np.float64_t abs_tol = 1e-3,
    np.float64_t rel_tol = 1e-4,
    int min_iter = 3
):
    """Fit non-Gaussian (Binomial / Poisson) ElasticNet via ADMM.

    Solves:
        min_w  -l(w; X, y) + l1_reg * ||w||_1 + (l2_reg / 2) * ||w||_2^2

    where l(w; X, y) is the log-likelihood of the chosen family:
        Binomial : l = sum_i [ y_i eta_i + log(1 - sigma(eta_i)) ]
        Poisson  : l = sum_i [ y_i eta_i - exp(eta_i) ]
    with eta = Xw.  An intercept column is prepended and excluded from penalty.

    ADMM updates mirror ``fit_gaussian`` but the z-update uses a
    Newton step on the penalised negative log-likelihood.

    Parameters
    ----------
    x : ndarray of shape (n, p). Design matrix (without intercept column).
    y : ndarray of shape (n,). Response vector.
    family_code : int. 0 = binomial, 1 = poisson.
    l1_reg : float. L1 penalty coefficient (sparsity).
    l2_reg : float. L2 penalty coefficient (ridge).
    rho : float. ADMM augmented-Lagrangian parameter.
    """
    cdef size_t n = x.shape[0], p = x.shape[1]
    cdef np.ndarray[np.float64_t, ndim=2] x2 = np.concatenate([np.repeat(1, n).reshape((n, 1)), x], axis = 1)
    cdef np.ndarray[np.float64_t] xty = np.matmul(x2.T, y)

    cdef np.ndarray[np.float64_t] coef = np.zeros((p+1,)), tau = np.zeros((p+1,))
    cdef np.ndarray[np.float64_t] z = np.zeros((p+1,)), z_new = np.zeros((p+1,)), temp = np.zeros((p+1,))

    cdef int iter = 0, j
    cdef np.float64_t rho_plus_l2_reg = rho + l2_reg, threshold = l1_reg / rho_plus_l2_reg
    cdef np.float64_t sqrt_n = np.sqrt(n), eps_pri, eps_dual
    cdef bint converged = False
    while ((iter < max_iter) and (not converged)) or (iter <= min_iter):
        iter += 1
        z_new = update_coef(family_code, z, x2, y, coef, tau, rho, xty, max_iter, abs_tol)

        # w-update: soft-thresholding (intercept coef[0] is NOT penalised)
        temp = (z_new + tau) * rho / rho_plus_l2_reg
        coef[0] = z_new[0]
        for j in range(1, p+1):
            if temp[j] > threshold:
                coef[j] = temp[j] - threshold
            elif temp[j] < -threshold:
                coef[j] = temp[j] + threshold
            else:
                coef[j] = 0.0
        tau += z_new - coef

        eps_primary = sqrt_n * abs_tol + rel_tol * np.maximum(np.linalg.norm(coef), np.linalg.norm(z_new))
        eps_dual = sqrt_n * abs_tol + rel_tol * rho * np.linalg.norm(tau)
        converged = (np.linalg.norm(coef - z_new) < eps_primary) and (rho * np.linalg.norm(z_new-z) < eps_dual)

        z = z_new
    return coef[0], coef[1:], iter
