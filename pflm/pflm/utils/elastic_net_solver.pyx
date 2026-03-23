import numpy as np
cimport numpy as np
from cython cimport floating
from libc.stdlib cimport malloc, free
from libc.math cimport sqrt, exp, log, fabs as c_fabs
from cython.parallel cimport prange
from pflm.utils.blas_helper cimport BLAS_Order, RowMajor, BLAS_Trans, NoTrans, Trans, BLAS_Uplo, Upper, _gemv, _gemm
from pflm.utils.lapack_helper cimport _posv


cdef int fit_gaussian_helper(
    int n, int p,
    floating* xtx_rho_inv,
    floating* xty,
    floating l1_reg,
    floating l2_reg,
    floating rho,
    int max_iter,
    floating abs_tol,
    floating rel_tol,
    int min_iter,
    floating* coef,
) noexcept nogil:
    """Core ADMM loop for weighted Gaussian ElasticNet, operating on raw pointers.

    The caller (``fit_gaussian_f64``/``fit_gaussian_f32``) pre-weights the data
    by multiplying X and y with ``sqrt(sample_weight)``, so the helper operates
    on the already-weighted sufficient statistics.

    Solves:
        min_w  1/(2n) sum_i w_i (y_i - x_i^T w)^2
               + l1_reg * ||w||_1 + (l2_reg / 2) * ||w||_2^2

    where w_i are sample weights (absorbed into xtx and xty by the wrapper).

    ADMM updates (with augmented-Lagrangian parameter rho):
        z^{k+1}   = A^{-1} (Xw^T yw / n + rho (w^k - tau^k))
                    [A = Xw^T Xw / n + rho I, precomputed]
        w^{k+1}   = S_{l1_reg/(rho+l2_reg)}( (z^{k+1} + tau^k) * rho / (rho + l2_reg) )
        tau^{k+1} = tau^k + z^{k+1} - w^{k+1}

    where Xw = X * sqrt(w), yw = y * sqrt(w), and
    S_t(x) = sign(x) max(|x| - t, 0) is the soft-thresholding operator.

    Parameters
    ----------
    n : int. Number of samples (used only for convergence scaling).
    p : int. Number of features / length of coef.
    xtx_rho_inv : floating*, shape (p, p), row-major.
        Precomputed (Xw^T Xw / n + rho I)^{-1}, where Xw = X * sqrt(w).
    xty : floating*, shape (p,).
        Precomputed Xw^T yw / n, where yw = y * sqrt(w).
    l1_reg : floating. L1 penalty (sparsity).
    l2_reg : floating. L2 penalty (ridge).
    rho : floating. ADMM augmented-Lagrangian parameter.
    max_iter : int. Maximum number of ADMM iterations.
    abs_tol : floating. Absolute convergence tolerance.
    rel_tol : floating. Relative convergence tolerance.
    min_iter : int. Minimum number of ADMM iterations.
    coef : floating*, shape (p,). Output coefficients (zero-initialised on entry).

    Returns
    -------
    int : number of ADMM iterations performed, or -1 on allocation failure.
    """
    cdef floating* z = <floating*> malloc(p * sizeof(floating))
    cdef floating* z_new = <floating*> malloc(p * sizeof(floating))
    cdef floating* tau = <floating*> malloc(p * sizeof(floating))
    cdef floating* rhs = <floating*> malloc(p * sizeof(floating))
    if z is NULL or z_new is NULL or tau is NULL or rhs is NULL:
        if z is not NULL: free(z)
        if z_new is not NULL: free(z_new)
        if tau is not NULL: free(tau)
        if rhs is not NULL: free(rhs)
        return -1

    cdef int it = 0, j
    cdef floating rho_plus_l2 = rho + l2_reg
    cdef floating threshold = l1_reg / rho_plus_l2
    cdef floating scale = rho / rho_plus_l2
    cdef floating sqrt_n = sqrt(<floating> n)
    cdef floating eps_pri, eps_dual
    cdef floating norm_coef, norm_z_new, norm_tau, norm_primal, norm_dual
    cdef floating val
    cdef bint converged = False

    for j in range(p):
        coef[j] = 0.0
        z[j] = 0.0
        z_new[j] = 0.0
        tau[j] = 0.0

    while ((it < max_iter) and (not converged)) or (it <= min_iter):
        it += 1

        # rhs = xty + rho * (coef - tau)
        for j in range(p):
            rhs[j] = xty[j] + rho * (coef[j] - tau[j])

        # z_new = xtx_rho_inv @ rhs  (BLAS GEMV)
        _gemv(RowMajor, NoTrans, p, p,
              <floating> 1.0, xtx_rho_inv, p, rhs, 1,
              <floating> 0.0, z_new, 1)

        # Soft-thresholding: coef = S_{threshold}((z_new + tau) * scale)
        for j in range(p):
            val = (z_new[j] + tau[j]) * scale
            if val > threshold:
                coef[j] = val - threshold
            elif val < -threshold:
                coef[j] = val + threshold
            else:
                coef[j] = 0.0

        # tau update
        for j in range(p):
            tau[j] = tau[j] + z_new[j] - coef[j]

        # Convergence check
        norm_coef = 0.0
        norm_z_new = 0.0
        norm_tau = 0.0
        norm_primal = 0.0
        norm_dual = 0.0
        for j in range(p):
            norm_coef += coef[j] * coef[j]
            norm_z_new += z_new[j] * z_new[j]
            norm_tau += tau[j] * tau[j]
            norm_primal += (coef[j] - z_new[j]) * (coef[j] - z_new[j])
            norm_dual += (z_new[j] - z[j]) * (z_new[j] - z[j])
        norm_coef = sqrt(norm_coef)
        norm_z_new = sqrt(norm_z_new)
        norm_tau = sqrt(norm_tau)
        norm_primal = sqrt(norm_primal)
        norm_dual = rho * sqrt(norm_dual)

        eps_pri = sqrt_n * abs_tol + rel_tol * (norm_coef if norm_coef > norm_z_new else norm_z_new)
        eps_dual = sqrt_n * abs_tol + rel_tol * rho * norm_tau
        converged = (norm_primal < eps_pri) and (norm_dual < eps_dual)

        # z = z_new
        for j in range(p):
            z[j] = z_new[j]

    free(z)
    free(z_new)
    free(tau)
    free(rhs)
    return it


cdef int _precompute_gaussian_stats(
    int n, int p,
    floating* x,
    floating* y,
    floating* sw,
    floating rho,
    floating* xtx_rho_inv,
    floating* xty,
) noexcept nogil:
    """Pre-compute weighted sufficient statistics for the Gaussian ADMM solver.

    Computes Xw = X * sqrt(w), yw = y * sqrt(w), then forms:
        xtx_rho_inv = (Xw^T Xw / n + rho I)^{-1}   via _gemm + _posv
        xty         = Xw^T yw / n                    via _gemv

    Parameters
    ----------
    n : int. Number of samples.
    p : int. Number of features.
    x : floating*, shape (n, p), row-major. Design matrix.
    y : floating*, shape (n,). Response vector.
    sw : floating*, shape (n,).
        Non-negative sample weights (normalised so that sum(sw) = n).
    rho : floating. ADMM augmented-Lagrangian parameter.
    xtx_rho_inv : floating*, shape (p, p), row-major.
        Output: (Xw^T Xw / n + rho I)^{-1}.
    xty : floating*, shape (p,).
        Output: Xw^T yw / n.

    Returns
    -------
    int : 0 on success, -1 on allocation failure, >0 on _posv failure.
    """
    cdef floating* xw = <floating*> malloc(n * p * sizeof(floating))
    cdef floating* yw = <floating*> malloc(n * sizeof(floating))
    cdef floating* xtx = <floating*> malloc(p * p * sizeof(floating))
    if xw is NULL or yw is NULL or xtx is NULL:
        if xw is not NULL: free(xw)
        if yw is not NULL: free(yw)
        if xtx is not NULL: free(xtx)
        return -1

    cdef int i, j
    cdef floating sw_sqrt
    cdef floating inv_n = <floating> 1.0 / <floating> n
    cdef int info = 0

    # xw = X * sqrt(sw),  yw = y * sqrt(sw)
    for i in prange(n):
        sw_sqrt = sqrt(sw[i])
        yw[i] = y[i] * sw_sqrt
        for j in range(p):
            xw[i * p + j] = x[i * p + j] * sw_sqrt

    # xty = Xw^T @ yw / n   (p,)
    _gemv(
        RowMajor, Trans, n, p,
        inv_n, xw, p, yw, 1,
        <floating> 0.0, xty, 1
    )

    # xtx = Xw^T @ Xw / n   (p x p, row-major)
    _gemm(
        RowMajor, Trans, NoTrans, p, p, n,
        inv_n, xw, p, xw, p,
        <floating> 0.0, xtx, p
    )

    # xtx += rho * I
    for j in range(p):
        xtx[j * p + j] += rho

    # Invert xtx via _posv:  solve xtx @ X = I  =>  X = xtx^{-1}
    # _posv overwrites xtx with Cholesky factor, xtx_rho_inv with solution
    _posv(RowMajor, Upper, p, p, xtx, p, xtx_rho_inv, p, &info)

    free(xw)
    free(yw)
    free(xtx)
    return info


def fit_gaussian_f64(
    np.ndarray[np.float64_t, ndim=2] x,
    np.ndarray[np.float64_t] y,
    np.ndarray[np.float64_t] sample_weight,
    np.float64_t l1_reg,
    np.float64_t l2_reg,
    np.float64_t rho = 1.0,
    int max_iter = 1000,
    np.float64_t abs_tol = 1e-3,
    np.float64_t rel_tol = 1e-4,
    int min_iter = 3
):
    """Float64 weighted Gaussian ElasticNet wrapper.

    Applies sample weights via ``_precompute_gaussian_stats`` (cdef, using
    BLAS _gemm/_gemv and LAPACK _posv) to form the pre-weighted sufficient
    statistics before delegating to ``fit_gaussian_helper``.

    Parameters
    ----------
    x : ndarray of float64, shape (n, p). Design matrix.
    y : ndarray of float64, shape (n,). Response vector.
    sample_weight : ndarray of float64, shape (n,).
        Non-negative sample weights (normalised so that sum(w) = n).
    l1_reg : float64. L1 penalty.
    l2_reg : float64. L2 penalty.
    rho : float64, default=1.0. ADMM augmented-Lagrangian parameter.
    max_iter : int, default=1000. Maximum ADMM iterations.
    abs_tol : float64, default=1e-3. Absolute convergence tolerance.
    rel_tol : float64, default=1e-4. Relative convergence tolerance.
    min_iter : int, default=3. Minimum ADMM iterations.

    Returns
    -------
    coef : ndarray of float64, shape (p,). Fitted coefficients.
    n_iter : int. Number of ADMM iterations performed.
    """
    cdef int n = x.shape[0], p = x.shape[1]
    cdef np.ndarray[np.float64_t, ndim=2] x_c = np.ascontiguousarray(x, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=2] xtx_rho_inv = np.ascontiguousarray(np.identity(p, dtype=np.float64))
    cdef np.ndarray[np.float64_t] xty = np.empty(p, order='C', dtype=np.float64)
    cdef int info = _precompute_gaussian_stats[double](
        n, p, &x_c[0, 0], &y[0], &sample_weight[0], rho,
        &xtx_rho_inv[0, 0], &xty[0]
    )
    if info != 0:
        raise RuntimeError(f"_precompute_gaussian_stats failed with info={info}")

    cdef np.ndarray[np.float64_t] coef = np.empty(p, order='C', dtype=np.float64)
    cdef int n_iter = fit_gaussian_helper(
        n, p, &xtx_rho_inv[0, 0], &xty[0], l1_reg, l2_reg,
        rho, max_iter, abs_tol, rel_tol, min_iter, &coef[0]
    )
    return coef, n_iter


def fit_gaussian_f32(
    np.ndarray[np.float32_t, ndim=2] x,
    np.ndarray[np.float32_t] y,
    np.ndarray[np.float32_t] sample_weight,
    np.float32_t l1_reg,
    np.float32_t l2_reg,
    np.float32_t rho = 1.0,
    int max_iter = 1000,
    np.float32_t abs_tol = 1e-3,
    np.float32_t rel_tol = 1e-4,
    int min_iter = 3
):
    """Float32 weighted Gaussian ElasticNet wrapper.

    Applies sample weights via ``_precompute_gaussian_stats`` (cdef, using
    BLAS _gemm/_gemv and LAPACK _posv) to form the pre-weighted sufficient
    statistics before delegating to ``fit_gaussian_helper``.

    Parameters
    ----------
    x : ndarray of float32, shape (n, p). Design matrix.
    y : ndarray of float32, shape (n,). Response vector.
    sample_weight : ndarray of float32, shape (n,).
        Non-negative sample weights (normalised so that sum(w) = n).
    l1_reg : float32. L1 penalty.
    l2_reg : float32. L2 penalty.
    rho : float32, default=1.0. ADMM augmented-Lagrangian parameter.
    max_iter : int, default=1000. Maximum ADMM iterations.
    abs_tol : float32, default=1e-3. Absolute convergence tolerance.
    rel_tol : float32, default=1e-4. Relative convergence tolerance.
    min_iter : int, default=3. Minimum ADMM iterations.

    Returns
    -------
    coef : ndarray of float32, shape (p,). Fitted coefficients.
    n_iter : int. Number of ADMM iterations performed.
    """
    cdef int n = x.shape[0], p = x.shape[1]
    cdef np.ndarray[np.float32_t, ndim=2] x_c = np.ascontiguousarray(x, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] xtx_rho_inv = np.ascontiguousarray(np.identity(p, dtype=np.float32))
    cdef np.ndarray[np.float32_t] xty = np.empty(p, order='C', dtype=np.float32)
    cdef int info = _precompute_gaussian_stats[float](
        n, p, &x_c[0, 0], &y[0], &sample_weight[0], rho,
        &xtx_rho_inv[0, 0], &xty[0]
    )
    if info != 0:
        raise RuntimeError(f"_precompute_gaussian_stats failed with info={info}")

    cdef np.ndarray[np.float32_t] coef = np.empty(p, order='C', dtype=np.float32)
    cdef int n_iter = fit_gaussian_helper(
        n, p, &xtx_rho_inv[0, 0], &xty[0], l1_reg, l2_reg,
        rho, max_iter, abs_tol, rel_tol, min_iter, &coef[0]
    )
    return coef, n_iter


cdef int fit_nongaussian_helper(
    int n, int p,
    floating* x,
    floating* y,
    floating* sw,
    int family_code,
    floating power,
    floating l1_reg,
    floating l2_reg,
    floating rho,
    int max_iter,
    floating abs_tol,
    floating rel_tol,
    int min_iter,
    floating* coef,
) noexcept nogil:
    """Core ADMM loop for weighted non-Gaussian ElasticNet.

    Supported families (family_code):
        1 = Binomial (logistic), 2 = Poisson (log link),
        3 = Gamma (log link), 5 = Tweedie (log link, power != 0,1,2).

    Sample weights ``sw`` multiply each sample's contribution to the
    negative log-likelihood, its gradient (d_eta), and its Hessian weight
    used for the IRLS-style working weights in the Newton z-update.

    The z-update uses Newton's method on the weighted augmented-Lagrangian
    subproblem.  The w-update uses soft-threshold (intercept coef[0] is
    NOT penalised).

    Parameters
    ----------
    n : int. Number of samples.
    p : int. Number of features (including intercept).
    x : floating*, shape (n, p), row-major. Design matrix with intercept.
    y : floating*, shape (n,). Response vector.
    sw : floating*, shape (n,).
        Non-negative sample weights (normalized so that sum(sw) = n).
        Each weight scales the objective, gradient, and Hessian for that
        sample.
    family_code : int.  1=binomial, 2=poisson, 3=gamma, 5=tweedie.
    power : floating. Tweedie variance power (unused for other families).
    l1_reg : floating. L1 penalty (sparsity).
    l2_reg : floating. L2 penalty (ridge).
    rho : floating. ADMM augmented-Lagrangian parameter.
    max_iter : int. Maximum number of ADMM iterations.
    abs_tol : floating. Absolute convergence tolerance.
    rel_tol : floating. Relative convergence tolerance.
    min_iter : int. Minimum number of ADMM iterations.
    coef : floating*, shape (p,). Output coefficients.

    Returns
    -------
    int : number of ADMM iterations, or -1 on allocation failure.
    """
    cdef floating THRESH_ = <floating> 30.0
    cdef floating MTHRESH_ = <floating> -30.0

    # Allocate ADMM buffers
    cdef floating* z = <floating*> malloc(p * sizeof(floating))
    cdef floating* z_new = <floating*> malloc(p * sizeof(floating))
    cdef floating* tau = <floating*> malloc(p * sizeof(floating))
    # Newton buffers
    cdef floating* eta = <floating*> malloc(n * sizeof(floating))
    cdef floating* pred = <floating*> malloc(n * sizeof(floating))
    cdef floating* grad = <floating*> malloc(p * sizeof(floating))
    cdef floating* hess = <floating*> malloc(p * p * sizeof(floating))
    cdef floating* xw = <floating*> malloc(n * p * sizeof(floating))
    cdef floating* z_newton = <floating*> malloc(p * sizeof(floating))

    if (z is NULL or z_new is NULL or tau is NULL or eta is NULL or
        pred is NULL or grad is NULL or hess is NULL or xw is NULL or
        z_newton is NULL):
        if z is not NULL: free(z)
        if z_new is not NULL: free(z_new)
        if tau is not NULL: free(tau)
        if eta is not NULL: free(eta)
        if pred is not NULL: free(pred)
        if grad is not NULL: free(grad)
        if hess is not NULL: free(hess)
        if xw is not NULL: free(xw)
        if z_newton is not NULL: free(z_newton)
        return -1

    cdef int it = 0, j, i, newton_it
    cdef floating rho_plus_l2 = rho + l2_reg
    cdef floating threshold = l1_reg / rho_plus_l2
    cdef floating scale = rho / rho_plus_l2
    cdef floating sqrt_n = sqrt(<floating> n)
    cdef floating val, obj, obj_old, sqrt_w, eta_val, mu_val
    cdef floating one_mp = <floating> 1.0 - power
    cdef floating two_mp = <floating> 2.0 - power
    cdef floating norm_coef, norm_z_new, norm_tau, norm_primal, norm_dual
    cdef floating eps_pri, eps_dual_val
    cdef bint converged = False
    cdef int info

    for j in range(p):
        coef[j] = <floating> 0.0
        z[j] = <floating> 0.0
        tau[j] = <floating> 0.0

    while ((it < max_iter) and (not converged)) or (it <= min_iter):
        it += 1

        # ===== z-update: Newton's method on augmented-Lagrangian subproblem =====
        for j in range(p):
            z_newton[j] = z[j]

        obj_old = <floating> 9999.0
        newton_it = 0
        while True:
            newton_it += 1

            # eta = X @ z_newton
            _gemv(RowMajor, NoTrans, n, p,
                  <floating> 1.0, x, p, z_newton, 1,
                  <floating> 0.0, eta, 1)

            # Compute mu (pred), objective, xw (Hessian weights), then d_eta (overwrite pred)
            obj = <floating> 0.0
            if family_code == 1:  # Binomial (logistic)
                for i in range(n):
                    eta_val = eta[i]
                    if eta_val > THRESH_:
                        eta_val = THRESH_
                    elif eta_val < MTHRESH_:
                        eta_val = MTHRESH_
                    pred[i] = <floating> 1.0 / (<floating> 1.0 + exp(-eta_val))
                    obj += sw[i] * (-y[i] * eta[i] - log(<floating> 1.0 - pred[i]))
                    sqrt_w = sqrt(sw[i] * pred[i] * (<floating> 1.0 - pred[i]))
                    for j in range(p):
                        xw[i * p + j] = x[i * p + j] * sqrt_w
                    pred[i] = sw[i] * (pred[i] - y[i])  # weighted d_eta
            elif family_code == 2:  # Poisson (log link)
                for i in range(n):
                    eta_val = eta[i]
                    if eta_val > THRESH_:
                        eta_val = THRESH_
                    pred[i] = exp(eta_val)
                    obj += sw[i] * (-y[i] * eta[i] + pred[i])
                    sqrt_w = sqrt(sw[i] * pred[i])
                    for j in range(p):
                        xw[i * p + j] = x[i * p + j] * sqrt_w
                    pred[i] = sw[i] * (pred[i] - y[i])  # weighted d_eta
            elif family_code == 3:  # Gamma (log link, Fisher scoring weight=1)
                for i in range(n):
                    eta_val = eta[i]
                    if eta_val > THRESH_:
                        eta_val = THRESH_
                    elif eta_val < MTHRESH_:
                        eta_val = MTHRESH_
                    mu_val = exp(eta_val)
                    obj += sw[i] * (y[i] / mu_val + eta_val)
                    sqrt_w = sqrt(sw[i])
                    for j in range(p):
                        xw[i * p + j] = x[i * p + j] * sqrt_w
                    pred[i] = sw[i] * (<floating> 1.0 - y[i] / mu_val)  # weighted d_eta
            elif family_code == 5:  # Tweedie (log link, power != 0,1,2)
                for i in range(n):
                    eta_val = eta[i]
                    if eta_val > THRESH_:
                        eta_val = THRESH_
                    elif eta_val < MTHRESH_:
                        eta_val = MTHRESH_
                    mu_val = exp(eta_val)
                    val = exp(one_mp * eta_val)    # mu^(1-p)
                    sqrt_w = exp(two_mp * eta_val) # mu^(2-p)
                    obj += sw[i] * (-y[i] * val / one_mp + sqrt_w / two_mp)
                    sqrt_w = sqrt(sw[i] * sqrt_w)  # sqrt(w_i * mu^(2-p))
                    for j in range(p):
                        xw[i * p + j] = x[i * p + j] * sqrt_w
                    pred[i] = sw[i] * val * (mu_val - y[i])  # weighted d_eta

            # Augmented Lagrangian: + rho/2 * ||z_newton - coef + tau||^2
            for j in range(p):
                val = z_newton[j] - coef[j] + tau[j]
                obj += rho / <floating> 2.0 * val * val

            # grad = X.T @ d_eta + rho*(z_newton - coef + tau)
            _gemv(RowMajor, Trans, n, p,
                  <floating> 1.0, x, p, pred, 1,
                  <floating> 0.0, grad, 1)
            for j in range(p):
                grad[j] += rho * (z_newton[j] - coef[j] + tau[j])

            # hess = xw.T @ xw + rho*I
            _gemm(RowMajor, Trans, NoTrans, p, p, n,
                  <floating> 1.0, xw, p, xw, p,
                  <floating> 0.0, hess, p)
            for j in range(p):
                hess[j * p + j] += rho

            # Solve hess @ delta = grad via Cholesky (hess is SPD)
            # _posv overwrites hess with factorisation and grad with solution delta
            info = 0
            _posv(RowMajor, Upper, p, 1, hess, p, grad, 1, &info)
            if info != 0:
                break

            # z_newton -= delta  (delta is now stored in grad)
            for j in range(p):
                z_newton[j] -= grad[j]

            # Newton convergence check (matches original: abs(obj-obj_old)/obj)
            if (c_fabs(obj - obj_old) / obj < abs_tol) or (newton_it > max_iter):
                break
            obj_old = obj

        # z_new = Newton result
        for j in range(p):
            z_new[j] = z_newton[j]

        # ===== w-update: soft-thresholding (intercept coef[0] NOT penalised) =====
        coef[0] = z_new[0]
        for j in range(1, p):
            val = (z_new[j] + tau[j]) * scale
            if val > threshold:
                coef[j] = val - threshold
            elif val < -threshold:
                coef[j] = val + threshold
            else:
                coef[j] = <floating> 0.0

        # tau update
        for j in range(p):
            tau[j] += z_new[j] - coef[j]

        # Convergence check
        norm_coef = <floating> 0.0
        norm_z_new = <floating> 0.0
        norm_tau = <floating> 0.0
        norm_primal = <floating> 0.0
        norm_dual = <floating> 0.0
        for j in range(p):
            norm_coef += coef[j] * coef[j]
            norm_z_new += z_new[j] * z_new[j]
            norm_tau += tau[j] * tau[j]
            norm_primal += (coef[j] - z_new[j]) * (coef[j] - z_new[j])
            norm_dual += (z_new[j] - z[j]) * (z_new[j] - z[j])
        norm_coef = sqrt(norm_coef)
        norm_z_new = sqrt(norm_z_new)
        norm_tau = sqrt(norm_tau)
        norm_primal = sqrt(norm_primal)
        norm_dual = rho * sqrt(norm_dual)

        eps_pri = sqrt_n * abs_tol + rel_tol * (norm_coef if norm_coef > norm_z_new else norm_z_new)
        eps_dual_val = sqrt_n * abs_tol + rel_tol * rho * norm_tau
        converged = (norm_primal < eps_pri) and (norm_dual < eps_dual_val)

        for j in range(p):
            z[j] = z_new[j]

    free(z)
    free(z_new)
    free(tau)
    free(eta)
    free(pred)
    free(grad)
    free(hess)
    free(xw)
    free(z_newton)
    return it


def fit_nongaussian_f64(
    np.ndarray[np.float64_t, ndim=2] x,
    np.ndarray[np.float64_t] y,
    np.ndarray[np.float64_t] sample_weight,
    int family_code,
    np.float64_t power = 1.5,
    np.float64_t l1_reg = 0.0,
    np.float64_t l2_reg = 0.0,
    np.float64_t rho = 1.0,
    int max_iter = 1000,
    np.float64_t abs_tol = 1e-3,
    np.float64_t rel_tol = 1e-4,
    int min_iter = 3
):
    """Float64 weighted non-Gaussian ElasticNet wrapper.

    Prepends an intercept column to X, then delegates to
    ``fit_nongaussian_helper`` which incorporates ``sample_weight``
    into the objective, gradient, and Hessian.

    Parameters
    ----------
    x : ndarray of float64, shape (n, p). Design matrix (with intercept).
    y : ndarray of float64, shape (n,). Response vector.
    sample_weight : ndarray of float64, shape (n,).
        Non-negative sample weights (normalised so that sum(w) = n).
    family_code : int. 1=binomial, 2=poisson, 3=gamma, 5=tweedie.
    power : float64, default=1.5. Tweedie variance power.
    l1_reg : float64, default=0.0. L1 penalty.
    l2_reg : float64, default=0.0. L2 penalty.
    rho : float64, default=1.0. ADMM augmented-Lagrangian parameter.
    max_iter : int, default=1000. Maximum ADMM iterations.
    abs_tol : float64, default=1e-3. Absolute convergence tolerance.
    rel_tol : float64, default=1e-4. Relative convergence tolerance.
    min_iter : int, default=3. Minimum ADMM iterations.

    Returns
    -------
    coef : ndarray of float64, shape (p,). Fitted coefficients, including intercept.
    n_iter : int. Number of ADMM iterations performed.
    """
    cdef int n = x.shape[0], p = x.shape[1]
    cdef np.ndarray[np.float64_t, ndim=2] x2 = np.ascontiguousarray(x, dtype=np.float64)
    cdef np.ndarray[np.float64_t] y_arr = np.ascontiguousarray(y, dtype=np.float64)
    cdef np.ndarray[np.float64_t] coef = np.empty(p, order='C', dtype=np.float64)
    cdef int n_iter = fit_nongaussian_helper(
        n, p, &x2[0, 0], &y_arr[0], &sample_weight[0], family_code, power,
        l1_reg, l2_reg, rho, max_iter, abs_tol, rel_tol, min_iter, &coef[0]
    )
    return coef, n_iter


def fit_nongaussian_f32(
    np.ndarray[np.float32_t, ndim=2] x,
    np.ndarray[np.float32_t] y,
    np.ndarray[np.float32_t] sample_weight,
    int family_code,
    np.float32_t power = 1.5,
    np.float32_t l1_reg = 0.0,
    np.float32_t l2_reg = 0.0,
    np.float32_t rho = 1.0,
    int max_iter = 1000,
    np.float32_t abs_tol = 1e-3,
    np.float32_t rel_tol = 1e-4,
    int min_iter = 3
):
    """Float32 weighted non-Gaussian ElasticNet wrapper.

    Prepends an intercept column to X, then delegates to
    ``fit_nongaussian_helper`` which incorporates ``sample_weight``
    into the objective, gradient, and Hessian.

    Parameters
    ----------
    x : ndarray of float32, shape (n, p). Design matrix (with intercept).
    y : ndarray of float32, shape (n,). Response vector.
    sample_weight : ndarray of float32, shape (n,).
        Non-negative sample weights (normalised so that sum(w) = n).
    family_code : int. 1=binomial, 2=poisson, 3=gamma, 5=tweedie.
    power : float32, default=1.5. Tweedie variance power.
    l1_reg : float32, default=0.0. L1 penalty.
    l2_reg : float32, default=0.0. L2 penalty.
    rho : float32, default=1.0. ADMM augmented-Lagrangian parameter.
    max_iter : int, default=1000. Maximum ADMM iterations.
    abs_tol : float32, default=1e-3. Absolute convergence tolerance.
    rel_tol : float32, default=1e-4. Relative convergence tolerance.
    min_iter : int, default=3. Minimum ADMM iterations.

    Returns
    -------
    coef : ndarray of float32, shape (p,). Fitted coefficients, including intercept.
    n_iter : int. Number of ADMM iterations performed.
    """
    cdef int n = x.shape[0], p = x.shape[1]
    cdef np.ndarray[np.float32_t, ndim=2] x2 = np.ascontiguousarray(x, dtype=np.float32)
    cdef np.ndarray[np.float32_t] y_arr = np.ascontiguousarray(y, dtype=np.float32)
    cdef np.ndarray[np.float32_t] coef = np.empty(p, order='C', dtype=np.float32)
    cdef int n_iter = fit_nongaussian_helper(
        n, p, &x2[0, 0], &y_arr[0], &sample_weight[0], family_code, power,
        l1_reg, l2_reg, rho, max_iter, abs_tol, rel_tol, min_iter, &coef[0]
    )
    return coef, n_iter


# ===========================================================================
# Multinomial logistic ElasticNet via ADMM
# ===========================================================================

cdef int fit_multinomial_helper(
    int n, int p, int K,
    floating* x,
    floating* y_onehot,
    floating* sw,
    floating l1_reg,
    floating l2_reg,
    floating rho,
    int max_iter,
    floating abs_tol,
    floating rel_tol,
    int min_iter,
    floating* coef,
) noexcept nogil:
    """ADMM loop for weighted multinomial logistic ElasticNet.

    Uses block-diagonal Hessian approximation (independent Newton per class).
    Sample weights ``sw`` multiply each sample's contribution to the
    cross-entropy objective, gradient (d_eta), and Hessian weight
    ``sqrt(sw[i] * p_k * (1 - p_k))`` used for the working-weights
    matrix in the Newton z-update.

    Parameters
    ----------
    n : int. Number of samples.
    p : int. Number of features (including intercept).
    K : int. Number of classes.
    x : floating*, (n, p) row-major. Design matrix with intercept column.
    y_onehot : floating*, (n, K) row-major. One-hot encoded labels.
    sw : floating*, shape (n,).
        Non-negative sample weights (normalised so that sum(sw) = n).
        Each weight scales the cross-entropy loss, gradient, and Hessian
        for that sample.
    l1_reg : floating. L1 penalty (sparsity, not applied to intercept).
    l2_reg : floating. L2 penalty (ridge).
    rho : floating. ADMM augmented-Lagrangian parameter.
    max_iter : int. Maximum number of ADMM iterations.
    abs_tol : floating. Absolute convergence tolerance.
    rel_tol : floating. Relative convergence tolerance.
    min_iter : int. Minimum number of ADMM iterations.
    coef : floating*, (K, p) row-major. Output coefficients.

    Returns
    -------
    int : number of ADMM iterations, or -1 on allocation failure.
    """
    cdef floating THRESH_ = <floating> 30.0

    # ADMM buffers
    cdef floating* z = <floating*> malloc(K * p * sizeof(floating))
    cdef floating* z_new = <floating*> malloc(K * p * sizeof(floating))
    cdef floating* tau = <floating*> malloc(K * p * sizeof(floating))
    # Newton buffers
    cdef floating* eta = <floating*> malloc(n * K * sizeof(floating))
    cdef floating* prob = <floating*> malloc(n * K * sizeof(floating))
    cdef floating* z_newton = <floating*> malloc(K * p * sizeof(floating))
    cdef floating* d_eta_k = <floating*> malloc(n * sizeof(floating))
    cdef floating* grad = <floating*> malloc(p * sizeof(floating))
    cdef floating* hess = <floating*> malloc(p * p * sizeof(floating))
    cdef floating* xw = <floating*> malloc(n * p * sizeof(floating))

    if (z is NULL or z_new is NULL or tau is NULL or eta is NULL or
        prob is NULL or z_newton is NULL or d_eta_k is NULL or
        grad is NULL or hess is NULL or xw is NULL):
        if z is not NULL: free(z)
        if z_new is not NULL: free(z_new)
        if tau is not NULL: free(tau)
        if eta is not NULL: free(eta)
        if prob is not NULL: free(prob)
        if z_newton is not NULL: free(z_newton)
        if d_eta_k is not NULL: free(d_eta_k)
        if grad is not NULL: free(grad)
        if hess is not NULL: free(hess)
        if xw is not NULL: free(xw)
        return -1

    cdef int it = 0, i, j, k, newton_it
    cdef floating rho_plus_l2 = rho + l2_reg
    cdef floating threshold = l1_reg / rho_plus_l2
    cdef floating sc = rho / rho_plus_l2
    cdef floating sqrt_n = sqrt(<floating> n)
    cdef floating obj, obj_old, val, sqrt_w, max_eta, sum_exp, pk
    cdef floating norm_coef, norm_z_new, norm_tau, norm_primal, norm_dual
    cdef floating eps_pri, eps_dual_val
    cdef bint converged = False
    cdef int info

    for j in range(K * p):
        coef[j] = <floating> 0.0
        z[j] = <floating> 0.0
        tau[j] = <floating> 0.0

    while ((it < max_iter) and (not converged)) or (it <= min_iter):
        it += 1

        # ===== z-update: block-diagonal Newton =====
        for j in range(K * p):
            z_newton[j] = z[j]

        obj_old = <floating> 9999.0
        newton_it = 0
        while True:
            newton_it += 1

            # eta = X @ Z^T :  (n, K)  stored row-major
            _gemm(RowMajor, NoTrans, Trans, n, K, p,
                  <floating> 1.0, x, p, z_newton, p,
                  <floating> 0.0, eta, K)

            # softmax  +  objective
            obj = <floating> 0.0
            for i in range(n):
                max_eta = eta[i * K]
                for k in range(1, K):
                    if eta[i * K + k] > max_eta:
                        max_eta = eta[i * K + k]
                sum_exp = <floating> 0.0
                for k in range(K):
                    val = eta[i * K + k] - max_eta
                    if val < -THRESH_:
                        val = -THRESH_
                    prob[i * K + k] = exp(val)
                    sum_exp += prob[i * K + k]
                for k in range(K):
                    prob[i * K + k] /= sum_exp
                    pk = prob[i * K + k]
                    if pk < <floating> 1e-15:
                        pk = <floating> 1e-15
                    obj += sw[i] * (-y_onehot[i * K + k] * log(pk))

            # augmented Lagrangian
            for j in range(K * p):
                val = z_newton[j] - coef[j] + tau[j]
                obj += rho / <floating> 2.0 * val * val

            # Newton step for each class k
            for k in range(K):
                # d_eta_k = sw * (prob[:,k] - y_onehot[:,k])
                for i in range(n):
                    d_eta_k[i] = sw[i] * (prob[i * K + k] - y_onehot[i * K + k])
                # grad_k = X^T d_eta_k + rho*(z_newton_k - coef_k + tau_k)
                _gemv(RowMajor, Trans, n, p,
                      <floating> 1.0, x, p, d_eta_k, 1,
                      <floating> 0.0, grad, 1)
                for j in range(p):
                    grad[j] += rho * (z_newton[k * p + j] - coef[k * p + j] + tau[k * p + j])
                # xw[i,j] = x[i,j] * sqrt(sw[i] * prob[i,k]*(1-prob[i,k]))
                for i in range(n):
                    pk = prob[i * K + k]
                    sqrt_w = sqrt(sw[i] * pk * (<floating> 1.0 - pk))
                    for j in range(p):
                        xw[i * p + j] = x[i * p + j] * sqrt_w
                # hess_k = xw^T xw + rho*I
                _gemm(RowMajor, Trans, NoTrans, p, p, n,
                      <floating> 1.0, xw, p, xw, p,
                      <floating> 0.0, hess, p)
                for j in range(p):
                    hess[j * p + j] += rho
                # solve
                info = 0
                _posv(RowMajor, Upper, p, 1, hess, p, grad, 1, &info)
                if info != 0:
                    break
                for j in range(p):
                    z_newton[k * p + j] -= grad[j]

            if info != 0:
                break
            if (c_fabs(obj - obj_old) / (obj + <floating> 1e-30) < abs_tol) or (newton_it > max_iter):
                break
            obj_old = obj

        # z_new = Newton result
        for j in range(K * p):
            z_new[j] = z_newton[j]

        # ===== w-update: soft-thresholding (intercept col NOT penalised) =====
        for k in range(K):
            coef[k * p] = z_new[k * p]  # intercept
            for j in range(1, p):
                val = (z_new[k * p + j] + tau[k * p + j]) * sc
                if val > threshold:
                    coef[k * p + j] = val - threshold
                elif val < -threshold:
                    coef[k * p + j] = val + threshold
                else:
                    coef[k * p + j] = <floating> 0.0

        # tau update
        for j in range(K * p):
            tau[j] += z_new[j] - coef[j]

        # Convergence
        norm_coef = <floating> 0.0
        norm_z_new = <floating> 0.0
        norm_tau = <floating> 0.0
        norm_primal = <floating> 0.0
        norm_dual = <floating> 0.0
        for j in range(K * p):
            norm_coef += coef[j] * coef[j]
            norm_z_new += z_new[j] * z_new[j]
            norm_tau += tau[j] * tau[j]
            norm_primal += (coef[j] - z_new[j]) * (coef[j] - z_new[j])
            norm_dual += (z_new[j] - z[j]) * (z_new[j] - z[j])
        norm_coef = sqrt(norm_coef)
        norm_z_new = sqrt(norm_z_new)
        norm_tau = sqrt(norm_tau)
        norm_primal = sqrt(norm_primal)
        norm_dual = rho * sqrt(norm_dual)
        eps_pri = sqrt_n * abs_tol + rel_tol * (norm_coef if norm_coef > norm_z_new else norm_z_new)
        eps_dual_val = sqrt_n * abs_tol + rel_tol * rho * norm_tau
        converged = (norm_primal < eps_pri) and (norm_dual < eps_dual_val)

        for j in range(K * p):
            z[j] = z_new[j]

    free(z)
    free(z_new)
    free(tau)
    free(eta)
    free(prob)
    free(z_newton)
    free(d_eta_k)
    free(grad)
    free(hess)
    free(xw)
    return it


def fit_multinomial_f64(
    np.ndarray[np.float64_t, ndim=2] x,
    np.ndarray[np.float64_t, ndim=2] y_onehot,
    np.ndarray[np.float64_t] sample_weight,
    np.float64_t l1_reg = 0.0,
    np.float64_t l2_reg = 0.0,
    np.float64_t rho = 1.0,
    int max_iter = 1000,
    np.float64_t abs_tol = 1e-3,
    np.float64_t rel_tol = 1e-4,
    int min_iter = 3
):
    """Float64 weighted multinomial logistic ElasticNet wrapper.

    Parameters
    ----------
    x : ndarray of float64, shape (n, p). Design matrix (with intercept).
    y_onehot : ndarray of float64, shape (n, K). One-hot encoded labels.
    sample_weight : ndarray of float64, shape (n,).
        Non-negative sample weights (normalised so that sum(w) = n).
    l1_reg : float64, default=0.0. L1 penalty.
    l2_reg : float64, default=0.0. L2 penalty.
    rho : float64, default=1.0. ADMM augmented-Lagrangian parameter.
    max_iter : int, default=1000. Maximum ADMM iterations.
    abs_tol : float64, default=1e-3. Absolute convergence tolerance.
    rel_tol : float64, default=1e-4. Relative convergence tolerance.
    min_iter : int, default=3. Minimum ADMM iterations.

    Returns
    -------
    coef : ndarray of float64, shape (K, p). Fitted coefficients.
    n_iter : int. Number of ADMM iterations performed.
    """
    cdef int n = x.shape[0], p = x.shape[1], K = y_onehot.shape[1]
    cdef np.ndarray[np.float64_t, ndim=2] x2 = np.ascontiguousarray(x, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=2] y_oh = np.ascontiguousarray(y_onehot, dtype=np.float64)
    cdef np.ndarray[np.float64_t] coef_flat = np.empty(K * p, order='C', dtype=np.float64)
    cdef int n_iter = fit_multinomial_helper(
        n, p, K, &x2[0, 0], &y_oh[0, 0], &sample_weight[0],
        l1_reg, l2_reg, rho, max_iter, abs_tol, rel_tol, min_iter, &coef_flat[0]
    )
    cdef np.ndarray[np.float64_t, ndim=2] coef2d = coef_flat.reshape(K, p)
    return coef2d, n_iter


def fit_multinomial_f32(
    np.ndarray[np.float32_t, ndim=2] x,
    np.ndarray[np.float32_t, ndim=2] y_onehot,
    np.ndarray[np.float32_t] sample_weight,
    np.float32_t l1_reg = 0.0,
    np.float32_t l2_reg = 0.0,
    np.float32_t rho = 1.0,
    int max_iter = 1000,
    np.float32_t abs_tol = 1e-3,
    np.float32_t rel_tol = 1e-4,
    int min_iter = 3
):
    """Float32 weighted multinomial logistic ElasticNet wrapper.

    Parameters
    ----------
    x : ndarray of float32, shape (n, p). Design matrix (with intercept).
    y_onehot : ndarray of float32, shape (n, K). One-hot encoded labels.
    sample_weight : ndarray of float32, shape (n,).
        Non-negative sample weights (normalised so that sum(w) = n).
    l1_reg : float32, default=0.0. L1 penalty.
    l2_reg : float32, default=0.0. L2 penalty.
    rho : float32, default=1.0. ADMM augmented-Lagrangian parameter.
    max_iter : int, default=1000. Maximum ADMM iterations.
    abs_tol : float32, default=1e-3. Absolute convergence tolerance.
    rel_tol : float32, default=1e-4. Relative convergence tolerance.
    min_iter : int, default=3. Minimum ADMM iterations.

    Returns
    -------
    coef : ndarray of float32, shape (K, p). Fitted coefficients.
    n_iter : int. Number of ADMM iterations performed.
    """
    cdef int n = x.shape[0], p = x.shape[1], K = y_onehot.shape[1]
    cdef np.ndarray[np.float32_t, ndim=2] x2 = np.ascontiguousarray(x, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] y_oh = np.ascontiguousarray(y_onehot, dtype=np.float32)
    cdef np.ndarray[np.float32_t] coef_flat = np.empty(K * p, order='C', dtype=np.float32)
    cdef int n_iter = fit_multinomial_helper(
        n, p, K, &x2[0, 0], &y_oh[0, 0], &sample_weight[0],
        l1_reg, l2_reg, rho, max_iter, abs_tol, rel_tol, min_iter, &coef_flat[0]
    )
    cdef np.ndarray[np.float32_t, ndim=2] coef2d = coef_flat.reshape(K, p)
    return coef2d, n_iter
