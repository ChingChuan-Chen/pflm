from cython cimport floating
from sklearn.utils._cython_blas cimport BLAS_Order

cdef void _trapz_mat_blas(floating[:, :] y, floating[:] dx, floating[:] out, int p, int n, BLAS_Order order) nogil
