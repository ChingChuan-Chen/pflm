from cython cimport floating
from sklearn.utils._cython_blas cimport BLAS_Order

cdef inline void _trapz_memview(
    floating[:, :], floating[:], floating[:], int, int, BLAS_Order
) noexcept nogil
