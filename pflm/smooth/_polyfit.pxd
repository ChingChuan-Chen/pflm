from cython cimport floating

cdef double half_pi
cdef double quarter_pi
cdef double inv_sqrt_2
cdef double inv_2pi
cdef double inv_sqrt_2pi
cdef double[11] factorials

cdef floating calculate_kernel_value(floating u, int kernel_type) noexcept nogil
