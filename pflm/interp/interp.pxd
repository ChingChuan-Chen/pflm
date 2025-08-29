from cython cimport floating
from libc.stdint cimport uint64_t


cdef void interp1d_linear(uint64_t, floating*, floating*, uint64_t, floating*, floating*) noexcept nogil
cdef void interp1d_spline_small(uint64_t, floating*, floating*, uint64_t, floating*, floating*) noexcept nogil
cdef void interp1d_spline(uint64_t, floating*, floating*, uint64_t, floating*, floating*) noexcept nogil
cdef void interp2d_linear(uint64_t, floating*, uint64_t, floating*, floating*, uint64_t, floating*, uint64_t, floating*, floating*) noexcept nogil
cdef void interp2d_spline(uint64_t, floating*, uint64_t, floating*, floating*, uint64_t, floating*, uint64_t, floating*, floating*) noexcept nogil
