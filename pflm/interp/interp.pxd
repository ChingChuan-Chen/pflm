import numpy as np
cimport numpy as np
from libc.stdint cimport int64_t, uint64_t

cdef extern from "src/interp.cpp" nogil:
    pass


cdef extern from "src/interp.h" nogil:
    void find_le_indices "find_le_indices"[T](T*, uint64_t, T*, uint64_t, int64_t*)
    void interp1d_linear "interp1d_linear"[T](T*, T*, T*, T*, uint64_t, uint64_t)
    void interp1d_spline_small "interp1d_spline_small"[T](T*, T*, T*, T*, uint64_t, uint64_t)
    void interp1d_spline "interp1d_spline"[T](T*, T*, T*, T*, uint64_t, uint64_t)
    void interp2d_linear "interp2d_linear"[T](T*, T*, T*, T*, T*, T*, uint64_t, uint64_t, uint64_t, uint64_t)
    void interp2d_spline "interp2d_spline"[T](T*, T*, T*, T*, T*, T*, uint64_t, uint64_t, uint64_t, uint64_t)
