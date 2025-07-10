from cython cimport floating


cdef void _gels(char, int, int, int, floating*, int, floating*, int, floating*, int, int*) noexcept nogil


cdef void _gelss(
    int, int, int, floating*, int, floating*, int, floating*, floating*, int*, floating*, int, int*
) noexcept nogil


cdef void _gels_helper(char, int, int, int, floating*, int, floating*, int, int*) noexcept nogil


cdef void _gelss_helper(int, int, int, floating*, int, floating*, int, floating*, int*, int*) noexcept nogil
