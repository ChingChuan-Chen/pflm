#ifndef __INTERP_H__
#define __INTERP_H__
#include <cstddef>

#ifdef _OPENMP
#include <omp.h>
#endif

template <typename T>
void find_le_indices(const T* a, std::size_t n, const T* b, std::size_t m, std::ptrdiff_t* result);

template <typename T>
void interp1d_linear(T[], T[], T[], T[], std::ptrdiff_t, std::ptrdiff_t);

template <typename T>
void interp1d_spline_small(T[], T[], T[], T[], std::ptrdiff_t, std::ptrdiff_t);

template <typename T>
void interp1d_spline(T[], T[], T[], T[], std::ptrdiff_t, std::ptrdiff_t);

template <typename T>
void interp2d_linear(
    T[], T[], T[], T[], T[], T[],
    std::ptrdiff_t, std::ptrdiff_t,
    std::ptrdiff_t, std::ptrdiff_t
);

template <typename T>
void interp2d_spline(
    T[], T[], T[], T[], T[], T[],
    std::ptrdiff_t, std::ptrdiff_t,
    std::ptrdiff_t, std::ptrdiff_t
);

#endif
