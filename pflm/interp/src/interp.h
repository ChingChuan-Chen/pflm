#ifndef __INTERP_H__
#define __INTERP_H__
#include <cstdint>

#ifdef _OPENMP
#include <omp.h>
#endif

template <typename T>
void find_le_indices(const T* a, std::uint64_t n, const T* b, std::uint64_t m, std::int64_t* result);

template <typename T>
void interp1d_linear(T[], T[], T[], T[], std::int64_t, std::int64_t);

template <typename T>
void interp1d_spline_small(T[], T[], T[], T[], std::int64_t, std::int64_t);

template <typename T>
void interp1d_spline(T[], T[], T[], T[], std::int64_t, std::int64_t);

template <typename T>
void interp2d_linear(
    T[], T[], T[], T[], T[], T[],
    std::int64_t, std::int64_t,
    std::int64_t, std::int64_t
);

template <typename T>
void interp2d_spline(
    T[], T[], T[], T[], T[], T[],
    std::int64_t, std::int64_t,
    std::int64_t, std::int64_t
);

#endif
