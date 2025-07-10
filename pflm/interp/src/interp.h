#ifndef __INTERP_H__
#define __INTERP_H__
#include <cstddef>

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
