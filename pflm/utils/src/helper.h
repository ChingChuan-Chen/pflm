#ifndef __HELPER_H__
#define __HELPER_H__
#include <cstddef>

#ifdef _OPENMP
#include <omp.h>
#endif

template <typename T>
std::ptrdiff_t search_upper_bound(T[], std::ptrdiff_t, T, bool);

template <typename T>
std::ptrdiff_t search_lower_bound(T[], std::ptrdiff_t, T, bool);

template <typename T>
std::ptrdiff_t search_location(T[], std::ptrdiff_t, T);

template <typename T>
void transpose_matrix(T*, T*, std::ptrdiff_t, std::ptrdiff_t);

#endif
