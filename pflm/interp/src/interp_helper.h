#ifndef __INTERP_HELPER_H__
#define __INTERP_HELPER_H__
#include <cstddef>

#ifdef _OPENMP
#include <omp.h>
#endif

template <typename T>
std::ptrdiff_t search_sorted(T[], std::ptrdiff_t, T);

template <typename T>
void transpose_matrix(T*, T*, std::ptrdiff_t, std::ptrdiff_t);

#endif
