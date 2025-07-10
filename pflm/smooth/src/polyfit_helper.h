#ifndef __POLYFIT_HELPER_H__
#define __POLYFIT_HELPER_H__
#include <cmath>
#include <vector>
#include <cstddef>

#ifdef _OPENMP
#include <omp.h>
#endif

template <typename T>
std::ptrdiff_t search_lower_bound(T[], std::ptrdiff_t, T, bool);

template <typename T>
std::ptrdiff_t search_location(T[], std::ptrdiff_t, T);

const static double half_pi = std::acos(0.0);
const static double quarter_pi = half_pi / 2.0;
const static double inv_sqrt_2 = 1.0 / std::sqrt(2.0);
const static double inv_2pi = 1.0/(2.0*std::acos(-1.0));
const static double inv_sqrt_2pi = std::sqrt(inv_2pi);
const static double factorials[] = {1,1,2,6,24,120,720,5040,40320,362880,3628800};

template <typename T>
T calculate_kernel_value(T, int, T);

double calculate_sqrt_kernel_value_f64(double, int, double);
float calculate_sqrt_kernel_value_f32(float, int, float);

template <typename T>
void polyfit1d_prepare(
    T bw,
    T center,
    const T* x,
    const T* y,
    const T* w,
    std::size_t n,
    int degree,
    int deriv,
    int kernel_type,
    std::vector<T>& lx,
    std::vector<T>& ly,
    std::ptrdiff_t *left,
    std::ptrdiff_t *right,
    int *info
);

#endif