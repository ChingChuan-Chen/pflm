#ifndef __POLYFIT_HELPER_H__
#define __POLYFIT_HELPER_H__
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

const static double half_pi = std::acos(0.0);
const static double quarter_pi = half_pi / 2.0;
const static double inv_sqrt_2 = 1.0 / std::sqrt(2.0);
const static double inv_2pi = 1.0/(2.0*std::acos(-1.0));
const static double inv_sqrt_2pi = std::sqrt(inv_2pi);
const static double factorials[] = {1,1,2,6,24,120,720,5040,40320,362880,3628800};

template <typename T> T calculate_kernel_value(T, int);

double calculate_kernel_value_f64(double, int);
float calculate_kernel_value_f32(float, int);

#endif