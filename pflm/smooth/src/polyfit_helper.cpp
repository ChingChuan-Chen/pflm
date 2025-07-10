#include "polyfit_helper.h"

template <typename T>
std::ptrdiff_t search_lower_bound(T sorted_array[], std::ptrdiff_t array_size, T target, bool right_inclusive = true) {
  std::ptrdiff_t left = 0, right = array_size - 1, mid;
  while (left <= right) {
    mid = left + (right - left) / 2;
    if ((sorted_array[mid] < target) || ((right_inclusive > 0) && sorted_array[mid] == target)) {
      left = mid + 1;
    } else {
      right = mid - 1;
    }
  }
  return left < array_size ? left : -1;
}

template std::ptrdiff_t search_lower_bound<double>(double sorted_array[], std::ptrdiff_t array_size, double target, bool right_inclusive);
template std::ptrdiff_t search_lower_bound<float>(float sorted_array[], std::ptrdiff_t array_size, float target, bool right_inclusive);

template <typename T>
std::ptrdiff_t search_location(T sorted_array[], std::ptrdiff_t array_size, T target) {
  std::ptrdiff_t left = 0, right = array_size - 1, mid;
  while (left <= right) {
    mid = left + (right - left) / 2;
    if (sorted_array[mid] == target) {
      return mid; // found the target
    } else if (sorted_array[mid] < target) {
      left = mid + 1;
    } else {
      right = mid - 1;
    }
  }
  return -1; // target not found
}

template std::ptrdiff_t search_location<double>(double sorted_array[], std::ptrdiff_t array_size, double target);
template std::ptrdiff_t search_location<float>(float sorted_array[], std::ptrdiff_t array_size, float target);

template <typename T>
T calculate_sqrt_kernel_value(
    T u,
    int kernel_type,
    T wj
) {
    switch (kernel_type) {
        case 0: { // GAUSSIAN
            return std::sqrt(wj * inv_sqrt_2pi * std::exp(-0.5 * u * u));
        }
        case 1: { // LOGISTIC
            return std::sqrt(wj * 1.0 / (std::exp(u) + 2.0 + std::exp(-u)));
        }
        case 2: { // SIGMOID
            return std::sqrt(wj / half_pi / (std::exp(u) + std::exp(-u)));
        }
        case 3: { // SILVERMAN
            T tmp = std::abs(u) * inv_sqrt_2;
            return std::sqrt(wj * 0.5 * std::exp(-tmp) * sin(tmp + quarter_pi));
        }
        case 4: { // GAUSSIAN_VAR
            T u_sq = u * u;
            if (u_sq >= 5.0) {
                return 0.0; // Return zero for large u to avoid overflow
            }
            return std::sqrt(wj * inv_sqrt_2pi * std::exp(-0.5 * u_sq) * (1.25 - 0.25 * u_sq));
        }
        case 5: { // RECTANGULAR
            return std::sqrt(wj * 0.5);
        }
        case 6: { // TRIANGULAR
            return std::sqrt(wj * (1.0 - std::abs(u)));
        }
        case 7: { // EPANECHNIKOV
            return std::sqrt(wj * 0.75 * (1.0 - u * u));
        }
        case 8: { // BIWEIGHT
            return std::sqrt(wj * 15.0 / 16.0 * std::pow(1.0 - u * u, 2.0));
        }
        case 9: { // TRIWEIGHT
            return std::sqrt(wj * 35.0 / 32.0 * std::pow(1.0 - u * u, 3.0));
        }
        case 10: { // TRICUBE
            return std::sqrt(wj * 70.0 / 81.0 * std::pow(1.0 - std::pow(std::abs(u), 3.0), 3.0));
        }
        case 11: { // COSINE
            return std::sqrt(wj * quarter_pi * std::cos(half_pi * u));
        }
        default:
            return 0.0; // Unknown kernel type, return zero
    }
}

double calculate_sqrt_kernel_value_f64(double u, int kernel_type, double wj) {
    return calculate_sqrt_kernel_value<double>(u, kernel_type, wj);
}
float calculate_sqrt_kernel_value_f32(float u, int kernel_type, float wj) {
    return calculate_sqrt_kernel_value<float>(u, kernel_type, wj);
}

template <typename T>
void polyfit1d_prepare(
    T bw,
    T center,
    T* x,
    T* y,
    T* w,
    std::ptrdiff_t n,
    int degree,
    int deriv,
    int kernel_type,
    std::vector<T>& lx,
    std::vector<T>& ly,
    std::ptrdiff_t *left,
    std::ptrdiff_t *right,
    int *info
) {
    std::ptrdiff_t n_rows = n;
    if (kernel_type >= 5) {
        *left = search_lower_bound<T>(x, n, center - bw);
        *right = search_lower_bound<T>(x, n, center + bw);
        if ((*left <= 0) || (*left == *right)) {
            *info = -1; // no data in the window
            return;
        }
        if (*left == -1) {
            *left = 0;
        }
        if (*right == -1) {
            *right = n;
        }
        if ((*left > *right) || (*right - *left <= deriv)) {
            *info = -2; // invalid window
            return;
        }
        n_rows = (*right - *left);
    } else {
        *left = 0;
        *right = n;
    }

    lx.assign(n_rows * (degree + 1), 0.0);
    ly.assign(n_rows, 0.0);

    std::ptrdiff_t i = 0;
    for (std::ptrdiff_t j = *left; j < *right; ++j) {
        T center_minus_xj = center - x[j];
        T u = center_minus_xj / bw;
        T sqrt_wj = calculate_sqrt_kernel_value<T>(u, kernel_type, w[j]);
        lx[i] = sqrt_wj;
        for (int k = 0; k < degree; ++k) {
            lx[i + n_rows * (k + 1)] = std::pow(center_minus_xj, k + 1) * sqrt_wj;
        }
        ly[i] = y[j] * sqrt_wj;
        ++i;
    }
}

template void polyfit1d_prepare<double>(
    double bw,
    double center,
    double* x,
    double* y,
    double* w,
    std::ptrdiff_t n,
    int degree,
    int deriv,
    int kernel_type,
    std::vector<double>& lx,
    std::vector<double>& ly,
    std::ptrdiff_t *left,
    std::ptrdiff_t *right,
    int *info
);

template void polyfit1d_prepare<float>(
    float bw,
    float center,
    float* x,
    float* y,
    float* w,
    std::ptrdiff_t n,
    int degree,
    int deriv,
    int kernel_type,
    std::vector<float>& lx,
    std::vector<float>& ly,
    std::ptrdiff_t *left,
    std::ptrdiff_t *right,
    int *info
);
