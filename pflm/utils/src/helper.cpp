#include "helper.h"

template <typename T>
std::ptrdiff_t search_upper_bound(T sorted_array[], std::ptrdiff_t array_size, T target, bool inclusive = true) {
  std::ptrdiff_t left = 0, right = array_size - 1, mid;
  while (left <= right) {
    mid = left + (right - left) / 2;
    if ((sorted_array[mid] > target) || (inclusive && sorted_array[mid] == target)) {
      right = mid - 1;
    } else {
      left = mid + 1;
    }
  }
  return left < array_size ? left : -1;
}

template <typename T>
std::ptrdiff_t search_lower_bound(T sorted_array[], std::ptrdiff_t array_size, T target, bool inclusive = true) {
  std::ptrdiff_t left = 0, right = array_size - 1, mid;
  while (left <= right) {
    mid = left + (right - left) / 2;
    if ((sorted_array[mid] < target) || (inclusive && sorted_array[mid] == target)) {
      left = mid + 1;
    } else {
      right = mid - 1;
    }
  }
  return right >= 0 ? right : -1;
}

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

// transpose a matrix of size n_rows x n_cols from src to dst with OpenMP parallelization
template <typename T>
void transpose_matrix(T *src, T *dst, ptrdiff_t n_rows, ptrdiff_t n_cols) {
  ptrdiff_t i, j;
  #pragma omp parallel for private(i, j)
  for (i = 0; i < n_rows; ++i) {
    for (j = 0; j < n_cols; ++j) {
      dst[j * n_rows + i] = src[i * n_cols + j];
    }
  }
}

template std::ptrdiff_t search_upper_bound<double>(double sorted_array[], std::ptrdiff_t array_size, double target, bool inclusive);
template std::ptrdiff_t search_upper_bound<float>(float sorted_array[], std::ptrdiff_t array_size, float target, bool inclusive);
template std::ptrdiff_t search_lower_bound<double>(double sorted_array[], std::ptrdiff_t array_size, double target, bool inclusive);
template std::ptrdiff_t search_lower_bound<float>(float sorted_array[], std::ptrdiff_t array_size, float target, bool inclusive);
template std::ptrdiff_t search_location<double>(double sorted_array[], std::ptrdiff_t array_size, double target);
template std::ptrdiff_t search_location<float>(float sorted_array[], std::ptrdiff_t array_size, float target);
template void transpose_matrix<double>(double*, double*, std::ptrdiff_t, std::ptrdiff_t);
template void transpose_matrix<float>(float*, float*, std::ptrdiff_t, std::ptrdiff_t);
