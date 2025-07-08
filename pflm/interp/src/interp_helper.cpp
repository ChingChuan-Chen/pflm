#include "interp_helper.h"

// Binary search for the index of the first element in sorted_array that is greater than **or equal to** the target
template <typename T>
std::ptrdiff_t search_sorted(T sorted_array[], std::ptrdiff_t array_size, T target) {
  std::ptrdiff_t left = 0, right = array_size - 1, mid;
  while (left <= right) {
    mid = left + (right - left) / 2;
    if (sorted_array[mid] <= target) {
      left = mid + 1;
    } else {
      right = mid - 1;
    }
  }
  if (left < array_size) {
    return left;
  } else if ((left == array_size) && (sorted_array[left - 1] == target)) {
    return left - 1;
  } else {
    return -1;
  }
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
