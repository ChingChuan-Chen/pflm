#include "interp.h"
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <limits>


template <typename T>
void find_le_indices(const T* a, std::uint64_t n, const T* b, std::uint64_t m, std::int64_t* result) {
    std::int64_t i = 0, n2 = static_cast<std::int64_t>(n);
    for (std::uint64_t j = 0; j < m; ++j) {
        while (i < n2 && a[i] <= b[j]) {
            ++i;
        }

        // Special case for the largest element of b
        // If the largest element of b is equal to the last element of a,
        // we set the last element of b's index to the last element of a's index
        if (i < n2) {
            result[j] = i - 1;
        } else if (i == n2 && a[i - 1] == b[j]) {
            result[j] = n2 - 2;
        } else {
            result[j] = -1; // No valid index found
        }
    }
}

template <typename T>
void interp1d_linear(T x[], T y[], T x_new[], T y_new[], std::int64_t x_size, std::int64_t x_new_size) {
    std::int64_t i, j;
    std::int64_t *x_new_idx = new std::int64_t[x_new_size];
    find_le_indices<T>(x, x_size, x_new, x_new_size, x_new_idx);

    #pragma omp parallel for private(i)
    for (i = 0; i < x_new_size; ++i) {
        j = x_new_idx[i];
        if (j >= 0) {
            y_new[i] = y[j] + (x_new[i] - x[j]) * (y[j + 1] - y[j]) / (x[j + 1] - x[j]);
        }
        else {
            y_new[i] = std::numeric_limits<T>::quiet_NaN();
        }
    }
    delete[] x_new_idx;
}

template <typename T>
void interp1d_spline_small(T x[], T y[], T x_new[], T y_new[], std::int64_t x_size, std::int64_t x_new_size) {
    T ca, cb, cc;
    if (x_size == 2) {
        cc = 0.0;
        cb = (y[1] - y[0]) / (x[1] - x[0]);
        ca = y[0];
    } else {
        cc = (y[2] - y[0]) / (x[2] - x[0]) / (x[2] - x[1]) - (y[1] - y[0]) / (x[1] - x[0]) / (x[2] - x[1]);
        cb = (y[1] - y[0]) * (x[2] - x[0]) / (x[1] - x[0]) / (x[2] - x[1]) - (y[2] - y[0]) * (x[1] - x[0]) / (x[2] - x[0]) / (x[2] - x[1]);
        ca = y[0];
    }

    std::int64_t i;
    T s;
    #pragma omp parallel for private(i)
    for (i = 0; i < x_new_size; ++i) {
        // small size, so no need to use binary search
        if ((x_new[i] < x[0]) || (x_new[i] > x[x_size - 1])) {
            y_new[i] = std::numeric_limits<T>::quiet_NaN();
        } else {
            s = x_new[i] - x[0];
            y_new[i] = ca + s * cb + std::pow(s, 2.0) * cc;
        }
    }
}

// Thomas algorithm for solving tri-diagonal systems of equations Ax = d
// where A is a tri-diagonal matrix with sub-diagonal ldg, diagonal dg, and super-diagonal udg
template <typename T>
void thomas_algorithm(T ldg[], T dg[], T udg[], T g[], T x[], std::int64_t n) {
    std::int64_t i;
    T p;
    // Forward substitution
    for (i = 1; i < n; ++i) {
        p = ldg[i] / dg[i - 1];
        dg[i] -= p * udg[i - 1];
        g[i] -= p * g[i - 1];
    }

    // Back substitution
    x[n - 1] = g[n - 1] / dg[n - 1];
    for (i = n - 2; i >= 0; i--) {
        x[i] = (g[i] - udg[i] * x[i + 1]) / dg[i];
    }
}

template <typename T>
void interp1d_spline(T x[], T y[], T x_new[], T y_new[], std::int64_t x_size, std::int64_t x_new_size) {
    T *h = new T[x_size - 1], *ca = new T[x_size - 1], *cb = new T[x_size - 1], *cc = new T[x_size], *cd = new T[x_size - 1], *g = new T[x_size - 1];
    std::int64_t i, j;

    #pragma omp parallel for private(i)
    for (i = 0; i < x_size - 1; ++i) {
        cb[i] = y[i + 1] - y[i];
        h[i] = x[i + 1] - x[i];
    }
    std::copy(y, y + x_size - 1, ca);

    g[0] = 3.0 / (h[0] + h[1]) * (cb[1] - h[1] / h[0] * cb[0]);
    g[x_size - 3] = 3.0 / (h[x_size - 3] + h[x_size - 2]) * (h[x_size - 3] / h[x_size - 2] * cb[x_size - 2] - cb[x_size - 3]);
    if (x_size > 4) {
        #pragma omp parallel for private(i)
        for (i = 1; i < x_size - 3; ++i) {
            g[i] = 3.0 * cb[i + 1] / h[i + 1] - 3.0 * cb[i] / h[i];
        }

        T *dg = new T[x_size - 2], *ldg = new T[x_size - 2], *udg = new T[x_size - 2], *tmp = new T[x_size - 2];
        ldg[0] = 0.0;
        udg[x_size - 3] = 0.0;
        #pragma omp parallel for private(i)
        for (i = 0; i < x_size - 3; ++i)
        {
            dg[i] = 2.0 * (h[i] + h[i + 1]);
            ldg[i + 1] = h[i + 1];
            udg[i] = h[i + 1];
        }
        dg[0] -= h[0];
        dg[x_size - 3] = 2.0 * h[x_size - 3] + h[x_size - 2];
        udg[0] -= h[0];
        ldg[x_size - 3] -= h[x_size - 2];

        thomas_algorithm<T>(ldg, dg, udg, g, tmp, x_size - 2);
        std::copy(tmp, tmp + x_size - 2, cc + 1);

        delete[] dg;
        delete[] ldg;
        delete[] udg;
        delete[] tmp;
    } else {
        // Special case for n = 4
        T det = 3 * h[0] * h[1] + 3 * h[1] * h[2] + 3 * h[1] * h[1];
        cc[1] = (2 * g[0] * h[1] + g[0] * h[2] + g[1] * h[0] - g[1] * h[1]) / det;
        cc[2] = (2 * g[1] * h[1] + g[1] * h[0] + g[0] * h[2] - g[0] * h[1]) / det;
    }

    cc[0] = cc[1] + h[0] / h[1] * (cc[1] - cc[2]);
    cc[x_size - 1] = cc[x_size - 2] + h[x_size - 2] / h[x_size - 3] * (cc[x_size - 2] - cc[x_size - 3]);

    #pragma omp parallel for private(i)
    for (i = 0; i < x_size - 1; ++i)
    {
        cb[i] /= h[i];
        cb[i] -= h[i] * (cc[i + 1] + 2.0 * cc[i]) / 3.0;
        cd[i] = (cc[i + 1] - cc[i]) / (3.0 * h[i]);
    }

    std::int64_t *x_new_idx = new std::int64_t[x_new_size];
    find_le_indices<T>(x, x_size, x_new, x_new_size, x_new_idx);

    // Calculate the interpolated values
    #pragma omp parallel for private(i)
    for (i = 0; i < x_new_size; ++i) {
        j = x_new_idx[i];
        if (j >= 0 && j < x_size - 1) {
            T s = x_new[i] - x[j];
            y_new[i] = ca[j] + s * cb[j] + std::pow(s, 2.0) * cc[j] + std::pow(s, 3.0) * cd[j];
        } else {
            y_new[i] = std::numeric_limits<T>::quiet_NaN();
        }
    }

    delete[] h;
    delete[] ca;
    delete[] cb;
    delete[] cc;
    delete[] cd;
    delete[] g;
    delete[] x_new_idx;
}

template <typename T>
void interp2d_linear(
    T x[], T y[], T v[], T x_new[], T y_new[], T v_new[],
    std::int64_t x_size, std::int64_t y_size,
    std::int64_t x_new_size, std::int64_t y_new_size
) {
    std::int64_t i;
    std::int64_t *x_new_idx = new std::int64_t[x_new_size];
    std::int64_t *y_new_idx = new std::int64_t[y_new_size];
    find_le_indices<T>(x, x_size, x_new, x_new_size, x_new_idx);
    find_le_indices<T>(y, y_size, y_new, y_new_size, y_new_idx);

    #pragma omp parallel for private(i)
    for (i = 0; i < x_new_size; ++i) {
        std::int64_t x_idx = x_new_idx[i];
        for (std::int64_t j = 0; j < y_new_size; ++j) {
            std::int64_t y_idx = y_new_idx[j];

            if (x_idx < 0 || y_idx < 0) {
                v_new[j * x_new_size + i] = std::numeric_limits<double>::quiet_NaN();
            } else {
                // perform the linear interpolation on the grid
                T dx = (x_new[i] - x[x_idx]) / (x[x_idx + 1] - x[x_idx]);
                T dy = (y_new[j] - y[y_idx]) / (y[y_idx + 1] - y[y_idx]);
                v_new[j * x_new_size + i] = v[y_idx * x_size + x_idx] +
                dx * (v[y_idx * x_size + x_idx + 1] - v[y_idx * x_size + x_idx]) +
                dy * (v[(y_idx + 1)* x_size +x_idx] - v[y_idx * x_size + x_idx]) +
                dx * dy * (v[(y_idx + 1)* x_size +x_idx + 1] - v[(y_idx + 1)* x_size + x_idx] - v[y_idx* x_size + x_idx + 1] + v[y_idx* x_size + x_idx]);
            }
        }
    }
    delete[] x_new_idx;
    delete[] y_new_idx;
}

// transpose a matrix of size n_rows x n_cols from src to dst with OpenMP parallelization
template <typename T>
void transpose_matrix(T *src, T *dst, int64_t n_rows, int64_t n_cols) {
  int64_t i, j;
  #pragma omp parallel for private(i, j)
  for (i = 0; i < n_rows; ++i) {
    for (j = 0; j < n_cols; ++j) {
      dst[j * n_rows + i] = src[i * n_cols + j];
    }
  }
}

template <typename T>
void interp2d_spline(
    T x[], T y[], T v[], T x_new[], T y_new[], T v_new[],
    std::int64_t x_size, std::int64_t y_size,
    std::int64_t x_new_size, std::int64_t y_new_size
) {
    std::int64_t i;
    T *tp_v = new T[y_size * x_size], *temp1 = new T[y_new_size * x_size], *temp2 = new T[y_new_size * x_size];

    // we need to transpose the matrix first for the spline interpolation on each row
    transpose_matrix<T>(v, tp_v, y_size, x_size);

    // do the spline interpolation on x axis first
    for (i = 0; i < x_size; ++i) {
        if (y_size <= 3) {
            interp1d_spline_small<T>(&y[0], &tp_v[i * y_size], &y_new[0], &temp1[i * y_new_size], y_size, y_new_size);
        } else {
            interp1d_spline<T>(&y[0], &tp_v[i * y_size], &y_new[0], &temp1[i * y_new_size], y_size, y_new_size);
        }
    }

    // we need to transpose the temp results from the spline interpolation on x axis to do the spline interpolation on y axis
    transpose_matrix<T>(temp1, temp2, x_size, y_new_size);

    // do the spline interpolation on y axis next
    for (i = 0; i < y_new_size; ++i) {
        if (x_size <= 3) {
            interp1d_spline_small<T>(&x[0], &temp2[i * x_size], &x_new[0], &v_new[i * x_new_size], x_size, x_new_size);
        } else {
            interp1d_spline<T>(&x[0], &temp2[i * x_size], &x_new[0], &v_new[i * x_new_size], x_size, x_new_size);
        }
    }
    delete[] temp1;
    delete[] temp2;
}

