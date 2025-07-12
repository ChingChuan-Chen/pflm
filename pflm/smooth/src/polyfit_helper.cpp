#include "polyfit_helper.h"

template <typename T>
T calculate_kernel_value(
    T u,
    int kernel_type
) {
    switch (kernel_type) {
        case 0: { // GAUSSIAN
            return inv_sqrt_2pi * std::exp(-0.5 * u * u);
        }
        case 1: { // LOGISTIC
            return 0.5 /  (std::cosh(u) + 1.0);
        }
        case 2: { // SIGMOID
            return 1.0 / std::cosh(u) / acos(-1.0);
        }
        // Shifted Gaussian kernel is not included since it might produce negative weights that are not supported in our implementation.
        // case 3: { // GAUSSIAN_VAR
        //     T u_sq = u * u;
        //     return inv_sqrt_2pi * std::exp(-0.5 * u_sq) * (1.25 - 0.25 * u_sq);
        // }
        // Silverman kernel is not included since it might produce negative weights that are not supported in our implementation.
        // case 4: { // SILVERMAN
        //     T temp = std::abs(u) * inv_sqrt_2;
        //     return 0.5 * std::exp(-0.5 * temp) * std::sin(temp + quarter_pi);
        // }
        case 100: { // RECTANGULAR
            return 0.5;
        }
        case 101: { // TRIANGULAR
            return (1.0 - std::abs(u));
        }
        case 102: { // EPANECHNIKOV
            return  0.75 * (1.0 - u * u);
        }
        case 103: { // BIWEIGHT
            return 15.0 / 16.0 * std::pow(1.0 - u * u, 2.0);
        }
        case 104: { // TRIWEIGHT
            return 35.0 / 32.0 * std::pow(1.0 - u * u, 3.0);
        }
        case 105: { // TRICUBE
            return 70.0 / 81.0 * std::pow(1.0 - std::pow(std::abs(u), 3.0), 3.0);
        }
        case 106: { // COSINE
            return quarter_pi * std::cos(half_pi * u);
        }
        default:
            return 0.0;
    }
}

double calculate_kernel_value_f64(double u, int kernel_type) {
    return calculate_kernel_value<double>(u, kernel_type);
}
float calculate_kernel_value_f32(float u, int kernel_type) {
    return calculate_kernel_value<float>(u, kernel_type);
}
