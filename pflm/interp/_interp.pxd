import numpy as np
cimport numpy as np

cdef void interp1d_memview_f64(
  np.float64_t[:] x,
  np.float64_t[:] y,
  np.float64_t[:] x_new,
  np.float64_t[:] y_new,
  int method = *
) noexcept nogil

cdef void interp1d_memview_f32(
  np.float32_t[:] x,
  np.float32_t[:] y,
  np.float32_t[:] x_new,
  np.float32_t[:] y_new,
  int method = *
) noexcept nogil

cdef void interp2d_memview_f64(
  np.float64_t[:] x,
  np.float64_t[:] y,
  np.float64_t[:, ::1] v,
  np.float64_t[:] x_new,
  np.float64_t[:] y_new,
  np.float64_t[:, ::1] v_new,
  int method = *
) noexcept nogil

cdef void interp2d_memview_f32(
  np.float32_t[:] x,
  np.float32_t[:] y,
  np.float32_t[:, ::1] v,
  np.float32_t[:] x_new,
  np.float32_t[:] y_new,
  np.float32_t[:, ::1] v_new,
  int method = *
) noexcept nogil

