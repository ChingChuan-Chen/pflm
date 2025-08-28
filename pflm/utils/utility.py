"""Utility functions for flattening data matrices and computing eigenvalues and eigenvectors"""

# Authors: Ching-Chuan Chen
# SPDX-License-Identifier: MIT
from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
from sklearn.utils.validation import check_array

from pflm.utils.trapz import trapz_f32, trapz_f64


def trapz(y: np.ndarray, x: np.ndarray) -> Union[np.ndarray, float]:
    """
    Compute the integrated area using the trapezoidal rule.

    Parameters
    ----------
    y : array_like
        1D or 2D array of function values with respect to `x`.
        Accepted shapes:
        - (n_features,) for a single curve.
        - (n_samples, n_features) for multiple curves.
        If `y` is 1D, it is treated as a single row (1, n_features).
    x : array_like of shape (n_features,)
        1D array of x-coordinates corresponding to the function values.

    Returns
    -------
    np.ndarray or float
        If `y` was 1D, returns a scalar float.
        If `y` was 2D, returns a 1D array of shape (n_samples,) with the integral
        per row of `y`.

    Raises
    ------
    ValueError
        If the number of points in `x` does not match either the number of rows
        or the number of columns of `y`.

    Mathematical definition
    -----------------------
    For a single curve ``y`` of length ``n`` and ``x`` of the same length:

    .. math::
        T(y, x) = \sum_{i=0}^{n-2} \frac{x_{i+1}-x_i}{2}\,\big(y_i + y_{i+1}\big).

    For a matrix ``Y`` of shape ``(m, n)`` with ``len(x) = n`` (integrate along axis 1):

    .. math::
        [T(Y, x)]_k = \sum_{i=0}^{n-2} \frac{x_{i+1}-x_i}{2}\,\big(Y_{k,i} + Y_{k,i+1}\big),
        \quad k=0,\dots,m-1.

    If instead ``len(x) = m`` (integrate along axis 0):

    .. math::
        [T(Y, x)]_k = \sum_{i=0}^{m-2} \frac{x_{i+1}-x_i}{2}\,\big(Y_{i,k} + Y_{i+1,k}\big),
        \quad k=0,\dots,n-1.

    Notes
    -----
    - The implementation dispatches to a float32/float64 optimized backend.
    - `x` is coerced to the dtype of `y` to avoid unintended up/down-casts.

    See Also
    --------
    numpy.trapz : Reference implementation for simple cases.
    """
    if y.ndim == 1:
        y = y.reshape((1, -1))  # Ensure y is 2D for consistency
    if y.shape[1] != x.shape[0] and y.shape[0] != x.shape[0]:
        raise ValueError("The number of columns or rows of y must match the size of x.")
    if y.dtype not in [np.float64, np.float32]:
        y = y.astype(np.float64, copy=False)  # Convert to float64 if not already

    # force x to match y's dtype
    x = x.astype(y.dtype, copy=False)
    trapz_func = trapz_f64 if y.dtype == np.float64 else trapz_f32
    trapz_result = trapz_func(y, x)
    if y.shape[0] == 1:
        return trapz_result[0]
    else:
        return trapz_result


@dataclass
class FlattenFunctionalData:
    """Flattened functional dataset in 1D arrays with indexing helpers.

    Attributes
    ----------
    y : np.ndarray of shape (M,)
        Flattened responses after removing NaN.
    t : np.ndarray of shape (M,)
        Flattened time points aligned to `y`.
    w : np.ndarray of shape (M,)
        Per-observation weights expanded from sample-level weights.
    tid : np.ndarray of shape (M,)
        Integer indices mapping each `t` to its position in `unique_tid`.
    unique_tid : np.ndarray of shape (nt,)
        Sorted unique time points (observation grid).
    inverse_tid_idx : np.ndarray of shape (M,)
        Inverse mapping indices from `t` back to `unique_tid`.
    sid : np.ndarray of shape (M,)
        Sample id for each observation (0-based).
    unique_sid : np.ndarray of shape (n_samples,)
        Unique sample ids present in the data.
    sid_cnt : np.ndarray of shape (n_samples,)
        Number of observations for each sample id.
    """

    y: np.ndarray
    t: np.ndarray
    w: np.ndarray
    tid: np.ndarray
    unique_tid: np.ndarray
    inverse_tid_idx: np.ndarray
    sid: np.ndarray
    unique_sid: np.ndarray
    sid_cnt: np.ndarray


def flatten_and_sort_data_matrices(
    y: List[np.ndarray],
    t: List[np.ndarray],
    input_dtype: Union[str, np.dtype] = np.float64,
    w: Optional[np.ndarray] = None,
) -> FlattenFunctionalData:
    """Flatten per-sample 1D arrays into contiguous vectors and build indices.

    This function concatenates lists of responses `y` and times `t`, expands
    per-sample weights `w` to observation level, drops NaNs, and constructs
    indexing helpers for the observation grid and subject ids.

    Parameters
    ----------
    y : list of np.ndarray
        Each element is a 1D array of shape (nt_i,) with responses for sample i.
    t : list of np.ndarray
        Each element is a 1D array of shape (nt_i,) with time points for sample i.
    input_dtype : str or np.dtype, default=np.float64
        Target dtype for numeric arrays.
    w : np.ndarray, optional
        1D array of length len(y) with per-sample weights. If None, uses ones.

    Returns
    -------
    FlattenFunctionalData
        Dataclass holding flattened arrays (y, t, w), grid/subject indices
        (tid, unique_tid, inverse_tid_idx, sid, unique_sid, sid_cnt).

    Raises
    ------
    ValueError
        If `y`/`t` are not lists of 1D arrays with matching lengths,
        if `w` is provided but invalid, or if all `y` values are NaN.

    Notes
    -----
    - NaN entries in `y` (and matching positions in `t`) are removed.
    - `unique_tid` is constructed from the de-duplicated sorted values of `t`.
    - The `tid` indices are built via `np.digitize` against `unique_tid`.
    """
    if not isinstance(y, list):
        raise ValueError("y must be a list of arrays.")
    if not isinstance(t, list):
        raise ValueError("t must be a list of arrays.")
    if len(y) != len(t):
        raise ValueError("The length of y and t must be the same.")
    for yi, ti in zip(y, t):
        if yi.ndim != 1 or ti.ndim != 1:
            raise ValueError("Each element of y and t must be a 1D array.")
        if yi.size != ti.size:
            raise ValueError("Each element of y and t must have the same length.")

    if w is None:
        w = np.ones((len(y),), dtype=input_dtype)
    else:
        if not isinstance(w, np.ndarray):
            raise ValueError("Weights w must be a 1D array.")
        if len(y) != w.size:
            raise ValueError("The length of y and w must be the same.")
        w = check_array(w, ensure_2d=False, dtype=input_dtype)
        if w.ndim != 1:
            raise ValueError("Each element of w must be a 1D array.")

    yy = np.concatenate(y).astype(input_dtype, copy=False)
    non_nan_mask = ~np.isnan(yy)
    if not non_nan_mask.any():
        raise ValueError("All values in y are NaN. Cannot flatten data matrices.")

    tt = np.concatenate(t).astype(input_dtype, copy=False)[non_nan_mask]
    ww = np.concatenate([np.full(yi.size, wi, dtype=input_dtype) for yi, wi in zip(y, w)])[non_nan_mask]
    sid = np.concatenate([np.full(yi.size, i, dtype=np.int64) for i, yi in enumerate(y)])[non_nan_mask]

    unique_tid, inverse_tid_idx = np.unique(tt, return_inverse=True, sorted=True)
    tid = np.digitize(tt, unique_tid, right=True)
    unique_sid, sid_cnt = np.unique(sid, return_counts=True)

    return FlattenFunctionalData(yy[non_nan_mask], tt, ww, tid, unique_tid, inverse_tid_idx, sid, unique_sid, sid_cnt)
