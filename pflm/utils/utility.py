"""Utility functions for flattening data matrices and computing eigenvalues and eigenvectors"""

# Authors: Ching-Chuan Chen
# SPDX-License-Identifier: MIT
from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
from sklearn.utils.validation import check_array

from pflm.utils._trapz import trapz_f32, trapz_f64


def trapz(y: np.ndarray, x: np.ndarray) -> Union[np.ndarray, float]:
    """
    Compute the integrated area value with trapezoidal rule.

    Parameters
    ----------
    y : array_like
        1D or 2D array of function values with respect to x.
        The expected shape could be (n_samples, n_features) or (n_features, n_samples).
        If n_samples is equal to n_features, we will use (n_samples, n_features) as the shape.
    x : array_like
        1D array of x-coordinates corresponding to the function values. The expected shape is (n_features,).

    Returns
    -------
    trapz_value : np.ndarray
        The integrated area value computed using the trapezoidal rule with shape (n_samples,).
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
        Integer indices mapping each t to its position in `unique_tid`.
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
    """Flatten and sort the data matrices.

    This function takes a list of response matrices `y` and corresponding time points `t`,
    and optionally weights `w`, and flattens them into 1D arrays. The function handles NaN values by excluding them
    from the output. If all values in `y` are NaN, it raises a ValueError.

    Parameters
    ----------
    y : list of array_like
        The response list, where each element is a 1D array of shape (nt_i,), i= 0, 1, ..., n-1.
    t : list of array_like
        The time points, where each element is a 1D array of shape (nt_i,), i= 0, 1, ..., n-1.
        It should be a 1D array where each element corresponds to a time point.
    input_dtype : str or np.dtype, optional
        The data type of the input arrays. Defaults to np.float64.
    w : array_like, optional
        The weights for each sample. If provided, it should have the same length as `y` and `t`.

    Returns
    -------
    tt : np.ndarray
        A 1D array of the corresponding time points, sorted to match `yy`.
    yy : np.ndarray
        A 1D array of the flattened response values, sorted by sample index and time points.
    ww : np.ndarray
        A 1D array of the weights corresponding to `yy`, sorted to match `yy`.
    sid : np.ndarray
        A 1D array of sample indices, where each index corresponds to the sample in `y`.
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
