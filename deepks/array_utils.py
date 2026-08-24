"""Shared NumPy array ownership helpers."""

import numpy as np


def immutable_array(value, *, dtype=None) -> np.ndarray:
    """Return one owned contiguous immutable array."""
    array = np.array(value, dtype=dtype, copy=True, order="C", subok=False)
    array.setflags(write=False)
    return array


__all__ = ["immutable_array"]
