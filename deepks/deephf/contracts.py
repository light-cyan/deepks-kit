"""Canonical value, array, fingerprint, and control contracts for DeePHF."""

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
import hashlib
from numbers import Real
import operator
from typing import Any

import numpy as np
import torch

from deepks.array_utils import immutable_array


def update_digest(digest, value: Any) -> None:
    """Encode supported scientific values into an incremental digest."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(repr(array.shape).encode("ascii"))
        digest.update(memoryview(array).cast("B"))
        return
    if isinstance(value, np.generic):
        update_digest(digest, value.item())
        return
    if is_dataclass(value) and not isinstance(value, type):
        for item in fields(value):
            digest.update(item.name.encode("utf-8"))
            update_digest(digest, getattr(value, item.name))
        return
    if isinstance(value, Mapping):
        for key in sorted(value, key=lambda item: (type(item).__name__, repr(item))):
            update_digest(digest, key)
            update_digest(digest, value[key])
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            update_digest(digest, item)
        return
    if value is None or isinstance(value, (bool, int, float, str)):
        digest.update(type(value).__name__.encode("ascii"))
        digest.update(repr(value).encode("utf-8"))
        return
    raise TypeError(
        f"cannot fingerprint scientific value of type {type(value).__name__}"
    )


def array_fingerprint(value: np.ndarray) -> str:
    """Hash one array without materializing its contiguous buffer as bytes."""
    digest = hashlib.sha256()
    update_digest(digest, np.asarray(value))
    return digest.hexdigest()


def dataclass_fingerprint(value, *, excluded=frozenset()) -> str:
    """Hash all fields of one dataclass except explicitly excluded fields."""
    digest = hashlib.sha256()
    for item in fields(value):
        if item.name in excluded:
            continue
        digest.update(item.name.encode("utf-8"))
        update_digest(digest, getattr(value, item.name))
    return digest.hexdigest()


def validated_float64_array(value, expected_shape, name: str, error_type=ValueError):
    """Validate one finite real float64 array without copying it."""
    try:
        array = np.asarray(value)
    except Exception as error:
        raise error_type(f"{name} is not a numerical array: {error}") from error
    if array.shape != expected_shape:
        raise error_type(
            f"unexpected {name} shape {array.shape}; expected {expected_shape}"
        )
    if array.dtype != np.dtype(np.float64) or np.iscomplexobj(array):
        raise error_type(f"{name} must be a real float64 array")
    if not np.isfinite(array).all():
        raise error_type(f"{name} must be finite")
    return array


def integer_control(value, name: str, *, prefix="response") -> int:
    """Validate an integer-valued solver control."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{prefix} {name} must be an integer")
    try:
        return operator.index(value)
    except TypeError as error:
        raise ValueError(f"{prefix} {name} must be an integer") from error


def real_control(value, name: str, *, prefix="response") -> float:
    """Validate a finite real-valued solver control."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{prefix} {name} must be a real numeric scalar")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{prefix} {name} must be finite")
    return result


def version_series(version: str, error_type=ValueError) -> tuple[int, int]:
    """Return the major/minor pair from one version string."""
    components = version.split(".")
    try:
        return int(components[0]), int(components[1])
    except (IndexError, ValueError) as error:
        raise error_type(f"cannot interpret the PySCF version {version!r}") from error


__all__ = [
    "array_fingerprint",
    "dataclass_fingerprint",
    "immutable_array",
    "integer_control",
    "real_control",
    "update_digest",
    "validated_float64_array",
    "version_series",
]
