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


class RootContinuityError(RuntimeError):
    """Raised when adjacent references do not define one continuous root."""


def validate_root_overlap_tolerance(value, *, owner="root") -> float:
    """Return one finite occupied-space overlap threshold in ``(0, 1]``."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{owner} root_overlap_tolerance must be a real number")
    try:
        tolerance = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(
            f"{owner} root_overlap_tolerance must be a real number"
        ) from error
    if not np.isfinite(tolerance) or tolerance <= 0.0 or tolerance > 1.0:
        raise ValueError(
            f"{owner} root_overlap_tolerance must be finite and in (0, 1]"
        )
    return tolerance


def occupied_coefficients(mo_coeff, mo_occ) -> tuple[np.ndarray, ...]:
    """Extract restricted or spin-resolved occupied coefficient matrices."""
    coefficients = np.asarray(mo_coeff)
    occupations = np.asarray(mo_occ)
    if np.iscomplexobj(coefficients) or np.iscomplexobj(occupations):
        raise ValueError("root tracking requires real orbital data")
    if coefficients.ndim == 2 and occupations.ndim == 1:
        channel_values = ((coefficients, occupations),)
    elif (
        coefficients.ndim == 3
        and occupations.ndim == 2
        and coefficients.shape[0] == occupations.shape[0] == 2
    ):
        channel_values = tuple(zip(coefficients, occupations))
    else:
        raise ValueError(
            "root tracking requires restricted orbitals or two spin channels"
        )
    channels = []
    for channel_index, (channel_coefficients, channel_occupations) in enumerate(
        channel_values
    ):
        if (
            channel_coefficients.ndim != 2
            or channel_occupations.ndim != 1
            or channel_coefficients.shape[1] != channel_occupations.shape[0]
            or not np.isfinite(channel_coefficients).all()
            or not np.isfinite(channel_occupations).all()
        ):
            raise ValueError(
                f"root tracking orbital channel {channel_index} is invalid"
            )
        channels.append(
            np.ascontiguousarray(
                channel_coefficients[:, channel_occupations > 0],
                dtype=np.float64,
            )
        )
    return tuple(channels)


def occupied_subspace_overlaps(
    previous_molecule,
    previous_occupied,
    candidate_molecule,
    candidate_occupied,
) -> tuple[float, ...]:
    """Return the minimum singular overlap for each occupied spin channel."""
    from pyscf import gto

    previous_channels = tuple(np.asarray(value) for value in previous_occupied)
    candidate_channels = tuple(np.asarray(value) for value in candidate_occupied)
    if len(previous_channels) not in (1, 2) or len(candidate_channels) != len(
        previous_channels
    ):
        raise ValueError("root tracking occupied channels are incompatible")
    previous_ao_count = int(previous_molecule.nao_nr())
    candidate_ao_count = int(candidate_molecule.nao_nr())
    for channel_index, (previous, candidate) in enumerate(
        zip(previous_channels, candidate_channels)
    ):
        if (
            previous.ndim != 2
            or candidate.ndim != 2
            or previous.shape[0] != previous_ao_count
            or candidate.shape[0] != candidate_ao_count
            or previous.shape[1] != candidate.shape[1]
            or np.iscomplexobj(previous)
            or np.iscomplexobj(candidate)
            or not np.isfinite(previous).all()
            or not np.isfinite(candidate).all()
        ):
            raise ValueError(
                f"root tracking occupied channel {channel_index} is incompatible"
            )
    try:
        cross_overlap = np.asarray(
            gto.intor_cross(
                "int1e_ovlp",
                previous_molecule,
                candidate_molecule,
            )
        )
    except Exception as error:
        raise RootContinuityError(
            f"cross-geometry AO overlap construction failed: {error}"
        ) from error
    if (
        cross_overlap.shape != (previous_ao_count, candidate_ao_count)
        or np.iscomplexobj(cross_overlap)
        or not np.isfinite(cross_overlap).all()
    ):
        raise RootContinuityError("cross-geometry AO overlap is invalid")
    overlaps = []
    for channel_index, (previous, candidate) in enumerate(
        zip(previous_channels, candidate_channels)
    ):
        if previous.shape[1] == 0:
            overlaps.append(1.0)
            continue
        occupied_overlap = previous.T @ cross_overlap @ candidate
        try:
            singular_values = np.linalg.svd(occupied_overlap, compute_uv=False)
        except np.linalg.LinAlgError as error:
            raise RootContinuityError(
                f"occupied-subspace overlap SVD failed for channel {channel_index}: {error}"
            ) from error
        minimum = float(np.min(singular_values))
        if not np.isfinite(minimum):
            raise RootContinuityError(
                f"occupied-subspace overlap is nonfinite for channel {channel_index}"
            )
        overlaps.append(float(np.clip(minimum, 0.0, 1.0)))
    return tuple(overlaps)


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
    "RootContinuityError",
    "array_fingerprint",
    "dataclass_fingerprint",
    "immutable_array",
    "integer_control",
    "occupied_coefficients",
    "occupied_subspace_overlaps",
    "real_control",
    "update_digest",
    "validated_float64_array",
    "validate_root_overlap_tolerance",
    "version_series",
]
