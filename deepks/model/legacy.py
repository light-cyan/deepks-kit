"""Safe import helpers for legacy DeePKS CorrNet checkpoints."""

from __future__ import annotations

from collections.abc import Mapping
from io import BytesIO
from importlib import import_module

import numpy as np
import torch

from .model import CHECKPOINT_FORMAT_VERSION, CorrNet, _as_checkpoint_metadata


def _legacy_numpy_safe_globals() -> list:
    """Return the narrow NumPy allowlist required by old metadata pickles."""
    multiarray = import_module("numpy.core.multiarray")
    reconstruct = multiarray._reconstruct
    scalar = multiarray.scalar
    return [
        (reconstruct, "numpy.core.multiarray._reconstruct"),
        (reconstruct, "numpy._core.multiarray._reconstruct"),
        (scalar, "numpy.core.multiarray.scalar"),
        (scalar, "numpy._core.multiarray.scalar"),
        np.ndarray,
        np.dtype,
        type(np.dtype(np.int64)),
        type(np.dtype(np.float64)),
    ]


def load_legacy_corrnet_bytes(payload: bytes) -> CorrNet:
    """Load one tensor-and-NumPy-only legacy checkpoint into the current model."""
    if not isinstance(payload, bytes) or not payload:
        raise ValueError("legacy CorrNet payload must be nonempty bytes")
    with torch.serialization.safe_globals(_legacy_numpy_safe_globals()):
        checkpoint = torch.load(
            BytesIO(payload),
            map_location="cpu",
            weights_only=True,
        )
    if not isinstance(checkpoint, Mapping):
        raise TypeError("legacy CorrNet checkpoint must be a mapping")
    if set(checkpoint) != {"state_dict", "init_args", "extra_info"}:
        raise ValueError("legacy CorrNet checkpoint fields are invalid")
    init_args = checkpoint["init_args"]
    if not isinstance(init_args, Mapping):
        raise ValueError("legacy CorrNet init_args must be a mapping")
    init_args = dict(_as_checkpoint_metadata(dict(init_args)))
    layer_norm = init_args.pop("layer_norm", False)
    if layer_norm is not False:
        raise ValueError("legacy CorrNet layer normalization is unsupported")
    normalized = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "state_dict": checkpoint["state_dict"],
        "init_args": init_args,
        "extra_info": _as_checkpoint_metadata(dict(checkpoint["extra_info"])),
    }
    return CorrNet.load_dict(normalized, strict=True)


__all__ = ["load_legacy_corrnet_bytes"]
