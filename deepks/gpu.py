"""GPU runtime and zero-copy array interoperability helpers."""

from __future__ import annotations

import ctypes
import importlib.util
import os
from pathlib import Path

import numpy as np
import torch


DEFAULT_CUDA_DEVICE = "cuda"
GPU_DIRECT_SCF_TOL = 1.0e-14
_CUDA_LIBRARY_HANDLES = []


class GPUConfigurationError(RuntimeError):
    """Raised when a GPU-only execution path cannot use CUDA."""


def _package_library_directory(module_name: str) -> Path | None:
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.submodule_search_locations is None:
        return None
    return Path(next(iter(spec.submodule_search_locations)))


def _preload_bundled_cuda_libraries() -> None:
    """Prefer the CUDA 12.8 runtime bundled with the locked PyTorch stack."""
    if _CUDA_LIBRARY_HANDLES:
        return
    runtime_directory = _package_library_directory("nvidia.cuda_runtime.lib")
    nvrtc_directory = _package_library_directory("nvidia.cuda_nvrtc.lib")
    candidates = []
    if runtime_directory is not None:
        candidates.append(runtime_directory / "libcudart.so.12")
    if nvrtc_directory is not None:
        candidates.extend(
            [
                nvrtc_directory / "libnvrtc-builtins.so.12.8",
                nvrtc_directory / "libnvrtc.so.12",
            ]
        )
    for library in candidates:
        if library.is_file():
            _CUDA_LIBRARY_HANDLES.append(
                ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
            )


def require_cuda_device(device=None) -> torch.device:
    """Select one visible CUDA device and reject every CPU fallback."""
    selected = torch.device(device or DEFAULT_CUDA_DEVICE)
    if selected.type != "cuda":
        raise GPUConfigurationError(
            f"DeepKS computations require a CUDA device; received {selected}"
        )
    if not os.environ.get("SLURM_JOB_ID"):
        raise GPUConfigurationError(
            "GPU workloads must run inside a Slurm allocation"
        )
    try:
        _preload_bundled_cuda_libraries()
    except OSError as error:
        raise GPUConfigurationError(
            f"the locked CUDA 12.8 runtime could not be loaded: {error}"
        ) from error
    if not torch.cuda.is_available():
        raise GPUConfigurationError(
            "CUDA is unavailable to PyTorch; run this GPU workload in a Slurm "
            "allocation with a visible NVIDIA GPU"
        )
    index = torch.cuda.current_device() if selected.index is None else selected.index
    if index < 0 or index >= torch.cuda.device_count():
        raise GPUConfigurationError(
            f"CUDA device index {index} is outside the {torch.cuda.device_count()} "
            "visible device(s)"
        )
    try:
        torch.cuda.set_device(index)
        import cupy

        cupy.cuda.Device(index).use()
    except Exception as error:
        raise GPUConfigurationError(
            f"CUDA device {index} could not be initialized by PyTorch and CuPy: {error}"
        ) from error
    return torch.device("cuda", index)


def torch_from_array(value, *, device=None, dtype=torch.float64) -> torch.Tensor:
    """Return an array as a tensor on the selected CUDA device."""
    selected = require_cuda_device(device)
    if isinstance(value, torch.Tensor):
        tensor = value
    elif hasattr(value, "__dlpack__") and hasattr(value, "__dlpack_device__"):
        tensor = torch.from_dlpack(value)
    else:
        tensor = torch.as_tensor(np.asanyarray(value))
    return tensor.to(device=selected, dtype=dtype)


def cupy_from_torch(value: torch.Tensor):
    """Share one CUDA tensor with CuPy through DLPack."""
    if not isinstance(value, torch.Tensor):
        raise TypeError("cupy_from_torch requires a torch.Tensor")
    if value.device.type != "cuda":
        raise GPUConfigurationError("CuPy interoperability requires a CUDA tensor")
    import cupy

    return cupy.from_dlpack(value.detach())


def as_numpy(value) -> np.ndarray:
    """Materialize a host NumPy array only at an output boundary."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if hasattr(value, "get") and hasattr(value, "__cuda_array_interface__"):
        return value.get()
    return np.asarray(value)


__all__ = [
    "DEFAULT_CUDA_DEVICE",
    "GPU_DIRECT_SCF_TOL",
    "GPUConfigurationError",
    "as_numpy",
    "cupy_from_torch",
    "require_cuda_device",
    "torch_from_array",
]
