from dataclasses import dataclass
import tracemalloc

import numpy as np
import torch

from deepks.deephf.contracts import (
    array_fingerprint,
    dataclass_fingerprint,
    immutable_array,
)
from deepks.descriptor.workspace import DescriptorDerivativeWorkspace


@dataclass(frozen=True)
class ArrayResult:
    integrity_fingerprint: str
    values: np.ndarray


class TensorDescriptor:
    shell_sizes = (3,)
    descriptor_atom_indices = tuple(range(5))

    def __init__(self):
        values = torch.arange(16 * 5 * 3, dtype=torch.float64)
        self.overlap_shells = (values.reshape(16, 5, 3) / values.numel(),)

    @staticmethod
    def as_ao_density_tensor(value):
        return torch.as_tensor(value, dtype=torch.float64)


def peak_allocation(function):
    tracemalloc.start()
    try:
        function()
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak


def test_contiguous_array_hashing_does_not_copy_the_full_buffer():
    values = np.arange(1024 * 1024, dtype=np.float64).reshape(1024, 1024)
    peak = peak_allocation(lambda: array_fingerprint(values))
    assert peak < values.nbytes // 4


def test_immutable_result_construction_and_hashing_use_one_full_allocation():
    values = np.arange(1024 * 1024, dtype=np.float64).reshape(1024, 1024)

    def construct_and_hash():
        frozen = immutable_array(values)
        result = ArrayResult("", frozen)
        dataclass_fingerprint(
            result,
            excluded=frozenset({"integrity_fingerprint"}),
        )

    peak = peak_allocation(construct_and_hash)
    assert values.nbytes <= peak < int(values.nbytes * 1.4)


def test_detailed_derivative_workspace_has_bounded_tensor_materialization():
    workspace = DescriptorDerivativeWorkspace(TensorDescriptor(), np.eye(16))
    with torch.profiler.profile(profile_memory=True) as profile:
        derivative = workspace.dq_dP()

    allocated = sum(
        max(0, event.self_cpu_memory_usage)
        for event in profile.key_averages()
    )
    assert derivative.shape == (5, 3, 16, 16)
    assert derivative.nbytes <= allocated < 3 * derivative.nbytes
