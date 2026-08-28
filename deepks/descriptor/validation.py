"""Differentiability diagnostics for ordered-eigenvalue descriptors."""

from dataclasses import dataclass

import numpy as np


class DescriptorDifferentiabilityError(ValueError):
    """Raised when an ordered-eigenvalue descriptor has no accepted derivative."""


@dataclass(frozen=True)
class DescriptorDiagnostics:
    minimum_scaled_gap: float
    structural_zero_blocks: tuple[tuple[int, int, int, int], ...]


def validate_differentiability(
    values,
    shell_sizes,
    n_occupied: int,
    sensitivity=None,
    *,
    gap_atol: float = 1.0e-9,
    gap_rtol: float = 1.0e-7,
    zero_atol: float = 1.0e-9,
    sensitivity_atol: float = 1.0e-8,
    symmetric_function: bool = False,
) -> DescriptorDiagnostics:
    """Validate simple spectra and sensitivity-symmetric degenerate spaces."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != sum(shell_sizes):
        raise ValueError("values must have shape (descriptor_atom, sum(shell_sizes))")
    if not np.isfinite(values).all():
        raise DescriptorDifferentiabilityError(
            "descriptor values must be finite"
        )
    if sensitivity is not None:
        sensitivity = np.asarray(sensitivity, dtype=float).reshape(values.shape)
        if not np.isfinite(sensitivity).all():
            raise DescriptorDifferentiabilityError(
                "descriptor sensitivity must be finite"
            )
    structural_zero_blocks = []
    minimum_scaled_gap = np.inf
    offset = 0
    for shell_index, shell_size in enumerate(shell_sizes):
        structural_nullity = max(shell_size - n_occupied, 0)
        for atom_index, atom_values in enumerate(values):
            block = atom_values[offset : offset + shell_size]
            block_sensitivity = None
            if sensitivity is not None:
                block_sensitivity = sensitivity[
                    atom_index,
                    offset : offset + shell_size,
                ]
            if structural_nullity:
                zero_block = block[:structural_nullity]
                if np.max(np.abs(zero_block), initial=0.0) > zero_atol:
                    raise DescriptorDifferentiabilityError(
                        f"descriptor atom {atom_index}, shell {shell_index}: "
                        f"fixed-rank zero block residual {np.max(np.abs(zero_block)):.3e} "
                        f"exceeds {zero_atol:.3e}"
                    )
                if structural_nullity > 1 and block_sensitivity is not None:
                    spread = np.ptp(block_sensitivity[:structural_nullity])
                    if spread > sensitivity_atol:
                        raise DescriptorDifferentiabilityError(
                            f"descriptor atom {atom_index}, shell {shell_index}: "
                            f"structural zero block sensitivity spread {spread:.3e} "
                            f"exceeds {sensitivity_atol:.3e}"
                        )
                structural_zero_blocks.append(
                    (atom_index, shell_index, offset, offset + structural_nullity)
                )
            gaps = np.diff(block)
            small_gaps = []
            for gap_index, gap in enumerate(gaps):
                scale = max(
                    abs(block[gap_index]),
                    abs(block[gap_index + 1]),
                    1.0,
                )
                threshold = gap_atol + gap_rtol * scale
                minimum_scaled_gap = min(minimum_scaled_gap, gap / threshold)
                small_gaps.append(gap <= threshold)
            gap_index = 0
            while gap_index < len(gaps):
                if not small_gaps[gap_index]:
                    gap_index += 1
                    continue
                cluster_start = gap_index
                while gap_index + 1 < len(gaps) and small_gaps[gap_index + 1]:
                    gap_index += 1
                cluster_stop = gap_index + 2
                inside_structural_zero_block = cluster_stop <= structural_nullity
                if not inside_structural_zero_block and not symmetric_function:
                    if block_sensitivity is None:
                        raise DescriptorDifferentiabilityError(
                            f"descriptor atom {atom_index}, shell {shell_index}: "
                            f"eigenvalue gap {gaps[cluster_start]:.3e} at block positions "
                            f"{cluster_start}:{cluster_stop} has no sensitivity symmetry evidence"
                        )
                    spread = np.ptp(block_sensitivity[cluster_start:cluster_stop])
                    if spread > sensitivity_atol:
                        raise DescriptorDifferentiabilityError(
                            f"descriptor atom {atom_index}, shell {shell_index}: "
                            f"degenerate block sensitivity spread {spread:.3e} "
                            f"exceeds {sensitivity_atol:.3e} at block positions "
                            f"{cluster_start}:{cluster_stop}"
                        )
                gap_index += 1
        offset += shell_size
    return DescriptorDiagnostics(
        minimum_scaled_gap=float(minimum_scaled_gap),
        structural_zero_blocks=tuple(structural_zero_blocks),
    )
