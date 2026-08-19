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
) -> DescriptorDiagnostics:
    """Validate nondegenerate blocks and fixed-rank structural zero spaces."""
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
            for gap_index, gap in enumerate(np.diff(block)):
                scale = max(
                    abs(block[gap_index]),
                    abs(block[gap_index + 1]),
                    1.0,
                )
                threshold = gap_atol + gap_rtol * scale
                minimum_scaled_gap = min(minimum_scaled_gap, gap / threshold)
                inside_structural_zero_block = gap_index < structural_nullity - 1
                if gap <= threshold and not inside_structural_zero_block:
                    raise DescriptorDifferentiabilityError(
                        f"descriptor atom {atom_index}, shell {shell_index}: "
                        f"eigenvalue gap {gap:.3e} at block positions "
                        f"{gap_index}:{gap_index + 2} does not exceed {threshold:.3e}"
                    )
        offset += shell_size
    return DescriptorDiagnostics(
        minimum_scaled_gap=float(minimum_scaled_gap),
        structural_zero_blocks=tuple(structural_zero_blocks),
    )
