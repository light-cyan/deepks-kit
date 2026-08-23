"""Explicit fixed-AO-density nuclear derivatives of shared descriptors."""

from collections.abc import Sequence

import torch

from .core import batch_jacobian, projected_density, shell_eigenvalues


def dD_dR_explicit(
    mol,
    ao_density: torch.Tensor,
    overlap_shells: Sequence[torch.Tensor],
    derivative_overlap_shells: Sequence[torch.Tensor],
    descriptor_atom_indices: Sequence[int],
    raw_atom_indices: Sequence[int] | None = None,
) -> list[torch.Tensor]:
    """Return fixed-density dD/dR with raw-atom and descriptor-atom axes."""
    if raw_atom_indices is None:
        raw_atom_indices = tuple(range(mol.natm))
    else:
        raw_atom_indices = tuple(raw_atom_indices)
    raw_to_result = {
        raw_atom_index: result_index
        for result_index, raw_atom_index in enumerate(raw_atom_indices)
    }
    atom_slices = mol.aoslice_by_atom()
    shell_sizes = [overlap.shape[-1] for overlap in overlap_shells]
    jacobian_blocks = [
        torch.zeros(
            (len(raw_atom_indices), 3, len(descriptor_atom_indices), size, size),
            dtype=ao_density.dtype,
            device=ao_density.device,
        )
        for size in shell_sizes
    ]
    for jacobian, derivative_overlap, overlap in zip(
        jacobian_blocks,
        derivative_overlap_shells,
        overlap_shells,
    ):
        projector_motion = torch.einsum(
            "xrap,rs,saq->xapq", derivative_overlap, ao_density, overlap
        )
        for result_index, raw_atom_index in enumerate(raw_atom_indices):
            ao_start, ao_stop = atom_slices[raw_atom_index, 2:]
            jacobian[result_index] -= torch.einsum(
                "xrap,rs,saq->xapq",
                derivative_overlap[:, ao_start:ao_stop],
                ao_density[ao_start:ao_stop],
                overlap,
            )
        for descriptor_atom_index, raw_atom_index in enumerate(
            descriptor_atom_indices
        ):
            result_index = raw_to_result.get(raw_atom_index)
            if result_index is not None:
                jacobian[result_index, :, descriptor_atom_index] += (
                    projector_motion[:, descriptor_atom_index]
                )
        jacobian += jacobian.clone().transpose(-1, -2)
    return jacobian_blocks


def dq_dR_explicit(
    mol,
    ao_density: torch.Tensor,
    overlap_shells: Sequence[torch.Tensor],
    derivative_overlap_shells: Sequence[torch.Tensor],
    descriptor_atom_indices: Sequence[int],
    raw_atom_indices: Sequence[int] | None = None,
) -> torch.Tensor:
    """Return fixed-density dq/dR with axes (raw_atom, xyz, descriptor_atom, feature)."""
    density_blocks = [
        block.requires_grad_(True)
        for block in projected_density(ao_density, overlap_shells)
    ]
    eigenvalue_jacobians = [
        batch_jacobian(shell_eigenvalues, block, block.shape[-1])
        for block in density_blocks
    ]
    coordinate_jacobians = dD_dR_explicit(
        mol,
        ao_density,
        overlap_shells,
        derivative_overlap_shells,
        descriptor_atom_indices,
        raw_atom_indices,
    )
    shell_results = [
        torch.einsum("bxapq,avpq->bxav", coordinate, eigenvalue)
        for coordinate, eigenvalue in zip(
            coordinate_jacobians,
            eigenvalue_jacobians,
        )
    ]
    return torch.cat(shell_results, dim=-1)


def dq_dR_explicit_component(
    mol,
    ao_density: torch.Tensor,
    component_density: torch.Tensor,
    overlap_shells: Sequence[torch.Tensor],
    derivative_overlap_shells: Sequence[torch.Tensor],
    descriptor_atom_indices: Sequence[int],
) -> torch.Tensor:
    """Return one additive fixed-density component of the descriptor derivative."""
    if component_density.shape != ao_density.shape:
        raise ValueError("component_density must match ao_density")
    density_blocks = [
        block.requires_grad_(True)
        for block in projected_density(ao_density, overlap_shells)
    ]
    eigenvalue_jacobians = [
        batch_jacobian(shell_eigenvalues, block, block.shape[-1])
        for block in density_blocks
    ]
    coordinate_jacobians = dD_dR_explicit(
        mol,
        component_density,
        overlap_shells,
        derivative_overlap_shells,
        descriptor_atom_indices,
    )
    shell_results = [
        torch.einsum("bxapq,avpq->bxav", coordinate, eigenvalue)
        for coordinate, eigenvalue in zip(
            coordinate_jacobians,
            eigenvalue_jacobians,
        )
    ]
    return torch.cat(shell_results, dim=-1)
