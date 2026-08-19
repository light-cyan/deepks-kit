"""Method-independent projected-density descriptor operations."""

from collections.abc import Callable, Sequence

import torch


def projected_density(
    ao_density: torch.Tensor,
    overlap_shells: Sequence[torch.Tensor],
) -> list[torch.Tensor]:
    """Return projected-density blocks D = O^T P O for every shell."""
    return [
        torch.einsum("rap,...rs,saq->...apq", overlap, ao_density, overlap)
        for overlap in overlap_shells
    ]


def shell_eigenvalues(projected_density_block: torch.Tensor) -> torch.Tensor:
    """Return ordered eigenvalues for one projected-density shell."""
    return torch.linalg.eigvalsh(projected_density_block)


def descriptor(
    ao_density: torch.Tensor,
    overlap_shells: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Return the concatenated ordered-eigenvalue descriptor q."""
    blocks = projected_density(ao_density, overlap_shells)
    return torch.cat([shell_eigenvalues(block) for block in blocks], dim=-1)


def batch_jacobian(
    function: Callable[[torch.Tensor], torch.Tensor],
    argument: torch.Tensor,
    output_size: int,
) -> torch.Tensor:
    """Evaluate independent Jacobians along the first argument axis."""
    input_rank = argument.ndim - 1
    expanded = argument.unsqueeze(1)
    batch_size = expanded.shape[0]
    expanded = expanded.repeat(1, output_size, *([1] * input_rank))
    expanded.requires_grad_(True)
    output = function(expanded)
    seeds = torch.eye(output_size, dtype=output.dtype, device=output.device)
    seeds = seeds.reshape(1, output_size, output_size).repeat(batch_size, 1, 1)
    return torch.autograd.grad(output, expanded, seeds)[0]


def dq_dP(
    ao_density: torch.Tensor,
    overlap_shells: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Return dq/dP with axes (descriptor_atom, feature, ao, ao)."""
    density_blocks = [
        block.requires_grad_(True)
        for block in projected_density(ao_density, overlap_shells)
    ]
    block_jacobians = [
        batch_jacobian(shell_eigenvalues, block, block.shape[-1])
        for block in density_blocks
    ]
    ao_jacobians = [
        torch.einsum("rap,avpq,saq->avrs", overlap, jacobian, overlap)
        for overlap, jacobian in zip(overlap_shells, block_jacobians)
    ]
    return torch.cat(ao_jacobians, dim=1)
