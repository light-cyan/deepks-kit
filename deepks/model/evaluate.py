"""Shared neural correction evaluation for DeePKS and DeePHF."""

from collections.abc import Sequence

import torch

from deepks.descriptor.core import descriptor, projected_density, shell_eigenvalues


def model_reference(model):
    """Return a tensor carrying the model dtype and device."""
    try:
        return next(model.parameters())
    except StopIteration:
        return torch.empty((), dtype=torch.float64)


def correction(
    model,
    ao_density: torch.Tensor,
    overlap_shells: Sequence[torch.Tensor],
    *,
    with_potential: bool = True,
):
    """Return the correction energy and optionally its AO potential."""
    ao_density.requires_grad_(with_potential)
    values = descriptor(ao_density, overlap_shells)
    energy = model(values.to(model_reference(model)))
    if not with_potential:
        return energy.to(values)
    (potential,) = torch.autograd.grad(
        energy,
        ao_density,
        torch.ones_like(energy),
    )
    return energy.to(values), potential


def correction_projected_density_gradients(
    model,
    ao_density: torch.Tensor,
    overlap_shells: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, ...]:
    """Return dE_corr/dD for every projected-density shell."""
    blocks = tuple(
        block.requires_grad_(True)
        for block in projected_density(ao_density, overlap_shells)
    )
    values = torch.cat([shell_eigenvalues(block) for block in blocks], dim=-1)
    energy = model(values.to(model_reference(model)))
    return torch.autograd.grad(energy, blocks)


def descriptor_sensitivity(model, values: torch.Tensor) -> torch.Tensor:
    """Return dE_corr/dq for descriptor compatibility validation."""
    values = values.detach().clone().requires_grad_(True)
    energy = model(values.to(model_reference(model)))
    (sensitivity,) = torch.autograd.grad(
        energy,
        values,
        torch.ones_like(energy),
    )
    return sensitivity
