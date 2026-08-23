"""Shared neural correction evaluation for DeePKS and DeePHF."""

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from deepks.descriptor.core import descriptor, projected_density, shell_eigenvalues
from deepks.model.model import validate_force_model_architecture


@dataclass(frozen=True)
class CorrectionPrediction:
    """One correction-energy prediction and its optional relaxed force."""

    energy: torch.Tensor
    force: torch.Tensor | None
    descriptor_gradient: torch.Tensor | None


def _require_float64_tensor(
    value,
    name: str,
    *,
    ndim: int | None = None,
    check_finite: bool = True,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.dtype != torch.float64:
        raise TypeError(f"{name} must use torch.float64")
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}; received shape {tuple(value.shape)}")
    if check_finite and not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")
    return value


def _validate_model_state(model, descriptor_values: torch.Tensor) -> None:
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    floating_state = [
        value
        for value in (*model.parameters(), *model.buffers())
        if value.is_floating_point()
    ]
    for value in floating_state:
        if value.dtype != torch.float64:
            raise TypeError("model parameters and buffers must use torch.float64")
        if not torch.isfinite(value).all():
            raise ValueError("model parameters and buffers must contain only finite values")
        if value.device != descriptor_values.device:
            raise ValueError(
                "descriptor and model parameters/buffers must be on the same device"
            )


def model_state_evidence(model) -> tuple:
    """Return metadata-only evidence for parameter and buffer mutation."""
    return tuple(
        (id(value), value._version, value.dtype, value.device)
        for value in (*model.parameters(), *model.buffers())
    )


def predict_correction(
    model,
    descriptor: torch.Tensor,
    dq_dR_relaxed: torch.Tensor | None = None,
    require_force: bool = False,
    create_graph: bool = False,
    *,
    _validated_inputs: bool = False,
    _validated_model: bool = False,
) -> CorrectionPrediction:
    """Predict a correction energy and, only from ``dq_dR_relaxed``, its force.

    Serialized force data use axes ``(frame, raw_atom, xyz,
    descriptor_atom, descriptor_feature)``.  The contraction in this helper is
    intentionally explicit so a fixed-density Jacobian cannot be substituted
    through broadcasting or an alternate axis convention.
    """
    descriptor = _require_float64_tensor(
        descriptor,
        "descriptor",
        ndim=3,
        check_finite=not _validated_inputs,
    )
    if any(size <= 0 for size in descriptor.shape):
        raise ValueError("descriptor axes must all be nonempty")
    if not isinstance(require_force, bool):
        raise TypeError("require_force must be bool")
    if not isinstance(create_graph, bool):
        raise TypeError("create_graph must be bool")
    if require_force and dq_dR_relaxed is None:
        raise ValueError("force prediction requires dq_dR_relaxed")

    calculate_force = dq_dR_relaxed is not None
    if calculate_force:
        if not _validated_model:
            validate_force_model_architecture(model, training=model.training)
        dq_dR_relaxed = _require_float64_tensor(
            dq_dR_relaxed,
            "dq_dR_relaxed",
            ndim=5,
            check_finite=not _validated_inputs,
        )
        expected_outer_axes = (
            descriptor.shape[0],
            3,
            descriptor.shape[1],
            descriptor.shape[2],
        )
        actual_outer_axes = (
            dq_dR_relaxed.shape[0],
            dq_dR_relaxed.shape[2],
            dq_dR_relaxed.shape[3],
            dq_dR_relaxed.shape[4],
        )
        if actual_outer_axes != expected_outer_axes or dq_dR_relaxed.shape[1] <= 0:
            raise ValueError(
                "dq_dR_relaxed must have shape "
                "(frame, raw_atom, 3, descriptor_atom, descriptor_feature) "
                f"matching descriptor; received {tuple(dq_dR_relaxed.shape)} for "
                f"descriptor {tuple(descriptor.shape)}"
            )
        if dq_dR_relaxed.device != descriptor.device:
            raise ValueError("descriptor and dq_dR_relaxed must be on the same device")

    if not _validated_model:
        _validate_model_state(model, descriptor)
    values = descriptor
    if calculate_force and not values.requires_grad:
        values = values.detach().requires_grad_(True)
    energy = model(values)
    energy = _require_float64_tensor(energy, "predicted energy", ndim=2)
    expected_energy_shape = (descriptor.shape[0], 1)
    if tuple(energy.shape) != expected_energy_shape:
        raise ValueError(
            "model must return one correction energy per frame with shape "
            f"{expected_energy_shape}; received {tuple(energy.shape)}"
        )

    if not calculate_force:
        return CorrectionPrediction(
            energy=energy,
            force=None,
            descriptor_gradient=None,
        )

    if energy.requires_grad:
        (descriptor_gradient,) = torch.autograd.grad(
            energy,
            values,
            grad_outputs=torch.ones_like(energy),
            retain_graph=create_graph,
            create_graph=create_graph,
            only_inputs=True,
            allow_unused=True,
        )
    else:
        descriptor_gradient = None
    if descriptor_gradient is None:
        descriptor_gradient = torch.zeros_like(values)
    descriptor_gradient = _require_float64_tensor(
        descriptor_gradient,
        "predicted descriptor gradient",
        ndim=3,
    )
    if descriptor_gradient.shape != descriptor.shape:
        raise ValueError(
            "predicted descriptor gradient shape does not match descriptor: "
            f"{tuple(descriptor_gradient.shape)} != {tuple(descriptor.shape)}"
        )
    force = -torch.einsum(
        "fbxik,fik->fbx",
        dq_dR_relaxed,
        descriptor_gradient,
    )
    force = _require_float64_tensor(force, "predicted force", ndim=3)
    return CorrectionPrediction(
        energy=energy,
        force=force,
        descriptor_gradient=descriptor_gradient,
    )


def model_reference(model):
    """Return a tensor carrying the model dtype and device."""
    try:
        return next(model.parameters())
    except StopIteration:
        try:
            return next(model.buffers())
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
    if not energy.requires_grad:
        return torch.zeros_like(values)
    (sensitivity,) = torch.autograd.grad(
        energy,
        values,
        torch.ones_like(energy),
        allow_unused=True,
    )
    return torch.zeros_like(values) if sensitivity is None else sensitivity
