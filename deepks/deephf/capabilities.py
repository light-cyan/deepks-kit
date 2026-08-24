"""Strict capability checks for perturbative DeePHF references and models."""

from contextvars import ContextVar
import functools
import hashlib

import numpy as np
import torch

from deepks.model.model import validate_force_model_architecture

from .contracts import update_digest


class DeePHFCapabilityError(ValueError):
    """Raised when a reference is outside the declared DeePHF domain."""


_VALIDATED_REFERENCES = ContextVar("deepks_validated_references", default=())


def begin_reference_validation_transaction(reference, fingerprint: str):
    """Trust one reference only within an enclosing checked calculation."""
    return _VALIDATED_REFERENCES.set(
        (*_VALIDATED_REFERENCES.get(), (reference, fingerprint))
    )


def end_reference_validation_transaction(token) -> None:
    _VALIDATED_REFERENCES.reset(token)


def reference_is_transaction_validated(reference) -> bool:
    return transaction_reference_fingerprint(reference) is not None


def transaction_reference_fingerprint(reference) -> str | None:
    for accepted, fingerprint in reversed(_VALIDATED_REFERENCES.get()):
        if reference is accepted:
            return fingerprint
    return None


def science_state_transaction(function):
    """Run one public calculation under a single scientific-state token."""

    @functools.wraps(function)
    def wrapped(owner, *args, **kwargs):
        method = getattr(owner, "_bound_base", getattr(owner, "base", owner))
        try:
            with method._calculation_context(
                validate_cached_state=True,
                publisher=owner,
            ):
                return function(owner, *args, **kwargs)
        except Exception:
            reset = getattr(owner, "_reset_results", None)
            if reset is not None:
                reset()
            raise

    return wrapped


def _update_model_tensor_fingerprint(digest, tensor: torch.Tensor) -> None:
    digest.update(str(tensor.layout).encode("utf-8"))
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(repr(tuple(tensor.shape)).encode("ascii"))
    if tensor.device.type == "meta":
        raise DeePHFCapabilityError(
            "the force correction model cannot use meta-device state"
        )
    try:
        value = tensor.detach().cpu()
        if value.layout != torch.strided:
            value = value.to_dense()
        update_digest(digest, value)
    except Exception as error:
        raise DeePHFCapabilityError(
            f"the force correction model tensor could not be fingerprinted: {error}"
        ) from error


def _metadata_signature(value):
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return tuple(_metadata_signature(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def validate_force_model(model):
    """Require the supported deterministic CorrNet in stable evaluation mode."""
    if model is None:
        return None
    if not isinstance(model, torch.nn.Module):
        raise DeePHFCapabilityError(
            "the force correction model must be a torch.nn.Module or None"
        )
    try:
        validate_force_model_architecture(model, training=False)
    except (TypeError, ValueError) as error:
        raise DeePHFCapabilityError(str(error)) from error
    training_modules = [
        name or "<root>"
        for name, module in model.named_modules(remove_duplicate=False)
        if module.training is not False
    ]
    if training_modules:
        raise DeePHFCapabilityError(
            "the force correction model must remain in evaluation mode; "
            f"training modules: {', '.join(training_modules)}"
        )
    return model


def model_state_fingerprint(model) -> str:
    """Bind one correction model's graph identity, mode, parameters, and buffers."""
    digest = hashlib.sha256()
    if model is None:
        digest.update(b"deepks.deephf.none-force-correction-model")
        return digest.hexdigest()
    if not isinstance(model, torch.nn.Module):
        raise DeePHFCapabilityError(
            "the DeePHF correction model must be a torch.nn.Module or None"
        )
    metadata = tuple(
        (
            name,
            f"{type(module).__module__}.{type(module).__qualname__}",
            bool(module.training),
            id(getattr(type(module), "forward", None)),
            id(getattr(type(module), "__call__", None)),
            id(vars(module).get("forward")),
            id(vars(module).get("_compiled_call_impl")),
        )
        for name, module in model.named_modules(remove_duplicate=False)
    )
    digest.update(repr(metadata).encode("utf-8"))
    for name, parameter in model.named_parameters(remove_duplicate=False):
        digest.update(b"parameter\0")
        digest.update(name.encode("utf-8"))
        digest.update(repr(bool(parameter.requires_grad)).encode("ascii"))
        _update_model_tensor_fingerprint(digest, parameter)
    for name, buffer in model.named_buffers(remove_duplicate=False):
        digest.update(b"buffer\0")
        digest.update(name.encode("utf-8"))
        _update_model_tensor_fingerprint(digest, buffer)
    return digest.hexdigest()


def model_state_evidence(model):
    """Return low-cost mutation evidence for one bound correction model."""
    if model is None:
        return None
    modules = tuple(model.named_modules(remove_duplicate=False))
    tensors = (
        *(('parameter', name, tensor) for name, tensor in model.named_parameters(remove_duplicate=False)),
        *(('buffer', name, tensor) for name, tensor in model.named_buffers(remove_duplicate=False)),
    )
    return (
        id(model),
        tuple(
            (
                name,
                id(module),
                id(type(module)),
                bool(module.training),
                id(getattr(type(module), "forward", None)),
                id(getattr(type(module), "__call__", None)),
                id(vars(module).get("forward")),
                id(vars(module).get("_compiled_call_impl")),
            )
            for name, module in modules
        ),
        tuple(
            (
                kind,
                name,
                id(tensor),
                tensor._version,
                str(tensor.layout),
                str(tensor.dtype),
                str(tensor.device),
                tuple(tensor.shape),
                bool(getattr(tensor, "requires_grad", False)),
            )
            for kind, name, tensor in tensors
        ),
    )


def force_model_fingerprint(model) -> str:
    """Bind the fixed supported force-model graph, parameters, and buffers."""
    validate_force_model(model)
    return model_state_fingerprint(model)


def validate_model(model, projector_basis, descriptor_features: int):
    """Validate the strict double-precision scalar correction-model contract."""
    if model is None:
        return None
    if not isinstance(model, torch.nn.Module):
        raise DeePHFCapabilityError(
            "the DeePHF correction model must be a torch.nn.Module or None"
        )
    tensors = list(model.parameters()) + list(model.buffers())
    for tensor in tensors:
        if tensor.is_complex():
            raise DeePHFCapabilityError("the correction model must be real")
        if tensor.is_floating_point() and tensor.dtype != torch.float64:
            raise DeePHFCapabilityError(
                "the correction model must use torch.float64"
            )
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise DeePHFCapabilityError(
                "the correction model parameters and buffers must be finite"
            )
    input_dimension = getattr(model, "input_dim", descriptor_features)
    if input_dimension != descriptor_features:
        raise DeePHFCapabilityError(
            "the correction model input dimension does not match the descriptor: "
            f"{input_dimension} != {descriptor_features}"
        )
    model_basis = getattr(model, "_pbas", None)
    if (
        model_basis is not None
        and _metadata_signature(model_basis) != _metadata_signature(projector_basis)
    ):
        raise DeePHFCapabilityError(
            "the correction model projector metadata does not match projector_basis"
        )
    return model


def validate_model_output(model, descriptor_values: torch.Tensor) -> torch.Tensor:
    """Evaluate and validate one real finite scalar correction energy."""
    if model is None:
        return torch.zeros((), dtype=torch.float64)
    try:
        reference_tensor = next(model.parameters())
    except StopIteration:
        try:
            reference_tensor = next(model.buffers())
        except StopIteration:
            reference_tensor = torch.empty((), dtype=torch.float64)
    try:
        output = model(descriptor_values.to(reference_tensor))
    except Exception as error:
        raise DeePHFCapabilityError(
            f"the correction model evaluation failed: {error}"
        ) from error
    if not isinstance(output, torch.Tensor):
        raise DeePHFCapabilityError("the correction model output must be a tensor")
    if output.is_complex():
        raise DeePHFCapabilityError("the correction model output must be real")
    if output.dtype != torch.float64:
        raise DeePHFCapabilityError(
            "the correction model output must use torch.float64"
        )
    if output.ndim > 1:
        raise DeePHFCapabilityError(
            "the correction model output must have rank zero or one"
        )
    if output.numel() != 1:
        raise DeePHFCapabilityError(
            "the correction model must produce exactly one scalar energy; "
            f"received shape {tuple(output.shape)}"
        )
    if not torch.isfinite(output).all():
        raise DeePHFCapabilityError("the correction model output must be finite")
    return output.reshape(())
