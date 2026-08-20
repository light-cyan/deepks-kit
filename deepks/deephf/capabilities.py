"""Strict capability checks for perturbative DeePHF references and models."""

from collections.abc import Mapping
import functools
import hashlib
import marshal
import random
import types

import numpy as np
import torch


class DeePHFCapabilityError(ValueError):
    """Raised when a reference is outside the declared DeePHF domain."""


_MODULE_EXECUTION_HOOK_FIELDS = (
    ("forward-pre", "_forward_pre_hooks"),
    ("forward", "_forward_hooks"),
    ("backward-pre", "_backward_pre_hooks"),
    ("backward", "_backward_hooks"),
)
_GLOBAL_MODULE_EXECUTION_HOOK_FIELDS = (
    ("global-forward-pre", "_global_forward_pre_hooks"),
    ("global-forward", "_global_forward_hooks"),
    ("global-backward-pre", "_global_backward_pre_hooks"),
    ("global-backward", "_global_backward_hooks"),
)
_MODULE_CONTAINER_FIELDS = frozenset({"_parameters", "_buffers", "_modules"})


def _qualified_type(value) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


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
        flat_value = torch.empty(value.numel(), dtype=value.dtype, device="cpu")
        flat_value.copy_(value.reshape(-1))
        digest.update(flat_value.view(torch.uint8).numpy().tobytes())
    except Exception as error:
        raise DeePHFCapabilityError(
            f"the force correction model tensor could not be fingerprinted: {error}"
        ) from error


def _update_callable_fingerprint(digest, value, active_objects) -> bool:
    if isinstance(value, functools.partial):
        digest.update(b"functools.partial\0")
        _update_model_metadata_fingerprint(digest, value.func, active_objects)
        _update_model_metadata_fingerprint(digest, value.args, active_objects)
        _update_model_metadata_fingerprint(digest, value.keywords, active_objects)
        return True
    if isinstance(value, types.MethodType):
        digest.update(b"python.bound-method\0")
        _update_model_metadata_fingerprint(digest, value.__func__, active_objects)
        bound_self = value.__self__
        if isinstance(bound_self, torch.nn.Module):
            digest.update(_qualified_type(bound_self).encode("utf-8"))
        else:
            _update_model_metadata_fingerprint(
                digest,
                bound_self,
                active_objects,
            )
        return True
    if isinstance(value, types.FunctionType):
        identity = id(value)
        if identity in active_objects:
            digest.update(b"<recursive-function>")
            return True
        active_objects.add(identity)
        try:
            digest.update(b"python.function\0")
            digest.update(str(value.__module__).encode("utf-8"))
            digest.update(str(value.__qualname__).encode("utf-8"))
            digest.update(marshal.dumps(value.__code__))
            for global_name in sorted(set(value.__code__.co_names)):
                if global_name not in value.__globals__:
                    continue
                global_value = value.__globals__[global_name]
                digest.update(b"global\0")
                digest.update(global_name.encode("utf-8"))
                if isinstance(global_value, types.ModuleType):
                    digest.update(global_value.__name__.encode("utf-8"))
                    module_version = getattr(global_value, "__version__", None)
                    if module_version is not None:
                        digest.update(str(module_version).encode("utf-8"))
                else:
                    _update_model_metadata_fingerprint(
                        digest,
                        global_value,
                        active_objects,
                    )
            _update_model_metadata_fingerprint(
                digest,
                value.__defaults__,
                active_objects,
            )
            _update_model_metadata_fingerprint(
                digest,
                value.__kwdefaults__,
                active_objects,
            )
            _update_model_metadata_fingerprint(
                digest,
                value.__annotations__,
                active_objects,
            )
            _update_model_metadata_fingerprint(
                digest,
                vars(value),
                active_objects,
            )
            closure = value.__closure__ or ()
            for cell in closure:
                try:
                    cell_value = cell.cell_contents
                except ValueError:
                    digest.update(b"<empty-closure-cell>")
                else:
                    _update_model_metadata_fingerprint(
                        digest,
                        cell_value,
                        active_objects,
                    )
        finally:
            active_objects.remove(identity)
        return True
    if isinstance(value, (types.BuiltinFunctionType, types.BuiltinMethodType)):
        digest.update(b"builtin.callable\0")
        digest.update(str(getattr(value, "__module__", "")).encode("utf-8"))
        digest.update(
            str(getattr(value, "__qualname__", "")).encode("utf-8")
        )
        bound_self = getattr(value, "__self__", None)
        if bound_self is not None:
            if isinstance(bound_self, torch.nn.Module):
                digest.update(_qualified_type(bound_self).encode("utf-8"))
            elif isinstance(bound_self, types.ModuleType):
                digest.update(bound_self.__name__.encode("utf-8"))
                module_version = getattr(bound_self, "__version__", None)
                if module_version is not None:
                    digest.update(str(module_version).encode("utf-8"))
            else:
                _update_model_metadata_fingerprint(
                    digest,
                    bound_self,
                    active_objects,
                )
        return True
    if isinstance(value, type):
        digest.update(b"callable.type\0")
        digest.update(_qualified_type(value).encode("utf-8"))
        return True
    return False


def _update_model_metadata_fingerprint(digest, value, active_objects) -> None:
    digest.update(_qualified_type(value).encode("utf-8"))
    if value is None:
        return
    if isinstance(value, (bool, int, str, bytes)):
        digest.update(repr(value).encode("utf-8"))
        return
    if isinstance(value, float):
        digest.update(value.hex().encode("ascii"))
        return
    if isinstance(value, np.generic):
        _update_model_metadata_fingerprint(digest, value.item(), active_objects)
        return
    if isinstance(value, torch.Tensor):
        _update_model_tensor_fingerprint(digest, value)
        return
    if isinstance(value, np.ndarray):
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(repr(value.shape).encode("ascii"))
        if value.dtype.hasobject:
            for item in value.reshape(-1):
                _update_model_metadata_fingerprint(
                    digest,
                    item,
                    active_objects,
                )
        else:
            digest.update(np.ascontiguousarray(value).tobytes())
        return
    if isinstance(value, random.Random):
        digest.update(repr(value.getstate()).encode("utf-8"))
        return
    if isinstance(value, np.random.Generator):
        _update_model_metadata_fingerprint(
            digest,
            value.bit_generator.state,
            active_objects,
        )
        return
    if isinstance(value, np.random.RandomState):
        _update_model_metadata_fingerprint(
            digest,
            value.get_state(),
            active_objects,
        )
        return
    if isinstance(value, torch.Generator):
        digest.update(str(value.device).encode("utf-8"))
        _update_model_tensor_fingerprint(digest, value.get_state())
        return
    if callable(value) and _update_callable_fingerprint(
        digest,
        value,
        active_objects,
    ):
        return
    identity = id(value)
    if identity in active_objects:
        digest.update(b"<recursive>")
        return
    active_objects.add(identity)
    try:
        if isinstance(value, Mapping):
            ordered_items = sorted(
                value.items(),
                key=lambda item: (_qualified_type(item[0]), repr(item[0])),
            )
            for key, item in ordered_items:
                _update_model_metadata_fingerprint(digest, key, active_objects)
                _update_model_metadata_fingerprint(digest, item, active_objects)
            return
        if isinstance(value, (tuple, list)):
            for item in value:
                _update_model_metadata_fingerprint(digest, item, active_objects)
            return
        if isinstance(value, (set, frozenset)):
            for item in sorted(
                value,
                key=lambda item: (_qualified_type(item), repr(item)),
            ):
                _update_model_metadata_fingerprint(digest, item, active_objects)
            return
        if isinstance(value, torch.nn.Module):
            digest.update(_qualified_type(value).encode("utf-8"))
            return
        handled_state = False
        try:
            attributes = vars(value)
        except TypeError:
            attributes = None
        if isinstance(attributes, dict):
            for name, item in sorted(attributes.items()):
                digest.update(name.encode("utf-8"))
                _update_model_metadata_fingerprint(digest, item, active_objects)
            handled_state = True
        slot_names = set()
        for value_type in type(value).__mro__:
            for name, descriptor in vars(value_type).items():
                if isinstance(descriptor, types.MemberDescriptorType):
                    if name not in {"__dict__", "__weakref__"}:
                        slot_names.add(name)
        for name in sorted(slot_names):
            digest.update(b"slot\0")
            digest.update(name.encode("utf-8"))
            try:
                item = getattr(value, name)
            except AttributeError:
                digest.update(b"<unset-slot>")
            else:
                _update_model_metadata_fingerprint(digest, item, active_objects)
            handled_state = True
        if handled_state:
            return
        if callable(value):
            raise DeePHFCapabilityError(
                "the force correction model contains a callable that cannot be "
                f"fingerprinted stably: {_qualified_type(value)}"
            )
        raise DeePHFCapabilityError(
            "the force correction model contains opaque state that cannot be "
            f"fingerprinted stably: {_qualified_type(value)}"
        )
    finally:
        active_objects.remove(identity)


def _metadata_signature(value):
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return tuple(_metadata_signature(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def validate_force_model(model):
    """Require stable evaluation mode and hook-free execution for force inference."""
    if model is None:
        return None
    if not isinstance(model, torch.nn.Module):
        raise DeePHFCapabilityError(
            "the force correction model must be a torch.nn.Module or None"
        )
    try:
        modules = tuple(model.named_modules(remove_duplicate=False))
    except Exception as error:
        raise DeePHFCapabilityError(
            f"the force correction model modules could not be inspected: {error}"
        ) from error
    training_modules = [
        name or "<root>"
        for name, module in modules
        if module.training is not False
    ]
    if training_modules:
        raise DeePHFCapabilityError(
            "the force correction model must remain in evaluation mode; "
            f"training modules: {', '.join(training_modules)}"
        )
    active_hooks = []
    for name, module in modules:
        module_name = name or "<root>"
        for hook_name, field_name in _MODULE_EXECUTION_HOOK_FIELDS:
            try:
                registry = getattr(module, field_name)
            except Exception as error:
                raise DeePHFCapabilityError(
                    "the force correction model hook registry could not be "
                    f"inspected: {module_name}:{hook_name}: {error}"
                ) from error
            if not isinstance(registry, Mapping):
                raise DeePHFCapabilityError(
                    "the force correction model hook registry is invalid: "
                    f"{module_name}:{hook_name}"
                )
            if registry:
                active_hooks.append(f"{module_name}:{hook_name}")
    global_module_hooks = torch.nn.modules.module
    for hook_name, field_name in _GLOBAL_MODULE_EXECUTION_HOOK_FIELDS:
        try:
            registry = getattr(global_module_hooks, field_name)
        except Exception as error:
            raise DeePHFCapabilityError(
                "the global force-model hook registry could not be inspected: "
                f"{hook_name}: {error}"
            ) from error
        if not isinstance(registry, Mapping):
            raise DeePHFCapabilityError(
                f"the global force-model hook registry is invalid: {hook_name}"
            )
        if registry:
            active_hooks.append(hook_name)
    if active_hooks:
        raise DeePHFCapabilityError(
            "the force correction model cannot contain module execution hooks; "
            f"active hooks: {', '.join(active_hooks)}"
        )
    return model


def force_model_fingerprint(model) -> str:
    """Bind force-model structure, semantic attributes, parameters, and buffers."""
    validate_force_model(model)
    digest = hashlib.sha256()
    if model is None:
        digest.update(b"deepks.deephf.none-force-correction-model")
        return digest.hexdigest()
    try:
        modules = tuple(model.named_modules(remove_duplicate=False))
        parameters = tuple(model.named_parameters(remove_duplicate=False))
        buffers = tuple(model.named_buffers(remove_duplicate=False))
        state = model.state_dict()
    except Exception as error:
        raise DeePHFCapabilityError(
            f"the force correction model state could not be enumerated: {error}"
        ) from error
    for name, module in modules:
        digest.update(b"module\0")
        digest.update(name.encode("utf-8"))
        digest.update(_qualified_type(module).encode("utf-8"))
        try:
            resolved_forward = getattr(module, "forward")
        except Exception as error:
            raise DeePHFCapabilityError(
                f"the force correction model forward could not be resolved: {error}"
            ) from error
        digest.update(b"resolved-forward\0")
        _update_model_metadata_fingerprint(digest, resolved_forward, set())
        if not name:
            resolved_element_constant = getattr(
                module,
                "get_elem_const",
                None,
            )
            digest.update(b"resolved-element-constant\0")
            _update_model_metadata_fingerprint(
                digest,
                resolved_element_constant,
                set(),
            )
        for attribute_name, value in sorted(vars(module).items()):
            if attribute_name in _MODULE_CONTAINER_FIELDS:
                continue
            digest.update(attribute_name.encode("utf-8"))
            _update_model_metadata_fingerprint(digest, value, set())
    for name, parameter in parameters:
        digest.update(b"parameter\0")
        digest.update(name.encode("utf-8"))
        digest.update(repr(bool(parameter.requires_grad)).encode("ascii"))
        _update_model_tensor_fingerprint(digest, parameter)
    for name, buffer in buffers:
        digest.update(b"buffer\0")
        digest.update(name.encode("utf-8"))
        _update_model_tensor_fingerprint(digest, buffer)
    if not isinstance(state, Mapping):
        raise DeePHFCapabilityError(
            "the force correction model state_dict must be a mapping"
        )
    for name, value in sorted(state.items()):
        digest.update(b"state\0")
        digest.update(str(name).encode("utf-8"))
        _update_model_metadata_fingerprint(digest, value, set())
    return digest.hexdigest()


def force_rng_fingerprints() -> dict[str, str]:
    """Snapshot global Python, NumPy, and initialized Torch RNG states."""
    result = {}
    python_digest = hashlib.sha256(repr(random.getstate()).encode("utf-8"))
    result["Python"] = python_digest.hexdigest()
    numpy_digest = hashlib.sha256()
    numpy_state = np.random.get_state()
    for value in numpy_state:
        if isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            numpy_digest.update(array.dtype.str.encode("ascii"))
            numpy_digest.update(repr(array.shape).encode("ascii"))
            numpy_digest.update(array.tobytes())
        else:
            numpy_digest.update(repr(value).encode("utf-8"))
    result["NumPy"] = numpy_digest.hexdigest()
    torch_cpu_state = torch.random.get_rng_state()
    torch_cpu_digest = hashlib.sha256(
        torch_cpu_state.cpu().numpy().tobytes()
    )
    result["Torch CPU"] = torch_cpu_digest.hexdigest()
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        cuda_digest = hashlib.sha256()
        for state in torch.cuda.get_rng_state_all():
            cuda_digest.update(state.cpu().numpy().tobytes())
        result["Torch CUDA"] = cuda_digest.hexdigest()
    return result


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
    if output.numel() != 1:
        raise DeePHFCapabilityError(
            "the correction model must produce exactly one scalar energy; "
            f"received shape {tuple(output.shape)}"
        )
    if not torch.isfinite(output).all():
        raise DeePHFCapabilityError("the correction model output must be finite")
    return output
