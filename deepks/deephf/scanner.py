"""Strict fresh-reference geometry scanner for RHF DeePHF gradients."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import math
from types import MappingProxyType
from typing import Any

import numpy as np
import torch

from .gradient import _validate_atom_indices
from .method import DeePHF
from .pyscf_rhf import RHFScannerReferenceFactory


class RHFDeePHFScannerError(RuntimeError):
    """Raised when strict scanner state cannot be constructed or published."""


@dataclass(frozen=True)
class _ScannerResult:
    """One complete scanner result ready for atomic publication."""

    mol: object
    reference: object
    method: DeePHF
    gradient_driver: object
    e_tot: float
    de: np.ndarray
    model_state_fingerprint: str


@dataclass(frozen=True)
class _AtomDomain:
    """Minimal immutable molecule view used for pre-SCF atom validation."""

    natm: int


_MODULE_CONTAINER_FIELDS = frozenset({"_parameters", "_buffers", "_modules"})


def _qualified_type(value: object) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _update_tensor_fingerprint(digest, tensor: torch.Tensor) -> None:
    digest.update(str(tensor.layout).encode("utf-8"))
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(repr(tuple(tensor.shape)).encode("ascii"))
    if tensor.device.type == "meta":
        raise RHFDeePHFScannerError(
            "the correction model cannot be fingerprinted on the meta device"
        )
    try:
        value = tensor.detach().cpu()
        if value.layout != torch.strided:
            value = value.to_dense()
        flat_value = torch.empty(value.numel(), dtype=value.dtype, device="cpu")
        flat_value.copy_(value.reshape(-1))
        raw_bytes = flat_value.view(torch.uint8).numpy().tobytes()
    except Exception as error:
        raise RHFDeePHFScannerError(
            f"the correction model tensor could not be fingerprinted: {error}"
        ) from error
    digest.update(raw_bytes)


def _update_metadata_fingerprint(
    digest,
    value: Any,
    active_objects: set[int],
) -> None:
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
        _update_metadata_fingerprint(digest, value.item(), active_objects)
        return
    if isinstance(value, torch.Tensor):
        _update_tensor_fingerprint(digest, value)
        return
    if isinstance(value, np.ndarray):
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(repr(value.shape).encode("ascii"))
        if value.dtype.hasobject:
            digest.update(repr(value.tolist()).encode("utf-8"))
        else:
            digest.update(np.ascontiguousarray(value).tobytes())
        return
    if callable(value):
        digest.update(
            str(getattr(value, "__module__", "")).encode("utf-8")
        )
        digest.update(
            str(getattr(value, "__qualname__", repr(value))).encode("utf-8")
        )
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
                _update_metadata_fingerprint(digest, key, active_objects)
                _update_metadata_fingerprint(digest, item, active_objects)
            return
        if isinstance(value, (tuple, list)):
            for item in value:
                _update_metadata_fingerprint(digest, item, active_objects)
            return
        if isinstance(value, (set, frozenset)):
            for item in sorted(value, key=lambda item: (_qualified_type(item), repr(item))):
                _update_metadata_fingerprint(digest, item, active_objects)
            return
        if isinstance(value, torch.nn.Module):
            digest.update(_qualified_type(value).encode("utf-8"))
            return
        attributes = getattr(value, "__dict__", None)
        if isinstance(attributes, dict):
            for name, item in sorted(attributes.items()):
                digest.update(name.encode("utf-8"))
                _update_metadata_fingerprint(digest, item, active_objects)
            return
        digest.update(repr(value).encode("utf-8"))
    finally:
        active_objects.remove(identity)


def _model_state_fingerprint(model) -> str:
    """Bind model structure, semantic metadata, parameters, and buffers."""
    digest = hashlib.sha256()
    if model is None:
        digest.update(b"deepks.deephf.none-correction-model")
        return digest.hexdigest()
    if not isinstance(model, torch.nn.Module):
        raise RHFDeePHFScannerError(
            "the scanner correction model must be a torch.nn.Module or None"
        )
    try:
        modules = tuple(model.named_modules(remove_duplicate=False))
        parameters = tuple(model.named_parameters(remove_duplicate=False))
        buffers = tuple(model.named_buffers(remove_duplicate=False))
        state = model.state_dict()
    except Exception as error:
        raise RHFDeePHFScannerError(
            f"the correction model state could not be enumerated: {error}"
        ) from error
    for name, module in modules:
        digest.update(b"module\0")
        digest.update(name.encode("utf-8"))
        digest.update(_qualified_type(module).encode("utf-8"))
        for attribute_name, value in sorted(module.__dict__.items()):
            if attribute_name in _MODULE_CONTAINER_FIELDS or attribute_name == "training":
                continue
            digest.update(attribute_name.encode("utf-8"))
            _update_metadata_fingerprint(digest, value, set())
    for name, parameter in parameters:
        digest.update(b"parameter\0")
        digest.update(name.encode("utf-8"))
        digest.update(repr(bool(parameter.requires_grad)).encode("ascii"))
        _update_tensor_fingerprint(digest, parameter)
    for name, buffer in buffers:
        digest.update(b"buffer\0")
        digest.update(name.encode("utf-8"))
        _update_tensor_fingerprint(digest, buffer)
    if not isinstance(state, Mapping):
        raise RHFDeePHFScannerError(
            "the correction model state_dict must be a mapping"
        )
    for name, value in sorted(state.items()):
        digest.update(b"state\0")
        digest.update(str(name).encode("utf-8"))
        _update_metadata_fingerprint(digest, value, set())
    return digest.hexdigest()


def _validated_root_overlap_tolerance(value) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("scanner root_overlap_tolerance must be a real number")
    try:
        tolerance = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(
            "scanner root_overlap_tolerance must be a real number"
        ) from error
    if not math.isfinite(tolerance) or tolerance <= 0.0 or tolerance > 1.0:
        raise ValueError(
            "scanner root_overlap_tolerance must be finite and in (0, 1]"
        )
    return tolerance


def _immutable_float64_array(value, expected_shape) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != expected_shape:
        raise RHFDeePHFScannerError(
            f"the scanner gradient has shape {array.shape}; expected {expected_shape}"
        )
    if array.dtype != np.dtype(np.float64) or np.iscomplexobj(array):
        raise RHFDeePHFScannerError(
            "the scanner gradient must be a real numpy.float64 array"
        )
    if not np.isfinite(array).all():
        raise RHFDeePHFScannerError("the scanner gradient must be finite")
    contiguous = np.ascontiguousarray(array)
    return np.frombuffer(
        contiguous.tobytes(),
        dtype=contiguous.dtype,
    ).reshape(contiguous.shape)


class RHFDeePHFGradientScanner:
    """Rebuild the complete native RHF DeePHF object graph at each geometry."""

    @property
    def base(self):
        """Return the original gradient driver used as an immutable template."""
        return self._base

    @property
    def backend(self) -> str:
        """Return the fixed analytic-gradient backend."""
        return self._backend

    @property
    def response_options(self):
        """Return an immutable view of the fixed backend options."""
        return self._response_options_view

    def __init__(self, driver, *, root_overlap_tolerance=0.5):
        try:
            base_method = driver.base
            backend = driver.backend
            response_options = driver.response_options
        except AttributeError as error:
            raise TypeError(
                "the scanner requires an RHF DeePHF gradient driver"
            ) from error
        if type(base_method) is not DeePHF:
            raise TypeError(
                "the scanner requires a gradient driver bound to an exact DeePHF method"
            )
        if backend not in {"direct", "zvector"}:
            raise ValueError("the scanner gradient backend must be direct or zvector")
        if not isinstance(response_options, Mapping):
            raise TypeError("the scanner response_options must be a mapping")
        tolerance = _validated_root_overlap_tolerance(root_overlap_tolerance)
        try:
            method_options = deepcopy(base_method.response_options)
            driver_options = deepcopy(dict(response_options))
            projector_basis = deepcopy(base_method._descriptor.projector_basis)
        except Exception as error:
            raise RHFDeePHFScannerError(
                f"the scanner configuration could not be copied: {error}"
            ) from error

        self._base = driver
        self._backend = backend
        self._response_options_view = MappingProxyType(deepcopy(driver_options))
        self._driver_response_options = driver_options
        self._method_response_options = method_options
        self._projector_basis = projector_basis
        self._model = base_method.model
        self._device = base_method.device
        self._atom_domain = _AtomDomain(natm=int(base_method.mol.natm))
        self._reference_factory = RHFScannerReferenceFactory(
            base_method.reference,
            root_overlap_tolerance=tolerance,
        )
        self._root_anchor = self._reference_factory.initial_root
        self._current = None
        self._clear_public_result()

    def _clear_public_result(self) -> None:
        self._current = None
        self.mol = None
        self.reference = None
        self.method = None
        self.gradient_driver = None
        self.e_tot = None
        self.de = None
        self.converged = False
        self.model_state_fingerprint = None

    def _publish(self, result: _ScannerResult) -> None:
        self._current = result
        self.mol = result.mol
        self.reference = result.reference
        self.method = result.method
        self.gradient_driver = result.gradient_driver
        self.e_tot = result.e_tot
        self.de = result.de
        self.converged = True
        self.model_state_fingerprint = result.model_state_fingerprint

    def __call__(self, mol_or_coordinates, *, atmlst=None):
        """Return fresh-reference DeePHF energy and nuclear gradient."""
        self._clear_public_result()
        atom_indices = _validate_atom_indices(self._atom_domain, atmlst)
        model_fingerprint = _model_state_fingerprint(self._model)
        fresh_reference, candidate_root = self._reference_factory.build(
            mol_or_coordinates,
            self._root_anchor,
        )
        if fresh_reference.mol.natm != self._atom_domain.natm:
            raise RHFDeePHFScannerError(
                "the fresh scanner reference changed the raw atom count"
            )
        method = DeePHF(
            fresh_reference,
            self._model,
            projector_basis=deepcopy(self._projector_basis),
            device=self._device,
            response_options=deepcopy(self._method_response_options),
        )
        energy = float(method.kernel())
        if not np.isfinite(energy):
            raise RHFDeePHFScannerError(
                "the scanner total DeePHF energy must be finite"
            )
        if _model_state_fingerprint(self._model) != model_fingerprint:
            raise RHFDeePHFScannerError(
                "the correction model state changed during scanner energy evaluation"
            )
        gradient_driver = method.nuc_grad_method(
            backend=self.backend,
            **deepcopy(self._driver_response_options),
        )
        if gradient_driver.base is not method or gradient_driver.backend != self.backend:
            raise RHFDeePHFScannerError(
                "the scanner gradient driver does not preserve its method and backend"
            )
        gradient = gradient_driver.kernel(atmlst=atom_indices)
        final_model_fingerprint = _model_state_fingerprint(self._model)
        if final_model_fingerprint != model_fingerprint:
            raise RHFDeePHFScannerError(
                "the correction model state changed during scanner gradient evaluation"
            )
        expected_atom_count = (
            fresh_reference.mol.natm
            if atom_indices is None
            else len(atom_indices)
        )
        immutable_gradient = _immutable_float64_array(
            gradient,
            (expected_atom_count, 3),
        )
        result = _ScannerResult(
            mol=fresh_reference.mol,
            reference=fresh_reference,
            method=method,
            gradient_driver=gradient_driver,
            e_tot=energy,
            de=immutable_gradient,
            model_state_fingerprint=final_model_fingerprint,
        )
        self._publish(result)
        self._root_anchor = candidate_root
        return energy, immutable_gradient.copy()


__all__ = ["RHFDeePHFGradientScanner", "RHFDeePHFScannerError"]
