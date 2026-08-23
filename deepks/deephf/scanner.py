"""Strict fresh-reference geometry scanner for RHF DeePHF gradients."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import math
from types import MappingProxyType

import numpy as np

from .capabilities import (
    DeePHFCapabilityError,
    force_model_fingerprint,
    validate_force_model,
)
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


def _model_state_fingerprint(model) -> str:
    """Return the canonical strict force-model fingerprint for publication."""
    try:
        return force_model_fingerprint(model)
    except DeePHFCapabilityError as error:
        raise RHFDeePHFScannerError(
            f"the scanner correction model is incompatible: {error}"
        ) from error


def _validate_model_inference_preflight(model) -> None:
    """Require stable evaluation semantics before constructing a fresh reference."""
    try:
        validate_force_model(model)
    except DeePHFCapabilityError as error:
        raise RHFDeePHFScannerError(
            f"the scanner correction model is incompatible: {error}"
        ) from error


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
            backend_options = driver.response_options
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
        if not isinstance(backend_options, Mapping):
            raise TypeError("the scanner backend options must be a mapping")
        if not isinstance(base_method.response_options, Mapping):
            raise TypeError("the scanner method response_options must be a mapping")
        if not isinstance(base_method.adjoint_options, Mapping):
            raise TypeError("the scanner method adjoint_options must be a mapping")
        tolerance = _validated_root_overlap_tolerance(root_overlap_tolerance)
        try:
            method_response_options = deepcopy(dict(base_method.response_options))
            method_adjoint_options = deepcopy(dict(base_method.adjoint_options))
            driver_options = deepcopy(dict(backend_options))
            projector_basis = deepcopy(base_method._descriptor.projector_basis)
        except Exception as error:
            raise RHFDeePHFScannerError(
                f"the scanner configuration could not be copied: {error}"
            ) from error

        self._base = driver
        self._backend = backend
        self._response_options_view = MappingProxyType(deepcopy(driver_options))
        self._driver_backend_options = MappingProxyType(driver_options)
        self._method_response_options = MappingProxyType(method_response_options)
        self._method_adjoint_options = MappingProxyType(method_adjoint_options)
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
        _validate_model_inference_preflight(self._model)
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
            response_options=deepcopy(dict(self._method_response_options)),
            adjoint_options=deepcopy(dict(self._method_adjoint_options)),
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
            retain_details=False,
            **deepcopy(dict(self._driver_backend_options)),
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
