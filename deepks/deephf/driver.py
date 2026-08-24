"""Shared lifecycle for direct and scalar-adjoint DeePHF gradient drivers."""

from types import MappingProxyType
import operator

import numpy as np

from .capabilities import science_state_transaction


def validate_atom_indices(mol, atmlst):
    """Return unique raw-atom indices before any coordinate calculation."""
    if atmlst is None:
        return None
    try:
        requested_indices = tuple(atmlst)
    except TypeError as error:
        raise TypeError("gradient atmlst must be an iterable of integers") from error
    if not requested_indices:
        raise ValueError("gradient atmlst must not be empty")
    validated_indices = []
    for index in requested_indices:
        if isinstance(index, (bool, np.bool_)):
            raise TypeError("gradient atom indices must be integers")
        try:
            atom_index = operator.index(index)
        except TypeError as error:
            raise TypeError("gradient atom indices must be integers") from error
        if atom_index < 0 or atom_index >= mol.natm:
            raise IndexError("gradient atom index is outside the molecule")
        if atom_index in validated_indices:
            raise ValueError("gradient atmlst must not contain duplicates")
        validated_indices.append(atom_index)
    return tuple(validated_indices)


def validate_retain_details(value) -> bool:
    """Validate the public result-retention control."""
    if type(value) is not bool:
        raise TypeError("retain_details must be a Boolean")
    return value


_DRIVER_CONFIGURATION = frozenset(
    {
        "_base",
        "_bound_base",
        "_mol",
        "_bound_mol",
        "_backend",
        "_options",
        "_bound_options",
        "response_options",
        "_adjoint_options",
        "_bound_adjoint_options",
        "retain_details",
    }
)


def reset_driver_results(driver) -> None:
    """Clear published results while preserving immutable driver configuration."""
    for name in tuple(vars(driver)):
        if name not in _DRIVER_CONFIGURATION:
            delattr(driver, name)
    driver._response_diagnostics = None
    driver.descriptor_diagnostics = None
    driver.de = None


class GradientDriver:
    """Centralize binding, transaction, selection, publication, and force flow."""

    _backend_name = None
    _binding_error_type = RuntimeError
    _binding_error_message = "the DeePHF gradient driver binding is invalid"
    _construction_error_message = "the DeePHF gradient method type is invalid"

    @classmethod
    def _expected_method_type(cls):
        raise NotImplementedError

    def __init__(self, method, options=None, retain_details=True):
        if type(method) is not self._expected_method_type():
            raise TypeError(self._construction_error_message)
        self._base = method
        self._bound_base = method
        self._mol = method.mol
        self._bound_mol = method.mol
        self._backend = self._backend_name
        self.retain_details = validate_retain_details(retain_details)
        self._options = dict(options or {})
        self._bound_options = self._options
        self.response_options = self._options
        if self._backend_name == "zvector":
            self._adjoint_options = MappingProxyType(self._options)
            self._bound_adjoint_options = self._adjoint_options
        self._reset_results()

    @property
    def base(self):
        return self._base

    @property
    def mol(self):
        return self._mol

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def adjoint_options(self):
        if self._backend_name != "zvector":
            raise AttributeError("direct gradient drivers do not have adjoint options")
        return self._adjoint_options

    def _validate_driver_binding(self) -> None:
        options_match = (
            self.response_options is self._options
            if self._backend_name == "direct"
            else (
                self.response_options is self._options
                and self._adjoint_options is self._bound_adjoint_options
                and isinstance(self._adjoint_options, MappingProxyType)
            )
        )
        if (
            type(self._base) is not self._expected_method_type()
            or self._base is not self._bound_base
            or self._mol is not self._bound_mol
            or self._mol is not self._base.mol
            or self._backend != self._backend_name
            or self._options is not self._bound_options
            or type(self._options) is not dict
            or not options_match
        ):
            raise self._binding_error_type(self._binding_error_message)

    def _reset_results(self) -> None:
        reset_driver_results(self)

    @property
    def response_diagnostics(self):
        response = getattr(self, "response_result", None)
        if response is None:
            response = getattr(self, "adjoint_result", None)
        return (
            self._response_diagnostics
            if response is None
            else response.diagnostics
        )

    @property
    def adjoint_diagnostics(self):
        if self._backend_name != "zvector":
            raise AttributeError("direct gradient drivers do not have adjoint diagnostics")
        return self.response_diagnostics

    def _calculation_atom_indices(self, atom_indices):
        return (
            tuple(range(self.mol.natm))
            if atom_indices is None
            else atom_indices
        )

    @science_state_transaction
    def kernel(self, atmlst=None) -> np.ndarray:
        """Run one atomically published gradient calculation."""
        self._reset_results()
        self._validate_driver_binding()
        atom_indices = validate_atom_indices(self.mol, atmlst)
        calculation_atom_indices = self._calculation_atom_indices(atom_indices)
        results = (
            self._detail_kernel(calculation_atom_indices)
            if self.retain_details
            else self._compact_kernel(calculation_atom_indices)
        )
        for name, value in results.items():
            if name == "response_diagnostics":
                self._response_diagnostics = value
            else:
                setattr(self, name, value)
        if self.retain_details:
            self.de = self.de_full
        return self.de

    def run(self, atmlst=None):
        """Evaluate the gradient and return this populated driver."""
        self.kernel(atmlst=atmlst)
        return self

    def forces(self, atmlst=None) -> np.ndarray:
        """Evaluate nuclear forces as minus the energy gradient."""
        return -self.kernel(atmlst=atmlst)

    def as_scanner(self, **scanner_options):
        """Build the strict fresh-reference RHF gradient scanner."""
        from .scanner import RHFDeePHFGradientScanner

        return RHFDeePHFGradientScanner(self, **scanner_options)


__all__ = [
    "GradientDriver",
    "reset_driver_results",
    "validate_atom_indices",
    "validate_retain_details",
]
