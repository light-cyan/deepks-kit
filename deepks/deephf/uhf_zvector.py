"""Strict coupled scalar-adjoint nuclear gradients for UHF DeePHF."""

from types import MappingProxyType

import numpy as np

from .capabilities import science_state_transaction
from .gradient import (
    _reset_driver_results,
    _validate_atom_indices,
    _validate_retain_details,
)
from .pyscf_uhf import UHFAdjointError, _native_unrestricted_gradient


class UHFDeePHFZVectorGradients:
    """Evaluate one unrestricted correction through a coupled scalar adjoint."""

    def __init__(self, method, adjoint_options=None, retain_details=True):
        from .uhf_method import UHFDeePHF

        if type(method) is not UHFDeePHF:
            raise TypeError(
                "the UHF Z-vector gradient driver requires an exact UHFDeePHF method"
            )
        self._base = method
        self._bound_base = method
        self._mol = method.mol
        self._bound_mol = method.mol
        self._backend = "zvector"
        self.retain_details = _validate_retain_details(retain_details)
        self._adjoint_options = MappingProxyType(dict(adjoint_options or {}))
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
        return self._adjoint_options

    def _validate_driver_binding(self) -> None:
        from .uhf_method import UHFDeePHF

        if (
            type(self._base) is not UHFDeePHF
            or self._base is not self._bound_base
            or self._mol is not self._bound_mol
            or self._mol is not self._base.mol
            or self._backend != "zvector"
            or self._adjoint_options is not self._bound_adjoint_options
            or not isinstance(self._adjoint_options, MappingProxyType)
        ):
            raise UHFAdjointError(
                "the UHF DeePHF Z-vector driver binding is invalid"
            )

    def _reset_results(self) -> None:
        _reset_driver_results(self)

    @property
    def adjoint_diagnostics(self):
        return (
            self._response_diagnostics
            if getattr(self, "adjoint_result", None) is None
            else self.adjoint_result.diagnostics
        )

    @property
    def response_diagnostics(self):
        """Return scalar-adjoint diagnostics under the common driver name."""
        return self.adjoint_diagnostics

    def _validated_native_gradient(self, atom_indices) -> np.ndarray:
        self.base._validate_science_state("UHF Z-vector native gradient evaluation")
        try:
            gradient = _native_unrestricted_gradient(
                self.base.reference,
                self.base.reference.nuc_grad_method(),
                range(self.mol.natm) if atom_indices is None else atom_indices,
            )
        except Exception as error:
            raise UHFAdjointError(f"native UHF gradient evaluation failed: {error}") from error
        self.base._validate_science_state("UHF Z-vector native gradient evaluation")
        return gradient

    def _kernel(self, atom_indices) -> dict:
        descriptor_diagnostics, sensitivity, adjoint = self.base._zvector_inputs(
            self.adjoint_options,
            atom_indices=atom_indices,
        )
        reference_gradient = self._validated_native_gradient(atom_indices)
        dq_explicit_spin = self.base.dq_dR_explicit_spin(
            atom_indices=atom_indices
        )
        self.base._validate_science_state(
            "UHF Z-vector explicit descriptor gradient evaluation"
        )
        dq_explicit = dq_explicit_spin.sum(axis=0)
        correction_explicit_spin = np.einsum(
            "sbxap,ap->sbx",
            dq_explicit_spin,
            sensitivity,
        )
        correction_metric_spin = np.asarray(
            adjoint.correction_gradient_metric_spin
        )
        correction_adjoint_nuclear_spin = np.asarray(
            adjoint.correction_gradient_adjoint_nuclear_spin
        )
        correction_adjoint_metric_spin = np.asarray(
            adjoint.correction_gradient_adjoint_metric_spin
        )
        correction_occupied_virtual_spin = np.asarray(
            adjoint.correction_gradient_occupied_virtual_spin
        )
        correction_response_spin = (
            correction_metric_spin + correction_occupied_virtual_spin
        )
        correction_spin = correction_explicit_spin + correction_response_spin
        correction_explicit = correction_explicit_spin.sum(axis=0)
        correction_metric = np.asarray(adjoint.correction_gradient_metric)
        correction_adjoint_nuclear = np.asarray(
            adjoint.correction_gradient_adjoint_nuclear
        )
        correction_adjoint_metric = np.asarray(
            adjoint.correction_gradient_adjoint_metric
        )
        correction_occupied_virtual = np.asarray(
            adjoint.correction_gradient_occupied_virtual
        )
        correction_response = np.asarray(adjoint.correction_gradient_response)
        correction = correction_explicit + correction_response
        total = reference_gradient + correction
        if total.shape != (len(adjoint.atom_indices), 3) or not np.isfinite(total).all():
            raise UHFAdjointError("the UHF DeePHF Z-vector gradient is invalid")
        self.base._validate_science_state("UHF Z-vector gradient assembly")
        return {
            "adjoint_result": adjoint,
            "descriptor_diagnostics": descriptor_diagnostics,
            "reference_gradient": reference_gradient,
            "dq_dR_explicit_spin": dq_explicit_spin,
            "dq_dR_explicit": dq_explicit,
            "correction_gradient_explicit_spin": correction_explicit_spin,
            "correction_gradient_metric_spin": correction_metric_spin,
            "correction_gradient_adjoint_nuclear_spin": (
                correction_adjoint_nuclear_spin
            ),
            "correction_gradient_adjoint_metric_spin": (
                correction_adjoint_metric_spin
            ),
            "correction_gradient_occupied_virtual_spin": (
                correction_occupied_virtual_spin
            ),
            "correction_gradient_response_spin": correction_response_spin,
            "correction_gradient_spin": correction_spin,
            "correction_gradient_explicit": correction_explicit,
            "correction_gradient_metric": correction_metric,
            "correction_gradient_adjoint_nuclear": correction_adjoint_nuclear,
            "correction_gradient_adjoint_metric": correction_adjoint_metric,
            "correction_gradient_occupied_virtual": (
                correction_occupied_virtual
            ),
            "correction_gradient_response": correction_response,
            "correction_gradient": correction,
            "de_full": total,
        }

    def _compact_kernel(self, atom_indices) -> dict:
        descriptor_diagnostics, sensitivity, adjoint = self.base._zvector_inputs(
            self.adjoint_options,
            atom_indices=atom_indices,
        )
        reference_gradient = self._validated_native_gradient(atom_indices)
        explicit = self.base._correction_gradient_explicit(
            sensitivity,
            atom_indices,
        )
        total = reference_gradient + explicit + adjoint.correction_gradient_response
        if total.shape != (len(adjoint.atom_indices), 3) or not np.isfinite(total).all():
            raise UHFAdjointError("the compact UHF Z-vector gradient is invalid")
        self.base._validate_science_state("UHF Z-vector gradient assembly")
        return {
            "descriptor_diagnostics": descriptor_diagnostics,
            "response_diagnostics": adjoint.diagnostics,
            "de": total,
        }

    @science_state_transaction
    def kernel(self, atmlst=None) -> np.ndarray:
        """Evaluate d(E_base + E_corr)/dR without constructing dP/dR."""
        self._reset_results()
        try:
            self._validate_driver_binding()
            atom_indices = _validate_atom_indices(self.mol, atmlst)
            if not self.retain_details:
                results = self._compact_kernel(atom_indices)
                self.descriptor_diagnostics = results["descriptor_diagnostics"]
                self._response_diagnostics = results["response_diagnostics"]
                self.de = results["de"]
                return self.de
            results = self._kernel(atom_indices)
            for name, value in results.items():
                setattr(self, name, value)
            self.de = self.de_full
            return self.de
        except Exception:
            self._reset_results()
            raise

    def run(self, atmlst=None):
        """Evaluate the gradient and return this populated driver."""
        self.kernel(atmlst=atmlst)
        return self

    def forces(self, atmlst=None) -> np.ndarray:
        """Evaluate nuclear forces as minus the energy gradient."""
        return -self.kernel(atmlst=atmlst)

    def as_scanner(self, **scanner_options):
        """Reject unavailable unrestricted scanner construction."""
        raise UHFAdjointError(
            "UHF DeePHF does not provide a gradient scanner"
        )


__all__ = ["UHFDeePHFZVectorGradients"]
