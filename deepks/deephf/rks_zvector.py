"""Strict scalar-adjoint nuclear gradients for finite-grid RKS DeePHF."""

from types import MappingProxyType

import numpy as np

from .capabilities import science_state_transaction
from .gradient import (
    _reset_driver_results,
    _validate_atom_indices,
    _validate_retain_details,
)
from .pyscf_rks import (
    RKSAdjointError,
    native_rks_gradient,
)


class RKSDeePHFZVectorGradients:
    """Evaluate one pure-LDA RKS correction through a scalar adjoint."""

    def __init__(self, method, adjoint_options=None, retain_details=True):
        from .rks_method import RKSDeePHF

        if type(method) is not RKSDeePHF:
            raise TypeError(
                "the RKS Z-vector gradient driver requires an exact RKSDeePHF method"
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
        from .rks_method import RKSDeePHF

        if (
            type(self._base) is not RKSDeePHF
            or self._base is not self._bound_base
            or self._mol is not self._bound_mol
            or self._mol is not self._base.mol
            or self._backend != "zvector"
            or self._adjoint_options is not self._bound_adjoint_options
            or not isinstance(self._adjoint_options, MappingProxyType)
        ):
            raise RKSAdjointError(
                "the RKS DeePHF Z-vector driver binding is invalid"
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

    def _kernel(self, atom_indices) -> dict:
        descriptor_diagnostics, sensitivity, adjoint = self.base._zvector_inputs(
            self.adjoint_options,
            atom_indices=atom_indices,
        )
        self.base._validate_science_state(
            "RKS Z-vector native gradient evaluation"
        )
        native = native_rks_gradient(
            self.base.reference,
            atom_indices=atom_indices,
        )
        self.base._validate_science_state(
            "RKS Z-vector native gradient evaluation"
        )
        dq_explicit = self.base.dq_dR_explicit(atom_indices=atom_indices)
        self.base._validate_science_state(
            "RKS Z-vector explicit descriptor gradient evaluation"
        )
        correction_explicit = np.einsum(
            "bxap,ap->bx",
            dq_explicit,
            sensitivity,
        )
        correction_metric = np.asarray(adjoint.correction_gradient_metric)
        correction_adjoint_fixed_grid = np.asarray(
            adjoint.correction_gradient_adjoint_fixed_grid
        )
        correction_adjoint_grid_coordinate = np.asarray(
            adjoint.correction_gradient_adjoint_grid_coordinate
        )
        correction_adjoint_grid_weight = np.asarray(
            adjoint.correction_gradient_adjoint_grid_weight
        )
        correction_adjoint_nuclear = np.asarray(
            adjoint.correction_gradient_adjoint_nuclear
        )
        correction_adjoint_metric = np.asarray(
            adjoint.correction_gradient_adjoint_metric
        )
        correction_occupied_virtual = np.asarray(
            adjoint.correction_gradient_occupied_virtual
        )
        correction_response = np.asarray(
            adjoint.correction_gradient_response
        )
        correction = correction_explicit + correction_response
        total = native + correction
        expected_shape = (len(adjoint.atom_indices), 3)
        if total.shape != expected_shape or not np.isfinite(total).all():
            raise RKSAdjointError("the RKS DeePHF Z-vector gradient is invalid")
        self.base._validate_science_state("RKS Z-vector gradient assembly")
        return {
            "adjoint_result": adjoint,
            "descriptor_diagnostics": descriptor_diagnostics,
            "reference_gradient": native,
            "dq_dR_explicit": dq_explicit,
            "correction_gradient_explicit": correction_explicit,
            "correction_gradient_metric": correction_metric,
            "correction_gradient_adjoint_fixed_grid": (
                correction_adjoint_fixed_grid
            ),
            "correction_gradient_adjoint_grid_coordinate": (
                correction_adjoint_grid_coordinate
            ),
            "correction_gradient_adjoint_grid_weight": (
                correction_adjoint_grid_weight
            ),
            "correction_gradient_adjoint_nuclear": correction_adjoint_nuclear,
            "correction_gradient_adjoint_metric": correction_adjoint_metric,
            "correction_gradient_grid_coordinate": (
                correction_adjoint_grid_coordinate
            ),
            "correction_gradient_grid_weight": correction_adjoint_grid_weight,
            "correction_gradient_occupied_virtual": (
                correction_occupied_virtual
            ),
            "correction_gradient_response": correction_response,
            "correction_gradient": correction,
            "de_full": total,
        }

    def _compact_kernel(self, atom_indices) -> dict:
        diagnostics, explicit, adjoint_diagnostics, response_gradient = self.base._zvector_inputs(
            self.adjoint_options,
            atom_indices=atom_indices,
            compact=True,
        )
        self.base._validate_science_state("RKS Z-vector native gradient evaluation")
        reference = native_rks_gradient(self.base.reference, atom_indices)
        self.base._validate_science_state("RKS Z-vector native gradient evaluation")
        total = reference + explicit + response_gradient
        if total.shape != reference.shape or not np.isfinite(total).all():
            raise RKSAdjointError("the compact RKS Z-vector gradient is invalid")
        self.base._validate_science_state("RKS Z-vector gradient assembly")
        return {
            "descriptor_diagnostics": diagnostics,
            "response_diagnostics": adjoint_diagnostics,
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
        """Reject unavailable RKS scanner construction."""
        raise RKSAdjointError(
            "RKS DeePHF does not provide a gradient scanner"
        )


__all__ = ["RKSDeePHFZVectorGradients"]
