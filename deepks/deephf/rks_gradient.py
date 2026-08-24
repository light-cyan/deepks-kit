"""Strict direct-oracle nuclear gradients for finite-grid RKS DeePHF."""

import numpy as np

from .capabilities import science_state_transaction
from .gradient import (
    _reset_driver_results,
    _validate_atom_indices,
    _validate_retain_details,
)
from .pyscf_rks import (
    RKSResponseError,
    native_rks_gradient,
)


class RKSDeePHFGradients:
    """Contract the complete pure-LDA RKS response with one correction model."""

    def __init__(self, method, response_options=None, retain_details=True):
        from .rks_method import RKSDeePHF

        if type(method) is not RKSDeePHF:
            raise TypeError(
                "the RKS direct gradient driver requires an exact RKSDeePHF method"
            )
        self._base = method
        self._bound_base = method
        self._mol = method.mol
        self._bound_mol = method.mol
        self._backend = "direct"
        self.retain_details = _validate_retain_details(retain_details)
        self.response_options = dict(response_options or {})
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

    def _validate_driver_binding(self) -> None:
        from .rks_method import RKSDeePHF

        if (
            type(self._base) is not RKSDeePHF
            or self._base is not self._bound_base
            or self._mol is not self._bound_mol
            or self._mol is not self._base.mol
            or self._backend != "direct"
        ):
            raise RKSResponseError(
                "the RKS direct gradient driver binding is invalid"
            )

    def _reset_results(self) -> None:
        _reset_driver_results(self)

    @property
    def response_diagnostics(self):
        return (
            self._response_diagnostics
            if getattr(self, "response_result", None) is None
            else self.response_result.diagnostics
        )

    def _kernel(self, atom_indices) -> dict:
        descriptor_diagnostics, sensitivity = self.base._force_inputs()
        response_result, density_partitions = self.base._solve_response(
            self.response_options,
            atom_indices=atom_indices,
            result_mode="partitions",
        )
        self.base._validate_science_state("RKS native gradient evaluation")
        reference_gradient = native_rks_gradient(
            self.base.reference,
            atom_indices=atom_indices,
        )
        self.base._validate_science_state("RKS native gradient evaluation")
        dq_explicit = self.base.dq_dR_explicit(atom_indices=atom_indices)
        dq_dP = self.base.dq_dP()
        density, density_metric, density_occupied_virtual = density_partitions
        dq_response = np.einsum(
            "apij,bxij->bxap",
            dq_dP,
            density,
        )
        dq_relaxed = dq_explicit + dq_response
        objective_ao_potential = self.base._correction_ao_potential(
            sensitivity,
            dq_dP,
        )
        correction_explicit = np.einsum(
            "bxap,ap->bx",
            dq_explicit,
            sensitivity,
        )
        correction_metric = np.einsum(
            "ij,bxij->bx",
            objective_ao_potential,
            density_metric,
        )
        correction_occupied_virtual = np.einsum(
            "ij,bxij->bx",
            objective_ao_potential,
            density_occupied_virtual,
        )
        correction_response = np.einsum(
            "bxap,ap->bx",
            dq_response,
            sensitivity,
        )
        correction = correction_explicit + correction_response
        de_full = reference_gradient + correction
        if not np.isfinite(de_full).all():
            raise RKSResponseError("the RKS DeePHF gradient is nonfinite")
        self.base._validate_science_state("RKS gradient assembly")
        return {
            "response_result": response_result,
            "descriptor_diagnostics": descriptor_diagnostics,
            "reference_gradient": reference_gradient,
            "dq_dR_explicit": dq_explicit,
            "dq_dR_response": dq_response,
            "dq_dR_relaxed": dq_relaxed,
            "correction_gradient_explicit": correction_explicit,
            "correction_gradient_metric": correction_metric,
            "correction_gradient_occupied_virtual": (
                correction_occupied_virtual
            ),
            "correction_gradient_response": correction_response,
            "correction_gradient": correction,
            "de_full": de_full,
        }

    def _compact_kernel(self, atom_indices) -> dict:
        descriptor_diagnostics, sensitivity = self.base._force_inputs()
        response_diagnostics, density_partitions = self.base._solve_response(
            self.response_options,
            atom_indices=atom_indices,
            result_mode="gradient",
        )
        self.base._validate_science_state("RKS native gradient evaluation")
        reference_gradient = native_rks_gradient(
            self.base.reference,
            atom_indices,
        )
        self.base._validate_science_state("RKS native gradient evaluation")
        explicit = self.base._correction_gradient_explicit(
            sensitivity,
            atom_indices,
        )
        objective = self.base._correction_ao_potential(sensitivity)
        response = np.einsum("ij,bxij->bx", objective, density_partitions[0])
        total = reference_gradient + explicit + response
        if total.shape != (len(atom_indices), 3) or not np.isfinite(total).all():
            raise RKSResponseError("the compact RKS gradient is invalid")
        self.base._validate_science_state("RKS gradient assembly")
        return {
            "descriptor_diagnostics": descriptor_diagnostics,
            "response_diagnostics": response_diagnostics,
            "de": total,
        }

    @science_state_transaction
    def kernel(self, atmlst=None) -> np.ndarray:
        """Evaluate d(E_base + E_corr)/dR for all or selected atoms."""
        self._reset_results()
        try:
            self._validate_driver_binding()
            atom_indices = _validate_atom_indices(self.mol, atmlst)
            calculation_atom_indices = (
                tuple(range(self.mol.natm))
                if atom_indices is None
                else atom_indices
            )
            if not self.retain_details:
                results = self._compact_kernel(calculation_atom_indices)
                self.descriptor_diagnostics = results["descriptor_diagnostics"]
                self._response_diagnostics = results["response_diagnostics"]
                self.de = results["de"]
                return self.de
            results = self._kernel(calculation_atom_indices)
            for name, value in results.items():
                setattr(self, name, value)
            self.de = self.de_full
            return self.de
        except Exception:
            self._reset_results()
            raise

    def run(self, atmlst=None):
        """Evaluate the gradient and return this result object."""
        self.kernel(atmlst=atmlst)
        return self

    def forces(self, atmlst=None) -> np.ndarray:
        """Evaluate nuclear forces as minus the energy gradient."""
        return -self.kernel(atmlst=atmlst)

    def as_scanner(self, **scanner_options):
        """Reject unavailable RKS scanner construction."""
        raise RKSResponseError(
            "RKS DeePHF does not provide a gradient scanner"
        )


__all__ = ["RKSDeePHFGradients"]
