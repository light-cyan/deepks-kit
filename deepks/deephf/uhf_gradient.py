"""Strict direct-oracle nuclear gradients for UHF DeePHF."""

import numpy as np

from .capabilities import science_state_transaction
from .gradient import (
    _reset_driver_results,
    _validate_atom_indices,
    _validate_retain_details,
)
from .pyscf_uhf import UHFResponseError, _native_unrestricted_gradient


class UHFDeePHFGradients:
    """Contract the complete coupled UHF response with one correction model."""

    def __init__(self, method, response_options=None, retain_details=True):
        from .uhf_method import UHFDeePHF

        if type(method) is not UHFDeePHF:
            raise TypeError(
                "the UHF direct gradient driver requires an exact UHFDeePHF method"
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
        from .uhf_method import UHFDeePHF

        if (
            type(self._base) is not UHFDeePHF
            or self._base is not self._bound_base
            or self._mol is not self._bound_mol
            or self._mol is not self._base.mol
            or self._backend != "direct"
        ):
            raise UHFResponseError(
                "the UHF direct gradient driver binding is invalid"
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
        self.base._validate_science_state("UHF native gradient evaluation")
        reference_gradient = _native_unrestricted_gradient(
            self.base.reference,
            self.base.reference.nuc_grad_method(),
            atom_indices,
        )
        self.base._validate_science_state("UHF native gradient evaluation")
        dq_explicit_spin = self.base.dq_dR_explicit_spin(
            atom_indices=atom_indices
        )
        dq_dP = self.base.dq_dP()
        spin_density_response, metric_density, occupied_virtual_density = (
            np.stack(partition) for partition in density_partitions
        )
        dq_response_spin = np.einsum(
            "apij,sbxij->sbxap",
            dq_dP,
            spin_density_response,
        )
        dq_relaxed_spin = dq_explicit_spin + dq_response_spin
        dq_explicit = dq_explicit_spin.sum(axis=0)
        dq_response = dq_response_spin.sum(axis=0)
        dq_relaxed = dq_relaxed_spin.sum(axis=0)
        objective_ao_potential = self.base._correction_ao_potential(
            sensitivity,
            dq_dP,
        )
        correction_explicit_spin = np.einsum(
            "sbxap,ap->sbx",
            dq_explicit_spin,
            sensitivity,
        )
        correction_metric_spin = np.einsum(
            "ij,sbxij->sbx",
            objective_ao_potential,
            metric_density,
        )
        correction_occupied_virtual_spin = np.einsum(
            "ij,sbxij->sbx",
            objective_ao_potential,
            occupied_virtual_density,
        )
        correction_response_spin = np.einsum(
            "sbxap,ap->sbx",
            dq_response_spin,
            sensitivity,
        )
        correction_spin = correction_explicit_spin + correction_response_spin
        correction_explicit = correction_explicit_spin.sum(axis=0)
        correction_metric = correction_metric_spin.sum(axis=0)
        correction_occupied_virtual = correction_occupied_virtual_spin.sum(axis=0)
        correction_response = correction_response_spin.sum(axis=0)
        correction = correction_spin.sum(axis=0)
        de_full = reference_gradient + correction
        if not np.isfinite(de_full).all():
            raise UHFResponseError("the UHF DeePHF gradient is nonfinite")
        self.base._validate_science_state("UHF gradient assembly")
        return {
            "response_result": response_result,
            "descriptor_diagnostics": descriptor_diagnostics,
            "reference_gradient": reference_gradient,
            "dq_dR_explicit_spin": dq_explicit_spin,
            "dq_dR_response_spin": dq_response_spin,
            "dq_dR_relaxed_spin": dq_relaxed_spin,
            "dq_dR_explicit": dq_explicit,
            "dq_dR_response": dq_response,
            "dq_dR_relaxed": dq_relaxed,
            "correction_gradient_explicit_spin": correction_explicit_spin,
            "correction_gradient_metric_spin": correction_metric_spin,
            "correction_gradient_occupied_virtual_spin": (
                correction_occupied_virtual_spin
            ),
            "correction_gradient_response_spin": correction_response_spin,
            "correction_gradient_spin": correction_spin,
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
        self.base._validate_science_state("UHF native gradient evaluation")
        reference_gradient = _native_unrestricted_gradient(
            self.base.reference,
            self.base.reference.nuc_grad_method(),
            atom_indices,
        )
        self.base._validate_science_state("UHF native gradient evaluation")
        explicit = self.base._correction_gradient_explicit(
            sensitivity,
            atom_indices,
        )
        objective = self.base._correction_ao_potential(sensitivity)
        response = sum(
            np.einsum("ij,bxij->bx", objective, spin_density)
            for spin_density in density_partitions[0]
        )
        total = reference_gradient + explicit + response
        if total.shape != (len(atom_indices), 3) or not np.isfinite(total).all():
            raise UHFResponseError("the compact UHF gradient is invalid")
        self.base._validate_science_state("UHF gradient assembly")
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
        """Reject unavailable unrestricted scanner construction."""
        raise UHFResponseError(
            "UHF DeePHF does not provide a gradient scanner"
        )


__all__ = ["UHFDeePHFGradients"]
