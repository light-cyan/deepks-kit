"""Strict direct-oracle nuclear gradients for UHF DeePHF."""

import numpy as np

from .driver import GradientDriver
from .pyscf_uhf_reference import UHFResponseError, _native_unrestricted_gradient


class UHFDeePHFGradients(GradientDriver):
    """Contract the complete coupled UHF response with one correction model."""

    _backend_name = "direct"
    _binding_error_type = UHFResponseError
    _binding_error_message = "the UHF direct gradient driver binding is invalid"
    _construction_error_message = (
        "the UHF direct gradient driver requires an exact UHFDeePHF method"
    )

    @classmethod
    def _expected_method_type(cls):
        from .uhf_method import UHFDeePHF

        return UHFDeePHF

    def __init__(self, method, response_options=None, retain_details=True):
        super().__init__(method, response_options, retain_details)

    def _detail_kernel(self, atom_indices) -> dict:
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
        dq_dP = self.base._dq_dP()
        spin_density_response, metric_density, occupied_virtual_density = density_partitions
        dq_response_spin = np.stack(
            [np.einsum("apij,bxij->bxap", dq_dP, density) for density in spin_density_response]
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
        correction_metric_spin = np.stack(
            [np.einsum("ij,bxij->bx", objective_ao_potential, density) for density in metric_density]
        )
        correction_occupied_virtual_spin = np.stack(
            [np.einsum("ij,bxij->bx", objective_ao_potential, density) for density in occupied_virtual_density]
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
        reference_gradient = _native_unrestricted_gradient(
            self.base.reference,
            self.base.reference.nuc_grad_method(),
            atom_indices,
        )
        if not np.any(sensitivity):
            if (
                reference_gradient.shape != (len(atom_indices), 3)
                or reference_gradient.dtype != np.dtype(np.float64)
                or np.iscomplexobj(reference_gradient)
                or not np.isfinite(reference_gradient).all()
            ):
                raise UHFResponseError("the compact UHF native gradient is invalid")
            return {
                "descriptor_diagnostics": descriptor_diagnostics,
                "response_diagnostics": None,
                "de": reference_gradient,
            }
        explicit, objective = self.base._correction_derivatives(
            sensitivity,
            atom_indices,
        )
        response_diagnostics, response = self.base._solve_response(
            self.response_options,
            atom_indices=atom_indices,
            result_mode="gradient",
            objective=objective,
        )
        self.base._validate_science_state("UHF native gradient evaluation")
        total = reference_gradient + explicit + response
        if total.shape != (len(atom_indices), 3) or not np.isfinite(total).all():
            raise UHFResponseError("the compact UHF gradient is invalid")
        self.base._validate_science_state("UHF gradient assembly")
        return {
            "descriptor_diagnostics": descriptor_diagnostics,
            "response_diagnostics": response_diagnostics,
            "de": total,
        }

    def as_scanner(self, **scanner_options):
        """Reject unavailable unrestricted scanner construction."""
        raise UHFResponseError(
            "UHF DeePHF does not provide a gradient scanner"
        )


__all__ = ["UHFDeePHFGradients"]
