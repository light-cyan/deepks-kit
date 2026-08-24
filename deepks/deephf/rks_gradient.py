"""Strict direct-oracle nuclear gradients for finite-grid RKS DeePHF."""

import numpy as np

from .driver import GradientDriver
from .pyscf_dft_provenance import RKSResponseError
from .pyscf_rks_native import native_rks_gradient


class RKSDeePHFGradients(GradientDriver):
    """Contract the complete pure-LDA RKS response with one correction model."""

    _backend_name = "direct"
    _binding_error_type = RKSResponseError
    _binding_error_message = "the RKS direct gradient driver binding is invalid"
    _construction_error_message = (
        "the RKS direct gradient driver requires an exact RKSDeePHF method"
    )

    @classmethod
    def _expected_method_type(cls):
        from .rks_method import RKSDeePHF

        return RKSDeePHF

    def __init__(self, method, response_options=None, retain_details=True):
        super().__init__(method, response_options, retain_details)

    def _detail_kernel(self, atom_indices) -> dict:
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
        dq_dP = self.base._dq_dP()
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
        reference_gradient = native_rks_gradient(
            self.base.reference,
            atom_indices,
        )
        if not np.any(sensitivity):
            if (
                reference_gradient.shape != (len(atom_indices), 3)
                or reference_gradient.dtype != np.dtype(np.float64)
                or np.iscomplexobj(reference_gradient)
                or not np.isfinite(reference_gradient).all()
            ):
                raise RKSResponseError("the compact RKS native gradient is invalid")
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
        self.base._validate_science_state("RKS native gradient evaluation")
        total = reference_gradient + explicit + response
        if total.shape != (len(atom_indices), 3) or not np.isfinite(total).all():
            raise RKSResponseError("the compact RKS gradient is invalid")
        self.base._validate_science_state("RKS gradient assembly")
        return {
            "descriptor_diagnostics": descriptor_diagnostics,
            "response_diagnostics": response_diagnostics,
            "de": total,
        }

    def as_scanner(self, **scanner_options):
        """Reject unavailable RKS scanner construction."""
        raise RKSResponseError(
            "RKS DeePHF does not provide a gradient scanner"
        )


__all__ = ["RKSDeePHFGradients"]
