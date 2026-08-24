"""Strict direct-oracle nuclear gradients for finite-grid UKS DeePHF."""

import numpy as np

from .pyscf_uks_reference import UKSResponseError
from .pyscf_uks_response import native_uks_gradient
from .uhf_gradient import UHFDeePHFGradients


class UKSDeePHFGradients(UHFDeePHFGradients):
    """Contract the complete coupled UKS response with one correction model."""

    _binding_error_type = UKSResponseError
    _binding_error_message = "the UKS direct gradient driver binding is invalid"
    _construction_error_message = (
        "the UKS direct gradient driver requires an exact UKSDeePHF method"
    )

    @classmethod
    def _expected_method_type(cls):
        from .uks_method import UKSDeePHF

        return UKSDeePHF

    def __init__(self, method, response_options=None, retain_details=True):
        super().__init__(method, response_options, retain_details)

    def _detail_kernel(self, atom_indices) -> dict:
        descriptor_diagnostics, sensitivity = self.base._force_inputs()
        response, density_partitions = self.base._solve_response(
            self.response_options,
            atom_indices=atom_indices,
            result_mode="partitions",
        )
        self.base._validate_science_state("UKS native gradient evaluation")
        native = native_uks_gradient(
            self.base.reference,
            atom_indices=atom_indices,
        )
        self.base._validate_science_state("UKS native gradient evaluation")
        dq_explicit_spin = self.base.dq_dR_explicit_spin(
            atom_indices=atom_indices
        )
        dq_dP = self.base._dq_dP()
        spin_density_response, metric_density, ov_density = density_partitions
        dq_response_spin = np.stack(
            [np.einsum("apij,bxij->bxap", dq_dP, density) for density in spin_density_response]
        )
        dq_relaxed_spin = dq_explicit_spin + dq_response_spin
        dq_explicit = dq_explicit_spin.sum(axis=0)
        dq_response = dq_response_spin.sum(axis=0)
        dq_relaxed = dq_relaxed_spin.sum(axis=0)
        objective = self.base._correction_ao_potential(sensitivity, dq_dP)
        correction_explicit_spin = np.einsum("sbxap,ap->sbx", dq_explicit_spin, sensitivity)
        correction_metric_spin = np.stack([np.einsum("ij,bxij->bx", objective, density) for density in metric_density])
        correction_ov_spin = np.stack([np.einsum("ij,bxij->bx", objective, density) for density in ov_density])
        correction_response_spin = np.einsum("sbxap,ap->sbx", dq_response_spin, sensitivity)
        correction_spin = correction_explicit_spin + correction_response_spin
        correction_explicit = correction_explicit_spin.sum(axis=0)
        correction_metric = correction_metric_spin.sum(axis=0)
        correction_ov = correction_ov_spin.sum(axis=0)
        correction_response = correction_response_spin.sum(axis=0)
        correction = correction_spin.sum(axis=0)
        total = native + correction
        if not np.isfinite(total).all():
            raise UKSResponseError("the UKS DeePHF direct gradient is nonfinite")
        self.base._validate_science_state("UKS direct gradient assembly")
        return {
            "response_result": response,
            "descriptor_diagnostics": descriptor_diagnostics,
            "reference_gradient": native,
            "dq_dR_explicit_spin": dq_explicit_spin,
            "dq_dR_response_spin": dq_response_spin,
            "dq_dR_relaxed_spin": dq_relaxed_spin,
            "dq_dR_explicit": dq_explicit,
            "dq_dR_response": dq_response,
            "dq_dR_relaxed": dq_relaxed,
            "correction_gradient_explicit_spin": correction_explicit_spin,
            "correction_gradient_metric_spin": correction_metric_spin,
            "correction_gradient_occupied_virtual_spin": correction_ov_spin,
            "correction_gradient_response_spin": correction_response_spin,
            "correction_gradient_spin": correction_spin,
            "correction_gradient_explicit": correction_explicit,
            "correction_gradient_metric": correction_metric,
            "correction_gradient_occupied_virtual": correction_ov,
            "correction_gradient_response": correction_response,
            "correction_gradient": correction,
            "de_full": total,
        }

    def _compact_kernel(self, atom_indices) -> dict:
        diagnostics, sensitivity = self.base._force_inputs()
        reference = native_uks_gradient(self.base.reference, atom_indices)
        if not np.any(sensitivity):
            if (
                reference.shape != (len(atom_indices), 3)
                or reference.dtype != np.dtype(np.float64)
                or np.iscomplexobj(reference)
                or not np.isfinite(reference).all()
            ):
                raise UKSResponseError("the compact UKS native gradient is invalid")
            return {
                "descriptor_diagnostics": diagnostics,
                "response_diagnostics": None,
                "de": reference,
            }
        explicit, objective = self.base._correction_derivatives(sensitivity, atom_indices)
        response_diagnostics, response = self.base._solve_response(
            self.response_options,
            atom_indices=atom_indices,
            result_mode="gradient",
            objective=objective,
        )
        self.base._validate_science_state("UKS native gradient evaluation")
        total = reference + explicit + response
        if total.shape != (len(atom_indices), 3) or not np.isfinite(total).all():
            raise UKSResponseError("the compact UKS gradient is invalid")
        self.base._validate_science_state("UKS direct gradient assembly")
        return {
            "descriptor_diagnostics": diagnostics,
            "response_diagnostics": response_diagnostics,
            "de": total,
        }

    def as_scanner(self, **scanner_options):
        raise UKSResponseError("UKS DeePHF does not provide a gradient scanner")


__all__ = ["UKSDeePHFGradients"]
