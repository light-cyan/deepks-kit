"""Strict coupled scalar-adjoint nuclear gradients for UKS DeePHF."""

import numpy as np

from .pyscf_uks_reference import UKSAdjointError
from .pyscf_uks_response import native_uks_gradient
from .uhf_zvector import UHFDeePHFZVectorGradients


class UKSDeePHFZVectorGradients(UHFDeePHFZVectorGradients):
    """Evaluate one finite-grid UKS correction through one coupled adjoint."""

    _binding_error_type = UKSAdjointError
    _binding_error_message = "the UKS DeePHF Z-vector driver binding is invalid"
    _construction_error_message = (
        "the UKS Z-vector gradient driver requires an exact UKSDeePHF method"
    )

    @classmethod
    def _expected_method_type(cls):
        from .uks_method import UKSDeePHF

        return UKSDeePHF

    def __init__(self, method, adjoint_options=None, retain_details=True):
        super().__init__(method, adjoint_options, retain_details)

    def _detail_kernel(self, atom_indices) -> dict:
        diagnostics, sensitivity, adjoint = self.base._zvector_inputs(
            self.adjoint_options,
            atom_indices=atom_indices,
        )
        self.base._validate_science_state("UKS Z-vector native gradient evaluation")
        native = native_uks_gradient(
            self.base.reference,
            atom_indices=atom_indices,
        )
        self.base._validate_science_state("UKS Z-vector native gradient evaluation")
        dq_explicit_spin = self.base.dq_dR_explicit_spin(
            atom_indices=atom_indices
        )
        self.base._validate_science_state("UKS Z-vector explicit descriptor gradient evaluation")
        dq_explicit = dq_explicit_spin.sum(axis=0)
        correction_explicit_spin = np.einsum("sbxap,ap->sbx", dq_explicit_spin, sensitivity)
        metric_spin = np.asarray(adjoint.correction_gradient_metric_spin)
        nuclear_spin = np.asarray(adjoint.correction_gradient_adjoint_nuclear_spin)
        adjoint_metric_spin = np.asarray(adjoint.correction_gradient_adjoint_metric_spin)
        ov_spin = np.asarray(adjoint.correction_gradient_occupied_virtual_spin)
        response_spin = metric_spin + ov_spin
        correction_spin = correction_explicit_spin + response_spin
        correction_explicit = correction_explicit_spin.sum(axis=0)
        metric = np.asarray(adjoint.correction_gradient_metric)
        nuclear = np.asarray(adjoint.correction_gradient_adjoint_nuclear)
        adjoint_metric = np.asarray(adjoint.correction_gradient_adjoint_metric)
        ov = np.asarray(adjoint.correction_gradient_occupied_virtual)
        correction_response = np.asarray(adjoint.correction_gradient_response)
        correction = correction_explicit + correction_response
        total = native + correction
        fixed_spin = np.asarray(adjoint.correction_gradient_adjoint_fixed_grid_spin)
        coordinate_spin = np.asarray(adjoint.correction_gradient_adjoint_grid_coordinate_spin)
        weight_spin = np.asarray(adjoint.correction_gradient_adjoint_grid_weight_spin)
        shape = (len(adjoint.core.atom_indices), 3)
        if total.shape != shape or not np.isfinite(total).all():
            raise UKSAdjointError("the UKS DeePHF Z-vector gradient is invalid")
        self.base._validate_science_state("UKS Z-vector gradient assembly")
        return {
            "adjoint_result": adjoint,
            "descriptor_diagnostics": diagnostics,
            "reference_gradient": native,
            "dq_dR_explicit_spin": dq_explicit_spin,
            "dq_dR_explicit": dq_explicit,
            "correction_gradient_explicit_spin": correction_explicit_spin,
            "correction_gradient_metric_spin": metric_spin,
            "correction_gradient_adjoint_nuclear_spin": nuclear_spin,
            "correction_gradient_adjoint_fixed_grid_spin": fixed_spin,
            "correction_gradient_adjoint_grid_coordinate_spin": coordinate_spin,
            "correction_gradient_adjoint_grid_weight_spin": weight_spin,
            "correction_gradient_adjoint_metric_spin": adjoint_metric_spin,
            "correction_gradient_occupied_virtual_spin": ov_spin,
            "correction_gradient_response_spin": response_spin,
            "correction_gradient_spin": correction_spin,
            "correction_gradient_explicit": correction_explicit,
            "correction_gradient_metric": metric,
            "correction_gradient_adjoint_nuclear": nuclear,
            "correction_gradient_adjoint_fixed_grid": np.asarray(adjoint.correction_gradient_adjoint_fixed_grid),
            "correction_gradient_adjoint_grid_coordinate": np.asarray(
                adjoint.correction_gradient_adjoint_grid_coordinate
            ),
            "correction_gradient_adjoint_grid_weight": np.asarray(adjoint.correction_gradient_adjoint_grid_weight),
            "correction_gradient_adjoint_metric": adjoint_metric,
            "correction_gradient_occupied_virtual": ov,
            "correction_gradient_response": correction_response,
            "correction_gradient": correction,
            "de_full": total,
        }

    def _compact_kernel(self, atom_indices) -> dict:
        force_inputs = self.base._force_inputs()
        diagnostics, sensitivity = force_inputs
        self.base._validate_science_state("UKS Z-vector native gradient evaluation")
        reference = native_uks_gradient(self.base.reference, atom_indices)
        self.base._validate_science_state("UKS Z-vector native gradient evaluation")
        if not np.any(sensitivity):
            if (
                reference.dtype != np.dtype(np.float64)
                or np.iscomplexobj(reference)
                or not np.isfinite(reference).all()
            ):
                raise UKSAdjointError("the compact UKS native gradient is invalid")
            return {
                "descriptor_diagnostics": diagnostics,
                "response_diagnostics": None,
                "de": reference,
            }
        (
            _descriptor_diagnostics,
            explicit,
            adjoint_diagnostics,
            response_gradient,
        ) = self.base._zvector_inputs(
            self.adjoint_options,
            atom_indices=atom_indices,
            compact=True,
            force_inputs=force_inputs,
        )
        total = reference + explicit + response_gradient
        if total.shape != reference.shape or not np.isfinite(total).all():
            raise UKSAdjointError("the compact UKS Z-vector gradient is invalid")
        self.base._validate_science_state("UKS Z-vector gradient assembly")
        return {
            "descriptor_diagnostics": diagnostics,
            "response_diagnostics": adjoint_diagnostics,
            "de": total,
        }

    def as_scanner(self, **scanner_options):
        raise UKSAdjointError("UKS DeePHF does not provide a gradient scanner")


__all__ = ["UKSDeePHFZVectorGradients"]
