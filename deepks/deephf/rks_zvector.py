"""Strict scalar-adjoint nuclear gradients for finite-grid RKS DeePHF."""

import numpy as np

from .driver import GradientDriver
from .pyscf_dft_provenance import RKSAdjointError
from .pyscf_rks_native import native_rks_gradient


class RKSDeePHFZVectorGradients(GradientDriver):
    """Evaluate one pure-LDA RKS correction through a scalar adjoint."""

    _backend_name = "zvector"
    _binding_error_type = RKSAdjointError
    _binding_error_message = "the RKS DeePHF Z-vector driver binding is invalid"
    _construction_error_message = (
        "the RKS Z-vector gradient driver requires an exact RKSDeePHF method"
    )

    @classmethod
    def _expected_method_type(cls):
        from .rks_method import RKSDeePHF

        return RKSDeePHF

    def __init__(self, method, adjoint_options=None, retain_details=True):
        super().__init__(method, adjoint_options, retain_details)

    def _detail_kernel(self, atom_indices) -> dict:
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
        force_inputs = self.base._force_inputs()
        diagnostics, sensitivity = force_inputs
        self.base._validate_science_state("RKS Z-vector native gradient evaluation")
        reference = native_rks_gradient(self.base.reference, atom_indices)
        self.base._validate_science_state("RKS Z-vector native gradient evaluation")
        if not np.any(sensitivity):
            if (
                reference.dtype != np.dtype(np.float64)
                or np.iscomplexobj(reference)
                or not np.isfinite(reference).all()
            ):
                raise RKSAdjointError("the compact RKS native gradient is invalid")
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
            raise RKSAdjointError("the compact RKS Z-vector gradient is invalid")
        self.base._validate_science_state("RKS Z-vector gradient assembly")
        return {
            "descriptor_diagnostics": diagnostics,
            "response_diagnostics": adjoint_diagnostics,
            "de": total,
        }

    def as_scanner(self, **scanner_options):
        """Reject unavailable RKS scanner construction."""
        raise RKSAdjointError(
            "RKS DeePHF does not provide a gradient scanner"
        )


__all__ = ["RKSDeePHFZVectorGradients"]
