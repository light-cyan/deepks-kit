"""Strict coupled scalar-adjoint nuclear gradients for UHF DeePHF."""

import numpy as np

from .driver import GradientDriver
from .pyscf_uhf_reference import UHFAdjointError, _native_unrestricted_gradient


class UHFDeePHFZVectorGradients(GradientDriver):
    """Evaluate one unrestricted correction through a coupled scalar adjoint."""

    _backend_name = "zvector"
    _binding_error_type = UHFAdjointError
    _binding_error_message = "the UHF DeePHF Z-vector driver binding is invalid"
    _construction_error_message = (
        "the UHF Z-vector gradient driver requires an exact UHFDeePHF method"
    )

    @classmethod
    def _expected_method_type(cls):
        from .uhf_method import UHFDeePHF

        return UHFDeePHF

    def __init__(self, method, adjoint_options=None, retain_details=True):
        super().__init__(method, adjoint_options, retain_details)

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

    def _detail_kernel(self, atom_indices) -> dict:
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
        force_inputs = self.base._force_inputs()
        descriptor_diagnostics, sensitivity = force_inputs
        reference_gradient = self._validated_native_gradient(atom_indices)
        if not np.any(sensitivity):
            if (
                reference_gradient.dtype != np.dtype(np.float64)
                or np.iscomplexobj(reference_gradient)
                or not np.isfinite(reference_gradient).all()
            ):
                raise UHFAdjointError("the compact UHF native gradient is invalid")
            return {
                "descriptor_diagnostics": descriptor_diagnostics,
                "response_diagnostics": None,
                "de": reference_gradient,
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
        total = reference_gradient + explicit + response_gradient
        if total.shape != reference_gradient.shape or not np.isfinite(total).all():
            raise UHFAdjointError("the compact UHF Z-vector gradient is invalid")
        self.base._validate_science_state("UHF Z-vector gradient assembly")
        return {
            "descriptor_diagnostics": descriptor_diagnostics,
            "response_diagnostics": adjoint_diagnostics,
            "de": total,
        }

    def as_scanner(self, **scanner_options):
        """Reject unavailable unrestricted scanner construction."""
        raise UHFAdjointError(
            "UHF DeePHF does not provide a gradient scanner"
        )


__all__ = ["UHFDeePHFZVectorGradients"]
