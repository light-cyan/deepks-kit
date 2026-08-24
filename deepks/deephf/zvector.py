"""Strict scalar-adjoint nuclear gradients for RHF DeePHF."""

import numpy as np

from .driver import GradientDriver
from .pyscf_rhf_reference import RHFAdjointError


class RHFDeePHFZVectorGradients(GradientDriver):
    """Evaluate the correction response through one RHF scalar adjoint."""

    _backend_name = "zvector"
    _binding_error_type = RHFAdjointError
    _binding_error_message = "the RHF DeePHF Z-vector driver binding is invalid"
    _construction_error_message = (
        "the Z-vector driver requires an exact DeePHF method"
    )

    @classmethod
    def _expected_method_type(cls):
        from .method import DeePHF

        return DeePHF

    def __init__(self, method, adjoint_options=None, retain_details=True):
        super().__init__(method, adjoint_options, retain_details)

    def _compact_kernel(self, atom_indices):
        force_inputs = self.base._force_inputs()
        descriptor_diagnostics, sensitivity = force_inputs
        self.base._assert_science_state("native RHF gradient evaluation")
        reference_gradient = np.asarray(
            self.base.reference.nuc_grad_method().kernel(
                atmlst=None if atom_indices is None else list(atom_indices)
            )
        )
        self.base._validate_science_state("native RHF gradient evaluation")
        if not np.any(sensitivity):
            if (
                reference_gradient.dtype != np.dtype(np.float64)
                or np.iscomplexobj(reference_gradient)
                or not np.isfinite(reference_gradient).all()
            ):
                raise RHFAdjointError("the compact RHF native gradient is invalid")
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
            raise RHFAdjointError("the compact RHF Z-vector gradient is invalid")
        return {
            "descriptor_diagnostics": descriptor_diagnostics,
            "response_diagnostics": adjoint_diagnostics,
            "de": total,
        }

    def _detail_kernel(self, atom_indices) -> dict:
        """Evaluate d(E_base + E_corr)/dR without a coordinate-wise density response."""
        descriptor_diagnostics, sensitivity, adjoint = self.base._zvector_inputs(
            self.adjoint_options,
            atom_indices=atom_indices,
        )
        self.base._assert_science_state("native RHF gradient evaluation")
        reference_gradient = np.asarray(
            self.base.reference.nuc_grad_method().kernel(
                atmlst=None if atom_indices is None else list(atom_indices)
            )
        )
        self.base._validate_science_state("native RHF gradient evaluation")
        self.base._assert_science_state("explicit descriptor gradient evaluation")
        dq_dR_explicit = self.base.dq_dR_explicit(atom_indices=atom_indices)
        self.base._assert_science_state("explicit descriptor gradient evaluation")
        correction_gradient_explicit = np.einsum(
            "bxap,ap->bx",
            dq_dR_explicit,
            sensitivity,
        )
        correction_gradient_metric = np.asarray(
            adjoint.correction_gradient_metric
        )
        correction_gradient_adjoint_nuclear = np.asarray(
            adjoint.correction_gradient_adjoint_nuclear
        )
        correction_gradient_adjoint_metric = np.asarray(
            adjoint.correction_gradient_adjoint_metric
        )
        correction_gradient_occupied_virtual = np.asarray(
            adjoint.correction_gradient_occupied_virtual
        )
        correction_gradient_response = np.asarray(
            adjoint.correction_gradient_response
        )
        correction_gradient = (
            correction_gradient_explicit + correction_gradient_response
        )
        de_full = reference_gradient + correction_gradient
        expected_shape = (len(adjoint.atom_indices), 3)
        result_fields = {
            "reference gradient": reference_gradient,
            "explicit correction gradient": correction_gradient_explicit,
            "metric correction gradient": correction_gradient_metric,
            "adjoint nuclear correction gradient": (
                correction_gradient_adjoint_nuclear
            ),
            "adjoint metric correction gradient": (
                correction_gradient_adjoint_metric
            ),
            "occupied-virtual correction gradient": (
                correction_gradient_occupied_virtual
            ),
            "response correction gradient": correction_gradient_response,
            "complete correction gradient": correction_gradient,
            "RHF DeePHF Z-vector gradient": de_full,
        }
        for name, value in result_fields.items():
            if value.shape != expected_shape:
                raise RHFAdjointError(
                    f"the {name} has shape {value.shape}; expected {expected_shape}"
                )
            if value.dtype != np.dtype(np.float64) or np.iscomplexobj(value):
                raise RHFAdjointError(
                    f"the {name} must be a real numpy.float64 array"
                )
            if not np.isfinite(value).all():
                raise RHFAdjointError(f"the {name} must be finite")
        if not np.allclose(
            correction_gradient_occupied_virtual,
            correction_gradient_adjoint_nuclear
            + correction_gradient_adjoint_metric,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise RHFAdjointError(
                "the RHF occupied-virtual adjoint partitions are inconsistent"
            )
        if not np.allclose(
            correction_gradient_response,
            correction_gradient_metric
            + correction_gradient_occupied_virtual,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise RHFAdjointError(
                "the RHF scalar-adjoint response partitions are inconsistent"
            )
        self.base._assert_science_state("Z-vector gradient assembly")
        return {
            "adjoint_result": adjoint,
            "descriptor_diagnostics": descriptor_diagnostics,
            "reference_gradient": reference_gradient,
            "dq_dR_explicit": dq_dR_explicit,
            "correction_gradient_explicit": correction_gradient_explicit,
            "correction_gradient_metric": correction_gradient_metric,
            "correction_gradient_adjoint_nuclear": correction_gradient_adjoint_nuclear,
            "correction_gradient_adjoint_metric": correction_gradient_adjoint_metric,
            "correction_gradient_occupied_virtual": correction_gradient_occupied_virtual,
            "correction_gradient_response": correction_gradient_response,
            "correction_gradient": correction_gradient,
            "de_full": de_full,
        }

    def as_scanner(self, **scanner_options):
        """Build a strict fresh-reference RHF DeePHF gradient scanner."""
        from .scanner import RHFDeePHFGradientScanner

        return RHFDeePHFGradientScanner(self, **scanner_options)
