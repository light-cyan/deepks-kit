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

"""Strict scalar-adjoint nuclear gradients for finite-grid RKS DeePHF."""

import numpy as np

from .driver import GradientDriver
from .pyscf_dft_provenance import RKSAdjointError
from .pyscf_rks_reference import native_rks_gradient


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

"""Strict coupled scalar-adjoint nuclear gradients for UHF DeePHF."""

import numpy as np

from .driver import GradientDriver
from .unrestricted_reference import UHFAdjointError, _native_unrestricted_gradient


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
        from .unrestricted_method import UHFDeePHF

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

"""Strict coupled scalar-adjoint nuclear gradients for UKS DeePHF."""

import numpy as np

from .unrestricted_reference import UKSAdjointError
from .pyscf_uks_response import native_uks_gradient


class UKSDeePHFZVectorGradients(UHFDeePHFZVectorGradients):
    """Evaluate one finite-grid UKS correction through one coupled adjoint."""

    _binding_error_type = UKSAdjointError
    _binding_error_message = "the UKS DeePHF Z-vector driver binding is invalid"
    _construction_error_message = (
        "the UKS Z-vector gradient driver requires an exact UKSDeePHF method"
    )

    @classmethod
    def _expected_method_type(cls):
        from .unrestricted_method import UKSDeePHF

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

__all__ = [
    "RHFDeePHFZVectorGradients",
    "RKSDeePHFZVectorGradients",
    "UHFDeePHFZVectorGradients",
    "UKSDeePHFZVectorGradients",
]
