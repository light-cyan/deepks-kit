"""Strict direct-oracle nuclear gradients for RHF DeePHF."""

from dataclasses import replace
import operator

import numpy as np

from .driver import GradientDriver
from .pyscf_rhf_reference import (
    RHFBlockedResponseSummary,
    RHFResponseError,
    blocked_response_summary_integrity_fingerprint,
    reference_fingerprint,
)
from .pyscf_rhf_response import RHFResponseAdapter


class RHFDeePHFGradients(GradientDriver):
    """Contract the complete relaxed descriptor response with one correction model."""

    _backend_name = "direct"
    _binding_error_type = RHFResponseError
    _binding_error_message = "the direct gradient driver binding is invalid"
    _construction_error_message = (
        "the direct gradient driver requires an exact DeePHF method"
    )

    @classmethod
    def _expected_method_type(cls):
        from .method import DeePHF

        return DeePHF

    def __init__(self, method, response_options=None, retain_details=True):
        super().__init__(method, response_options, retain_details)

    def _blocked_response(
        self,
        block_size,
        objective_ao_potential,
        atom_indices,
        dq_dP=None,
    ):
        if isinstance(block_size, (bool, np.bool_)):
            raise TypeError("coordinate_block_size must be an integer")
        try:
            block_size = operator.index(block_size)
        except TypeError as error:
            raise TypeError("coordinate_block_size must be an integer") from error
        if block_size <= 0:
            raise ValueError("coordinate_block_size must be positive")
        options = {
            **self.base.response_options,
            **self.response_options,
        }
        options.pop("coordinate_block_size", None)
        adapter = RHFResponseAdapter(self.base.reference, **options)
        adapter._operation_hook = self.base._context().count
        self.base._context().count("direct_response_solves")
        if dq_dP is not None:
            self.base._context().count("full_density_partition_materializations")
        compact = dq_dP is None
        response_work = np.empty(
            (len(atom_indices), 3)
            if compact
            else (
                len(atom_indices),
                3,
                self.base.n_descriptor_atoms,
                self.base.n_descriptor_features,
            ),
            dtype=np.float64,
        )
        metric_gradient = (
            None
            if compact
            else np.empty((len(atom_indices), 3), dtype=np.float64)
        )
        occupied_virtual_gradient = (
            None if compact else np.empty_like(metric_gradient)
        )
        block_diagnostics = []
        result_positions = {
            atom_index: result_index
            for result_index, atom_index in enumerate(atom_indices)
        }
        for block_atoms, block_result in adapter.coordinate_blocks(
            block_size,
            atom_indices=atom_indices,
            result_mode="gradient" if compact else "partitions",
            objective=objective_ao_potential,
        ):
            target = [result_positions[atom_index] for atom_index in block_atoms]
            if compact:
                diagnostics, response_work[target] = block_result
            else:
                block_response, density_partitions = block_result
                diagnostics = block_response.diagnostics
                density, density_metric, density_occupied_virtual = density_partitions
            if not compact:
                response_work[target] = np.einsum("apij,bxij->bxap", dq_dP, density)
            if not compact:
                metric_gradient[target] = np.einsum(
                    "ij,bxij->bx",
                    objective_ao_potential,
                    density_metric,
                )
                occupied_virtual_gradient[target] = np.einsum(
                    "ij,bxij->bx",
                    objective_ao_potential,
                    density_occupied_virtual,
                )
            block_diagnostics.append(
                (len(block_atoms), diagnostics)
            )
        worst = max(
            (diagnostics for _, diagnostics in block_diagnostics),
            key=lambda diagnostics: diagnostics.maximum_residual,
        )
        total_atoms = sum(atom_count for atom_count, _ in block_diagnostics)
        diagnostics = replace(
            worst,
            residual_rms=float(
                np.sqrt(
                    sum(
                        atom_count * item.residual_rms**2
                        for atom_count, item in block_diagnostics
                    )
                    / total_atoms
                )
            ),
            metric_residual=max(
                item.metric_residual for _, item in block_diagnostics
            ),
            idempotency_residual=max(
                item.idempotency_residual for _, item in block_diagnostics
            ),
            particle_number_residual=max(
                item.particle_number_residual for _, item in block_diagnostics
            ),
        )
        summary = RHFBlockedResponseSummary(
            reference_identity=id(self.base.reference),
            state_fingerprint=reference_fingerprint(self.base.reference),
            integrity_fingerprint="",
            coordinate_block_size=block_size,
            block_count=len(block_diagnostics),
            diagnostics=diagnostics,
        )
        summary = replace(
            summary,
            integrity_fingerprint=(
                blocked_response_summary_integrity_fingerprint(summary)
            ),
        )
        return (
            summary,
            response_work,
            metric_gradient,
            occupied_virtual_gradient,
        )

    def _compact_kernel(self, atom_indices):
        descriptor_diagnostics, sensitivity = self.base._force_inputs()
        reference_gradient = np.asarray(
            self.base.reference.nuc_grad_method().kernel(
                atmlst=list(atom_indices)
            )
        )
        if not np.any(sensitivity):
            if (
                reference_gradient.shape != (len(atom_indices), 3)
                or reference_gradient.dtype != np.dtype(np.float64)
                or np.iscomplexobj(reference_gradient)
                or not np.isfinite(reference_gradient).all()
            ):
                raise RHFResponseError("the compact RHF native gradient is invalid")
            return {
                "descriptor_diagnostics": descriptor_diagnostics,
                "response_diagnostics": None,
                "de": reference_gradient,
            }
        explicit, objective = self.base._correction_derivatives(
            sensitivity,
            atom_indices,
        )
        coordinate_block_size = self.response_options.get(
            "coordinate_block_size",
            self.base.response_options.get("coordinate_block_size"),
        )
        if coordinate_block_size is None:
            response_options = {
                **self.base.response_options,
                **self.response_options,
            }
            response_options.pop("coordinate_block_size", None)
            self.base._context().count("direct_response_solves")
            adapter = RHFResponseAdapter(
                self.base.reference,
                **response_options,
            )
            adapter._operation_hook = self.base._context().count
            response_diagnostics, response = adapter._solve_for_gradient(
                objective,
                atom_indices=atom_indices,
            )
        else:
            summary, response, _metric, _occupied_virtual = (
                self._blocked_response(
                    coordinate_block_size,
                    objective,
                    atom_indices,
                )
            )
            response_diagnostics = summary.diagnostics
        total = reference_gradient + explicit + response
        if total.shape != (len(atom_indices), 3) or not np.isfinite(total).all():
            raise RHFResponseError("the compact RHF gradient is invalid")
        return {
            "descriptor_diagnostics": descriptor_diagnostics,
            "response_diagnostics": response_diagnostics,
            "de": total,
        }

    def _detail_kernel(self, atom_indices) -> dict:
        descriptor_diagnostics, sensitivity = self.base._force_inputs()
        reference_gradient = np.asarray(
            self.base.reference.nuc_grad_method().kernel(
                atmlst=list(atom_indices)
            )
        )
        dq_dR_explicit = self.base.dq_dR_explicit(
            atom_indices=atom_indices,
        )
        dq_dP = self.base._dq_dP()
        objective_ao_potential = self.base._correction_ao_potential(
            sensitivity,
            dq_dP,
        )
        coordinate_block_size = self.response_options.get(
            "coordinate_block_size",
            self.base.response_options.get("coordinate_block_size"),
        )
        if coordinate_block_size is None:
            response_options = {
                **self.base.response_options,
                **self.response_options,
            }
            response_options.pop("coordinate_block_size", None)
            self.base._context().count("direct_response_solves")
            self.base._context().count("full_density_partition_materializations")
            adapter = RHFResponseAdapter(
                self.base.reference,
                **response_options,
            )
            adapter._operation_hook = self.base._context().count
            response_result, density_partitions = adapter._solve_with_density_partitions(
                atom_indices=atom_indices
            )
            density, density_metric, density_occupied_virtual = density_partitions
            dq_dR_response = np.einsum(
                "apij,bxij->bxap",
                dq_dP,
                density,
            )
            correction_gradient_metric = np.einsum(
                "ij,bxij->bx",
                objective_ao_potential,
                density_metric,
            )
            correction_gradient_occupied_virtual = np.einsum(
                "ij,bxij->bx",
                objective_ao_potential,
                density_occupied_virtual,
            )
        else:
            (
                response_result,
                dq_dR_response,
                correction_gradient_metric,
                correction_gradient_occupied_virtual,
            ) = self._blocked_response(
                coordinate_block_size,
                objective_ao_potential,
                atom_indices,
                dq_dP,
            )
        dq_dR_relaxed = dq_dR_explicit + dq_dR_response
        correction_gradient_explicit = np.einsum(
            "bxap,ap->bx",
            dq_dR_explicit,
            sensitivity,
        )
        correction_gradient_response = np.einsum(
            "bxap,ap->bx",
            dq_dR_response,
            sensitivity,
        )
        correction_gradient = (
            correction_gradient_explicit
            + correction_gradient_response
        )
        de_full = reference_gradient + correction_gradient
        if not np.isfinite(de_full).all():
            raise RHFResponseError("the RHF DeePHF analytic gradient is nonfinite")
        return {
            "response_result": response_result,
            "descriptor_diagnostics": descriptor_diagnostics,
            "reference_gradient": reference_gradient,
            "dq_dR_explicit": dq_dR_explicit,
            "dq_dR_response": dq_dR_response,
            "dq_dR_relaxed": dq_dR_relaxed,
            "correction_gradient_explicit": correction_gradient_explicit,
            "correction_gradient_metric": correction_gradient_metric,
            "correction_gradient_occupied_virtual": correction_gradient_occupied_virtual,
            "correction_gradient_response": correction_gradient_response,
            "correction_gradient": correction_gradient,
            "de_full": de_full,
        }

"""Strict direct-oracle nuclear gradients for finite-grid RKS DeePHF."""

import numpy as np

from .driver import GradientDriver
from .pyscf_dft_provenance import RKSResponseError
from .pyscf_rks_reference import native_rks_gradient


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

"""Strict direct-oracle nuclear gradients for UHF DeePHF."""

import numpy as np

from .driver import GradientDriver
from .unrestricted_reference import UHFResponseError, _native_unrestricted_gradient


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
        from .unrestricted_method import UHFDeePHF

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

"""Strict direct-oracle nuclear gradients for finite-grid UKS DeePHF."""

import numpy as np

from .unrestricted_reference import UKSResponseError
from .pyscf_uks_response import native_uks_gradient


class UKSDeePHFGradients(UHFDeePHFGradients):
    """Contract the complete coupled UKS response with one correction model."""

    _binding_error_type = UKSResponseError
    _binding_error_message = "the UKS direct gradient driver binding is invalid"
    _construction_error_message = (
        "the UKS direct gradient driver requires an exact UKSDeePHF method"
    )

    @classmethod
    def _expected_method_type(cls):
        from .unrestricted_method import UKSDeePHF

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

__all__ = [
    "RHFDeePHFGradients",
    "RKSDeePHFGradients",
    "UHFDeePHFGradients",
    "UKSDeePHFGradients",
]
