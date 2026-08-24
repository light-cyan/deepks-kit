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

    def as_scanner(self, **scanner_options):
        """Build a strict fresh-reference RHF DeePHF gradient scanner."""
        from .scanner import RHFDeePHFGradientScanner

        return RHFDeePHFGradientScanner(self, **scanner_options)
