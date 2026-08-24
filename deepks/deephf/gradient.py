"""Strict direct-oracle nuclear gradients for RHF DeePHF."""

from dataclasses import replace
import operator

import numpy as np

from .capabilities import science_state_transaction
from .pyscf_rhf import (
    RHFBlockedResponseSummary,
    RHFResponseAdapter,
    RHFResponseError,
    blocked_response_summary_integrity_fingerprint,
    reference_fingerprint,
)


def _validate_atom_indices(mol, atmlst):
    """Return unique raw-atom indices before any coordinate calculation."""
    if atmlst is None:
        return None
    try:
        requested_indices = tuple(atmlst)
    except TypeError as error:
        raise TypeError("gradient atmlst must be an iterable of integers") from error
    if not requested_indices:
        raise ValueError("gradient atmlst must not be empty")
    validated_indices = []
    for index in requested_indices:
        if isinstance(index, (bool, np.bool_)):
            raise TypeError("gradient atom indices must be integers")
        try:
            atom_index = operator.index(index)
        except TypeError as error:
            raise TypeError("gradient atom indices must be integers") from error
        if atom_index < 0 or atom_index >= mol.natm:
            raise IndexError("gradient atom index is outside the molecule")
        if atom_index in validated_indices:
            raise ValueError("gradient atmlst must not contain duplicates")
        validated_indices.append(atom_index)
    return tuple(validated_indices)


def _validate_retain_details(value) -> bool:
    if type(value) is not bool:
        raise TypeError("retain_details must be a Boolean")
    return value


_DRIVER_CONFIGURATION = frozenset(
    {
        "_base",
        "_bound_base",
        "_mol",
        "_bound_mol",
        "_backend",
        "response_options",
        "_adjoint_options",
        "_bound_adjoint_options",
        "retain_details",
    }
)


def _reset_driver_results(driver) -> None:
    for name in tuple(vars(driver)):
        if name not in _DRIVER_CONFIGURATION:
            delattr(driver, name)
    driver._response_diagnostics = None
    driver.descriptor_diagnostics = None
    driver.de = None


class RHFDeePHFGradients:
    """Contract the complete relaxed descriptor response with one correction model."""

    def __init__(self, method, response_options=None, retain_details=True):
        from .method import DeePHF

        if type(method) is not DeePHF:
            raise TypeError(
                "the direct gradient driver requires an exact DeePHF method"
            )
        self._base = method
        self._bound_base = method
        self._mol = method.mol
        self._bound_mol = method.mol
        self._backend = "direct"
        self.retain_details = _validate_retain_details(retain_details)
        self.response_options = dict(response_options or {})
        self._reset_results()

    @property
    def base(self):
        return self._base

    @property
    def mol(self):
        return self._mol

    @property
    def backend(self) -> str:
        return self._backend

    def _validate_driver_binding(self) -> None:
        from .method import DeePHF

        if (
            type(self._base) is not DeePHF
            or self._base is not self._bound_base
            or self._mol is not self._bound_mol
            or self._mol is not self._base.mol
            or self._backend != "direct"
        ):
            raise RHFResponseError(
                "the direct gradient driver binding is invalid"
            )

    def _reset_results(self):
        _reset_driver_results(self)

    @property
    def response_diagnostics(self):
        return (
            self._response_diagnostics
            if getattr(self, "response_result", None) is None
            else self.response_result.diagnostics
        )

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
            response_diagnostics, response = RHFResponseAdapter(
                self.base.reference,
                **response_options,
            )._solve_for_gradient(
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
        return descriptor_diagnostics, response_diagnostics, total

    @science_state_transaction
    def kernel(self, atmlst=None) -> np.ndarray:
        """Evaluate d(E_base + E_corr)/dR for all or selected atoms."""
        self._reset_results()
        self._validate_driver_binding()
        atom_indices = _validate_atom_indices(self.mol, atmlst)
        calculation_atom_indices = (
            tuple(range(self.mol.natm))
            if atom_indices is None
            else atom_indices
        )
        if not self.retain_details:
            (
                self.descriptor_diagnostics,
                self._response_diagnostics,
                self.de,
            ) = self._compact_kernel(calculation_atom_indices)
            return self.de
        self.descriptor_diagnostics, sensitivity = self.base._force_inputs()
        self.reference_gradient = np.asarray(
            self.base.reference.nuc_grad_method().kernel(
                atmlst=(
                    None
                    if atom_indices is None
                    else list(calculation_atom_indices)
                )
            )
        )
        self.dq_dR_explicit = self.base.dq_dR_explicit(
            atom_indices=calculation_atom_indices,
        )
        dq_dP = self.base.dq_dP()
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
            self.response_result, density_partitions = RHFResponseAdapter(
                self.base.reference,
                **response_options,
            )._solve_with_density_partitions(
                atom_indices=calculation_atom_indices
            )
            density, density_metric, density_occupied_virtual = density_partitions
            self.dq_dR_response = np.einsum(
                "apij,bxij->bxap",
                dq_dP,
                density,
            )
            self.correction_gradient_metric = np.einsum(
                "ij,bxij->bx",
                objective_ao_potential,
                density_metric,
            )
            self.correction_gradient_occupied_virtual = np.einsum(
                "ij,bxij->bx",
                objective_ao_potential,
                density_occupied_virtual,
            )
        else:
            (
                self.response_result,
                self.dq_dR_response,
                self.correction_gradient_metric,
                self.correction_gradient_occupied_virtual,
            ) = self._blocked_response(
                coordinate_block_size,
                objective_ao_potential,
                calculation_atom_indices,
                dq_dP,
            )
        self.dq_dR_relaxed = self.dq_dR_explicit + self.dq_dR_response
        self.correction_gradient_explicit = np.einsum(
            "bxap,ap->bx",
            self.dq_dR_explicit,
            sensitivity,
        )
        self.correction_gradient_response = np.einsum(
            "bxap,ap->bx",
            self.dq_dR_response,
            sensitivity,
        )
        self.correction_gradient = (
            self.correction_gradient_explicit
            + self.correction_gradient_response
        )
        self.de_full = self.reference_gradient + self.correction_gradient
        if not np.isfinite(self.de_full).all():
            raise RHFResponseError("the RHF DeePHF analytic gradient is nonfinite")
        self.de = self.de_full
        return self.de

    def run(self, atmlst=None):
        """Evaluate the gradient and return this result object."""
        self.kernel(atmlst=atmlst)
        return self

    def forces(self, atmlst=None) -> np.ndarray:
        """Evaluate nuclear forces as minus the energy gradient."""
        return -self.kernel(atmlst=atmlst)

    def as_scanner(self, **scanner_options):
        """Build a strict fresh-reference RHF DeePHF gradient scanner."""
        from .scanner import RHFDeePHFGradientScanner

        return RHFDeePHFGradientScanner(self, **scanner_options)
