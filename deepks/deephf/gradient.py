"""Strict direct-oracle nuclear gradients for RHF DeePHF."""

from dataclasses import replace
import operator

import numpy as np

from .pyscf_rhf import (
    RHFBlockedResponseSummary,
    RHFResponseAdapter,
    RHFResponseError,
    blocked_response_summary_integrity_fingerprint,
    reference_fingerprint,
)


def _validate_atom_indices(mol, atmlst):
    """Return validated raw-atom indices before any response calculation."""
    if atmlst is None:
        return None
    try:
        requested_indices = tuple(atmlst)
    except TypeError as error:
        raise TypeError("gradient atmlst must be an iterable of integers") from error
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
        validated_indices.append(atom_index)
    return tuple(validated_indices)


class RHFDeePHFGradients:
    """Contract the complete relaxed descriptor response with one correction model."""

    def __init__(self, method, response_options=None):
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
        self.response_result = None
        self.descriptor_diagnostics = None
        self.reference_gradient = None
        self.dq_dR_explicit = None
        self.dq_dR_response = None
        self.dq_dR_relaxed = None
        self.correction_gradient_explicit = None
        self.correction_gradient_metric = None
        self.correction_gradient_occupied_virtual = None
        self.correction_gradient_response = None
        self.correction_gradient = None
        self.de_full = None
        self.de = None

    @property
    def response_diagnostics(self):
        if self.response_result is None:
            return None
        return self.response_result.diagnostics

    def _blocked_response(self, block_size, objective_ao_potential):
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
        dq_dP = self.base.dq_dP()
        expected_shape = (
            self.mol.natm,
            3,
            self.base.n_descriptor_atoms,
            self.base.n_descriptor_features,
        )
        dq_response = np.empty(expected_shape, dtype=np.float64)
        metric_gradient = np.empty((self.mol.natm, 3), dtype=np.float64)
        occupied_virtual_gradient = np.empty_like(metric_gradient)
        block_diagnostics = []
        for atom_indices, response in adapter.coordinate_blocks(block_size):
            target = list(atom_indices)
            dq_response[target] = np.einsum(
                "apij,bxij->bxap",
                dq_dP,
                response.density_response,
            )
            metric_gradient[target] = np.einsum(
                "ij,bxij->bx",
                objective_ao_potential,
                response.density_response_metric,
            )
            occupied_virtual_gradient[target] = np.einsum(
                "ij,bxij->bx",
                objective_ao_potential,
                response.density_response_occupied_virtual,
            )
            block_diagnostics.append(
                (len(atom_indices), response.diagnostics)
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
            dq_response,
            metric_gradient,
            occupied_virtual_gradient,
        )

    def kernel(self, atmlst=None) -> np.ndarray:
        """Evaluate d(E_base + E_corr)/dR for all or selected atoms."""
        self._reset_results()
        self._validate_driver_binding()
        atom_indices = _validate_atom_indices(self.mol, atmlst)
        self.descriptor_diagnostics = self.base.validate_force_compatibility()
        self.reference_gradient = np.asarray(
            self.base.reference.nuc_grad_method().kernel()
        )
        self.dq_dR_explicit = self.base.dq_dR_explicit()
        sensitivity = self.base.correction_sensitivity()
        objective_ao_potential = self.base._correction_ao_potential(
            sensitivity
        )
        coordinate_block_size = self.response_options.get(
            "coordinate_block_size",
            self.base.response_options.get("coordinate_block_size"),
        )
        if coordinate_block_size is None:
            response_options = dict(self.response_options)
            response_options.pop("coordinate_block_size", None)
            self.response_result = self.base.response(**response_options)
            self.dq_dR_response = self.base.dq_dR_response(
                response=self.response_result
            )
            self.correction_gradient_metric = np.einsum(
                "ij,bxij->bx",
                objective_ao_potential,
                self.response_result.density_response_metric,
            )
            self.correction_gradient_occupied_virtual = np.einsum(
                "ij,bxij->bx",
                objective_ao_potential,
                self.response_result.density_response_occupied_virtual,
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
        if not np.allclose(
            self.correction_gradient_response,
            self.correction_gradient_metric
            + self.correction_gradient_occupied_virtual,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise RHFResponseError(
                "the RHF direct response-gradient partitions are inconsistent"
            )
        self.correction_gradient = (
            self.correction_gradient_explicit
            + self.correction_gradient_response
        )
        self.de_full = self.reference_gradient + self.correction_gradient
        if not np.isfinite(self.de_full).all():
            raise RHFResponseError("the RHF DeePHF analytic gradient is nonfinite")
        if atom_indices is None:
            self.de = self.de_full
        else:
            self.de = self.de_full[list(atom_indices)]
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
