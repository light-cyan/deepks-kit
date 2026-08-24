"""Direct coupled UHF response solution and audit."""

from dataclasses import fields, replace
import numpy as np
import pyscf
from pyscf.scf import ucphf
from .unrestricted_reference import (
    UHFResponse,
    UHFResponseDiagnostics,
    UHFResponseError,
    _immutable_array,
    _validated_float64_array,
    uhf_response_integrity_fingerprint,
)
from .pyscf_uhf_response_core import (
    _UHFLinearResponseCore,
)

class UHFResponseAdapter(_UHFLinearResponseCore):
    """Solve and independently audit molecular UHF nuclear UC-PHF response."""

    def _orbital_residual(
        self,
        responses,
        hamiltonian_mo,
        overlap_mo,
        coefficient,
        energy,
        occupied,
        virtual,
    ) -> tuple[np.ndarray, np.ndarray]:
        induced = self._induced_mo_potential(
            responses[0],
            responses[1],
            coefficient,
            occupied,
        )
        residuals = []
        for spin_index in range(2):
            spin_residual = (
                hamiltonian_mo[spin_index]
                + induced[spin_index]
                - overlap_mo[spin_index] * energy[spin_index, occupied[spin_index]]
                + (
                    energy[spin_index, :, None]
                    - energy[spin_index, occupied[spin_index]]
                )
                * responses[spin_index]
            )
            residuals.append(spin_residual[..., virtual[spin_index], :])
        return tuple(residuals)

    def _solve_orbitals(
        self,
        hamiltonian_mo,
        overlap_mo,
        coefficient,
        energy,
        occupation,
        occupied,
        virtual,
    ):
        perturbation_shape = hamiltonian_mo[0].shape[:-2]
        nset = int(np.prod(perturbation_shape))
        nmo = coefficient.shape[2]
        alpha_nocc, beta_nocc, alpha_nvir, beta_nvir, _, _ = self._dimensions(
            occupied,
            virtual,
        )
        flat_hamiltonian = (
            hamiltonian_mo[0].reshape(nset, nmo, alpha_nocc),
            hamiltonian_mo[1].reshape(nset, nmo, beta_nocc),
        )
        flat_overlap = (
            overlap_mo[0].reshape(nset, nmo, alpha_nocc),
            overlap_mo[1].reshape(nset, nmo, beta_nocc),
        )

        def induced_full(response):
            self._count_operation("response_operator_actions")
            response = np.asarray(response).reshape(
                -1,
                nmo * alpha_nocc + nmo * beta_nocc,
            )
            alpha = response[:, : nmo * alpha_nocc].reshape(-1, nmo, alpha_nocc)
            beta = response[:, nmo * alpha_nocc :].reshape(-1, nmo, beta_nocc)
            induced = self._induced_mo_potential(
                alpha,
                beta,
                coefficient,
                occupied,
            )
            return np.concatenate(
                (induced[0].reshape(len(response), -1), induced[1].reshape(len(response), -1)),
                axis=1,
            )

        try:
            response, _ = ucphf.solve(
                induced_full,
                energy,
                occupation,
                flat_hamiltonian,
                flat_overlap,
                max_cycle=self.max_cycle,
                tol=self.cphf_tolerance,
                level_shift=self.level_shift,
                verbose=self.reference.verbose,
            )
        except Exception as error:
            raise UHFResponseError(
                f"PySCF UHF coupled CPHF solve failed: {error}"
            ) from error
        alpha_response = _validated_float64_array(
            response[0],
            flat_hamiltonian[0].shape,
            "PySCF alpha UHF CPHF response",
        ).reshape(*perturbation_shape, nmo, alpha_nocc)
        beta_response = _validated_float64_array(
            response[1],
            flat_hamiltonian[1].shape,
            "PySCF beta UHF CPHF response",
        ).reshape(*perturbation_shape, nmo, beta_nocc)
        residual = self._orbital_residual(
            (alpha_response, beta_response),
            hamiltonian_mo,
            overlap_mo,
            coefficient,
            energy,
            occupied,
            virtual,
        )

        def maximum_residual(values):
            return max(float(np.max(np.abs(value), initial=0.0)) for value in values)

        residual_history = [maximum_residual(residual)]
        while (
            residual_history[-1] > self.residual_tolerance
            and len(residual_history) - 1 < self.max_refinement_cycles
        ):
            flat_residual = (
                residual[0].reshape(nset, alpha_nvir, alpha_nocc),
                residual[1].reshape(nset, beta_nvir, beta_nocc),
            )
            combined = np.concatenate(
                (
                    flat_residual[0].reshape(nset, -1),
                    flat_residual[1].reshape(nset, -1),
                ),
                axis=1,
            )
            scales = np.linalg.norm(combined, axis=1)
            active = scales > np.finfo(float).eps
            alpha_correction = np.zeros_like(flat_residual[0])
            beta_correction = np.zeros_like(flat_residual[1])

            def induced_virtual(response):
                self._count_operation("response_operator_actions")
                response = np.asarray(response).reshape(
                    -1,
                    alpha_nvir * alpha_nocc + beta_nvir * beta_nocc,
                )
                alpha_virtual = response[:, : alpha_nvir * alpha_nocc].reshape(
                    -1,
                    alpha_nvir,
                    alpha_nocc,
                )
                beta_virtual = response[:, alpha_nvir * alpha_nocc :].reshape(
                    -1,
                    beta_nvir,
                    beta_nocc,
                )
                alpha_full = np.zeros((len(response), nmo, alpha_nocc))
                beta_full = np.zeros((len(response), nmo, beta_nocc))
                alpha_full[:, virtual[0]] = alpha_virtual
                beta_full[:, virtual[1]] = beta_virtual
                induced = self._induced_mo_potential(
                    alpha_full,
                    beta_full,
                    coefficient,
                    occupied,
                )
                return np.concatenate(
                    (
                        induced[0][:, virtual[0]].reshape(len(response), -1),
                        induced[1][:, virtual[1]].reshape(len(response), -1),
                    ),
                    axis=1,
                )

            if np.any(active):
                normalized_rhs = (
                    flat_residual[0][active] / scales[active, None, None],
                    flat_residual[1][active] / scales[active, None, None],
                )
                try:
                    normalized_correction, _ = ucphf.solve(
                        induced_virtual,
                        energy,
                        occupation,
                        normalized_rhs,
                        s1=None,
                        max_cycle=self.max_cycle,
                        tol=self.cphf_tolerance,
                        level_shift=self.level_shift,
                        verbose=self.reference.verbose,
                    )
                except Exception as error:
                    raise UHFResponseError(
                        f"PySCF UHF coupled CPHF residual refinement failed: {error}"
                    ) from error
                alpha_normalized = _validated_float64_array(
                    normalized_correction[0],
                    normalized_rhs[0].shape,
                    "PySCF alpha UHF CPHF refinement response",
                )
                beta_normalized = _validated_float64_array(
                    normalized_correction[1],
                    normalized_rhs[1].shape,
                    "PySCF beta UHF CPHF refinement response",
                )
                alpha_correction[active] = alpha_normalized * scales[active, None, None]
                beta_correction[active] = beta_normalized * scales[active, None, None]
                alpha_response[..., virtual[0], :] += alpha_correction.reshape(
                    *perturbation_shape,
                    alpha_nvir,
                    alpha_nocc,
                )
                beta_response[..., virtual[1], :] += beta_correction.reshape(
                    *perturbation_shape,
                    beta_nvir,
                    beta_nocc,
                )
            residual = self._orbital_residual(
                (alpha_response, beta_response),
                hamiltonian_mo,
                overlap_mo,
                coefficient,
                energy,
                occupied,
                virtual,
            )
            residual_history.append(maximum_residual(residual))
        return (
            (alpha_response, beta_response),
            residual,
            tuple(residual_history),
        )

    @staticmethod
    def _invariants(
        density_response,
        density_ground,
        overlap,
        overlap_derivative,
    ):
        idempotency = (
            np.einsum("...ij,jk,kl->...il", density_response, overlap, density_ground)
            + np.einsum("ij,...jk,kl->...il", density_ground, overlap_derivative, density_ground)
            + np.einsum("ij,jk,...kl->...il", density_ground, overlap, density_response)
            - density_response
        )
        particle_number = (
            np.einsum("...ij,ji->...", density_response, overlap)
            + np.einsum("ij,...ji->...", density_ground, overlap_derivative)
        )
        return (
            float(np.max(np.abs(idempotency), initial=0.0)),
            float(np.max(np.abs(particle_number), initial=0.0)),
        )

    def _density_work(
        self,
        responses,
        coefficient,
        occupied,
        virtual,
        overlap_mo,
        overlap,
        overlap_derivative,
        atom_indices,
        result_mode,
    ):
        density_responses = []
        metric_densities = []
        occupied_virtual_densities = []
        metric_residuals = []
        for spin_index in range(2):
            response = responses[spin_index]
            if result_mode == "partitions":
                metric_response = np.zeros_like(response)
                metric_response[..., occupied[spin_index], :] = response[
                    ..., occupied[spin_index], :
                ]
                occupied_virtual_response = np.zeros_like(response)
                occupied_virtual_response[..., virtual[spin_index], :] = response[
                    ..., virtual[spin_index], :
                ]
                metric_density = self._density_from_mo_response(
                    metric_response, coefficient[spin_index], occupied[spin_index]
                )
                occupied_virtual_density = self._density_from_mo_response(
                    occupied_virtual_response,
                    coefficient[spin_index],
                    occupied[spin_index],
                )
                density_response = metric_density + occupied_virtual_density
                metric_densities.append(metric_density)
                occupied_virtual_densities.append(occupied_virtual_density)
            else:
                density_response = self._density_from_mo_response(
                    response, coefficient[spin_index], occupied[spin_index]
                )
            density_responses.append(density_response)
            overlap_occupied = overlap_mo[spin_index][
                ..., occupied[spin_index], :
            ]
            metric_residuals.append(
                float(
                    np.max(
                        np.abs(
                            response[..., occupied[spin_index], :]
                            + response[..., occupied[spin_index], :].swapaxes(-1, -2)
                            + overlap_occupied
                        ),
                        initial=0.0,
                    )
                )
            )
        total_density = density_responses[0] + density_responses[1]
        translations = (None, None, None)
        if len(atom_indices) == self.molecule.natm:
            translations = tuple(
                float(np.max(np.abs(np.sum(value, axis=0)), initial=0.0))
                for value in (*density_responses, total_density)
            )
        density_ground = np.asarray(self.reference.make_rdm1())
        invariants = tuple(
            self._invariants(
                density_responses[spin_index],
                density_ground[spin_index],
                overlap,
                overlap_derivative,
            )
            for spin_index in range(2)
        )
        return (
            tuple(density_responses),
            tuple(metric_densities),
            tuple(occupied_virtual_densities),
            tuple(metric_residuals),
            total_density,
            translations,
            invariants,
        )

    def _response_diagnostics(
        self,
        minimum_gaps,
        dimensions,
        residuals,
        residual_history,
        metric_residuals,
        invariants,
        translations,
    ):
        alpha_maximum = float(np.max(np.abs(residuals[0]), initial=0.0))
        beta_maximum = float(np.max(np.abs(residuals[1]), initial=0.0))
        residual_square_sum = sum(
            float(np.sum(np.square(value))) for value in residuals
        )
        residual_size = sum(value.size for value in residuals)
        alpha_dimension, beta_dimension = dimensions
        return UHFResponseDiagnostics(
            minimum_alpha_orbital_gap=minimum_gaps[0],
            minimum_beta_orbital_gap=minimum_gaps[1],
            pyscf_version=pyscf.__version__,
            cphf_tolerance=self.cphf_tolerance,
            maximum_residual=max(alpha_maximum, beta_maximum),
            alpha_maximum_residual=alpha_maximum,
            beta_maximum_residual=beta_maximum,
            residual_rms=float(np.sqrt(residual_square_sum / residual_size)),
            residual_tolerance=self.residual_tolerance,
            invariant_tolerance=self.invariant_tolerance,
            orbital_gap_tolerance=self.orbital_gap_tolerance,
            max_cycle=self.max_cycle,
            max_refinement_cycles=self.max_refinement_cycles,
            level_shift=self.level_shift,
            response_dimension=alpha_dimension + beta_dimension,
            alpha_response_dimension=alpha_dimension,
            beta_response_dimension=beta_dimension,
            operator_is_self_adjoint=True,
            alpha_metric_residual=metric_residuals[0],
            beta_metric_residual=metric_residuals[1],
            alpha_idempotency_residual=invariants[0][0],
            beta_idempotency_residual=invariants[1][0],
            alpha_particle_number_residual=invariants[0][1],
            beta_particle_number_residual=invariants[1][1],
            alpha_translation_residual=translations[0],
            beta_translation_residual=translations[1],
            translation_residual=translations[2],
            refinement_cycles=len(residual_history) - 1,
            residual_history=residual_history,
        )

    def solve(self, atom_indices=None) -> UHFResponse:
        """Return the audited spin response for selected atoms."""
        return self._solve(atom_indices, "response")

    def _solve_with_density_partitions(self, atom_indices=None):
        """Return a response and its transient spin-density work arrays."""
        return self._solve(atom_indices, "partitions")

    def _solve_for_gradient(self, objective, atom_indices=None):
        """Return compact diagnostics and the final density contraction."""
        return self._solve(atom_indices, "gradient", objective)

    def _solve(self, atom_indices, result_mode, objective=None):
        self._validate_reference(self.reference)
        atom_indices = self._response_atom_indices(atom_indices)
        coefficient, energy, occupation, occupied, virtual, minimum_gaps = self._state()
        *_, alpha_dimension, beta_dimension = self._dimensions(occupied, virtual)
        overlap = np.asarray(self.reference.get_ovlp())
        overlap_derivative = self._overlap_derivative(atom_indices)
        hamiltonian_derivative = self._hamiltonian_derivative(
            coefficient,
            occupation,
            atom_indices,
        )
        hamiltonian_mo = tuple(
            np.einsum(
                "mp,...mn,ni->...pi",
                coefficient[spin_index],
                hamiltonian_derivative[spin_index],
                coefficient[spin_index][:, occupied[spin_index]],
            )
            for spin_index in range(2)
        )
        overlap_mo = tuple(
            np.einsum(
                "mp,...mn,ni->...pi",
                coefficient[spin_index],
                overlap_derivative,
                coefficient[spin_index][:, occupied[spin_index]],
            )
            for spin_index in range(2)
        )
        responses, residuals, residual_history = self._solve_orbitals(
            hamiltonian_mo,
            overlap_mo,
            coefficient,
            energy,
            occupation,
            occupied,
            virtual,
        )
        (
            density_responses,
            metric_densities,
            occupied_virtual_densities,
            metric_residuals,
            total_density,
            translations,
            invariants,
        ) = self._density_work(
            responses,
            coefficient,
            occupied,
            virtual,
            overlap_mo,
            overlap,
            overlap_derivative,
            atom_indices,
            result_mode,
        )
        diagnostics = self._response_diagnostics(
            minimum_gaps,
            (alpha_dimension, beta_dimension),
            residuals,
            residual_history,
            metric_residuals,
            invariants,
            translations,
        )
        arrays = {
            "alpha MO response": responses[0],
            "beta MO response": responses[1],
            "alpha density response": density_responses[0],
            "beta density response": density_responses[1],
            "total density response": total_density,
            "overlap derivative": overlap_derivative,
            "alpha Hamiltonian derivative": hamiltonian_derivative[0],
            "beta Hamiltonian derivative": hamiltonian_derivative[1],
            "alpha residual": residuals[0],
            "beta residual": residuals[1],
        }
        nonfinite = [name for name, value in arrays.items() if not np.isfinite(value).all()]
        if nonfinite:
            raise UHFResponseError(
                "nonfinite UHF response quantities: " + ", ".join(nonfinite)
            )
        diagnostic_values = tuple(
            getattr(diagnostics, field.name)
            for field in fields(diagnostics)
            if field.name not in {"pyscf_version", "residual_history"}
            and getattr(diagnostics, field.name) is not None
        ) + diagnostics.residual_history
        if not np.isfinite(diagnostic_values).all():
            raise UHFResponseError("nonfinite UHF response diagnostics")
        if diagnostics.maximum_residual > self.residual_tolerance:
            history = " -> ".join(f"{value:.3e}" for value in residual_history)
            raise UHFResponseError(
                "UHF coupled response residual exceeds tolerance: "
                f"{diagnostics.maximum_residual:.3e} > {self.residual_tolerance:.3e}; "
                f"refinement history: {history}"
            )
        invariant_values = tuple(value for value in (
            *metric_residuals,
            invariants[0][0],
            invariants[1][0],
            invariants[0][1],
            invariants[1][1],
            *translations,
        ) if value is not None)
        if max(invariant_values) > self.invariant_tolerance:
            raise UHFResponseError(
                "UHF response invariant exceeds tolerance "
                f"{self.invariant_tolerance:.3e}: maximum={max(invariant_values):.3e}"
            )
        if result_mode == "gradient":
            return diagnostics, sum(
                np.einsum("ij,bxij->bx", objective, density)
                for density in density_responses
            )
        response = UHFResponse(
            reference_identity=id(self.reference),
            state_fingerprint=self._reference_fingerprint(self.reference),
            integrity_fingerprint="",
            atom_indices=atom_indices,
            alpha_mo_response=_immutable_array(responses[0]),
            beta_mo_response=_immutable_array(responses[1]),
            _mo_coefficients=_immutable_array(coefficient),
            _mo_occupations=_immutable_array(occupation),
            overlap_derivative=_immutable_array(overlap_derivative),
            alpha_hamiltonian_derivative=_immutable_array(hamiltonian_derivative[0]),
            beta_hamiltonian_derivative=_immutable_array(hamiltonian_derivative[1]),
            alpha_orbital_response_residual=_immutable_array(residuals[0]),
            beta_orbital_response_residual=_immutable_array(residuals[1]),
            diagnostics=diagnostics,
        )
        response = replace(
            response,
            integrity_fingerprint=uhf_response_integrity_fingerprint(response),
        )
        return (
            (
                response,
                (
                    tuple(density_responses),
                    tuple(metric_densities),
                    tuple(occupied_virtual_densities),
                ),
            )
            if result_mode == "partitions"
            else response
        )

    def _validate_supplied_structure(
        self,
        response: UHFResponse,
        occupied: np.ndarray,
        virtual: np.ndarray,
    ) -> None:
        from .audits.unrestricted_response import _validate_supplied_structure as audit
        return audit(self, response, occupied, virtual)

    def audit_response_equations(self, response: UHFResponse) -> None:
        from .audits.unrestricted_response import audit_response_equations as audit
        return audit(self, response)
