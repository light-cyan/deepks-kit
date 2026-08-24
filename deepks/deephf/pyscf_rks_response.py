"""Internal implementation extracted from pyscf_rks.py."""

from dataclasses import replace
import numpy as np
import pyscf
from pyscf.dft import libxc
from pyscf.scf import cphf
from .contracts import immutable_array as _immutable_array
from .pyscf_dft_provenance import (
    RKSResponse,
    RKSResponseDiagnostics,
    RKSResponseError,
    _functional_provenance,
    _grid_provenance,
    _validated_float64_array,
)
from .pyscf_rks_reference import (
    rks_reference_fingerprint,
    rks_response_integrity_fingerprint,
    validate_rks_reference,
)
from .pyscf_rks_response_core import _RKSLinearResponseCore

class RKSResponseAdapter(_RKSLinearResponseCore):
    """Solve and independently audit molecular pure-LDA RKS nuclear CPKS."""

    def _orbital_residual(
        self,
        mo_response: np.ndarray,
        hamiltonian_mo: np.ndarray,
        overlap_mo: np.ndarray,
        coefficient: np.ndarray,
        energy: np.ndarray,
        occupation: np.ndarray,
        occupied: np.ndarray,
        virtual: np.ndarray,
    ) -> np.ndarray:
        induced = self._induced_mo_potential(
            mo_response,
            coefficient,
            occupation,
            occupied,
        )
        residual = (
            hamiltonian_mo
            + induced
            - overlap_mo * energy[occupied]
            + (energy[:, None] - energy[occupied]) * mo_response
        )
        return residual[..., virtual, :]

    def _solve_orbitals(
        self,
        hamiltonian_mo: np.ndarray,
        overlap_mo: np.ndarray,
        coefficient: np.ndarray,
        energy: np.ndarray,
        occupation: np.ndarray,
        occupied: np.ndarray,
        virtual: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, tuple[float, ...]]:
        perturbation_shape = hamiltonian_mo.shape[:-2]
        nmo = coefficient.shape[1]
        nocc = int(np.count_nonzero(occupied))
        nvir = int(np.count_nonzero(virtual))
        flattened_hamiltonian = hamiltonian_mo.reshape(-1, nmo, nocc)
        flattened_overlap = overlap_mo.reshape(-1, nmo, nocc)

        def solver_induced_full(response):
            self._count_operation("response_operator_actions")
            response = np.asarray(response).reshape(-1, nmo, nocc)
            return self._pyscf_induced_mo_potential(
                response,
                coefficient,
                occupation,
                occupied,
            )

        def physical_induced_full(response):
            response = np.asarray(response).reshape(-1, nmo, nocc)
            return self._induced_mo_potential(
                response,
                coefficient,
                occupation,
                occupied,
            )

        try:
            response, _ = cphf.solve(
                solver_induced_full,
                energy,
                occupation,
                flattened_hamiltonian,
                flattened_overlap,
                max_cycle=self.max_cycle,
                tol=self.cphf_tolerance,
                level_shift=self.level_shift,
                verbose=self.reference.verbose,
            )
        except Exception as error:
            raise RKSResponseError(f"PySCF RKS CPKS solve failed: {error}") from error
        response = _validated_float64_array(
            response,
            flattened_hamiltonian.shape,
            "PySCF RKS CPKS response",
        ).reshape(*perturbation_shape, nmo, nocc)
        residual = self._orbital_residual(
            response,
            hamiltonian_mo,
            overlap_mo,
            coefficient,
            energy,
            occupation,
            occupied,
            virtual,
        )
        residual_history = [float(np.max(np.abs(residual), initial=0.0))]
        while (
            residual_history[-1] > self.residual_tolerance
            and len(residual_history) - 1 < self.max_refinement_cycles
        ):
            flat_residual = residual.reshape(-1, nvir, nocc)
            root_scales = np.linalg.norm(
                flat_residual.reshape(len(flat_residual), -1),
                axis=1,
            )
            active = root_scales > np.finfo(float).eps
            correction = np.zeros_like(flat_residual)

            def induced_virtual(virtual_response):
                self._count_operation("response_operator_actions")
                virtual_response = np.asarray(virtual_response).reshape(-1, nvir, nocc)
                full_response = np.zeros(
                    (len(virtual_response), nmo, nocc),
                    dtype=np.float64,
                )
                full_response[:, virtual] = virtual_response
                return physical_induced_full(full_response)[:, virtual]

            if np.any(active):
                try:
                    normalized_correction, _ = cphf.solve(
                        induced_virtual,
                        energy,
                        occupation,
                        flat_residual[active] / root_scales[active, None, None],
                        s1=None,
                        max_cycle=self.max_cycle,
                        tol=self.cphf_tolerance,
                        level_shift=self.level_shift,
                        verbose=self.reference.verbose,
                    )
                except Exception as error:
                    raise RKSResponseError(
                        f"PySCF RKS CPKS residual refinement failed: {error}"
                    ) from error
                normalized_correction = _validated_float64_array(
                    normalized_correction,
                    flat_residual[active].shape,
                    "PySCF RKS CPKS refinement response",
                )
                correction[active] = (
                    normalized_correction * root_scales[active, None, None]
                )
                response[..., virtual, :] += correction.reshape(
                    *perturbation_shape,
                    nvir,
                    nocc,
                )
            residual = self._orbital_residual(
                response,
                hamiltonian_mo,
                overlap_mo,
                coefficient,
                energy,
                occupation,
                occupied,
                virtual,
            )
            residual_history.append(float(np.max(np.abs(residual), initial=0.0)))
        return response, residual, tuple(residual_history)

    def solve(self, atom_indices=None) -> RKSResponse:
        """Return the audited finite-grid response for selected atoms."""
        return self._solve(atom_indices, "response")

    def _solve_with_density_partitions(self, atom_indices=None):
        """Return a response and its transient AO density work arrays."""
        return self._solve(atom_indices, "partitions")

    def _solve_for_gradient(self, objective, atom_indices=None):
        """Return compact diagnostics and the final density contraction."""
        return self._solve(atom_indices, "gradient", objective)

    def _response_invariants(
        self,
        hamiltonian_derivatives,
        mo_response,
        density_response,
        overlap,
        overlap_derivative,
        overlap_mo,
        occupied,
        atom_indices,
    ):
        complete, fixed_grid, grid_coordinate, grid_weight = hamiltonian_derivatives
        reconstruction = float(
            np.max(
                np.abs(complete - fixed_grid - grid_coordinate - grid_weight),
                initial=0.0,
            )
        )
        overlap_occupied = overlap_mo[..., occupied, :]
        occupied_response = mo_response[..., occupied, :]
        metric = max(
            float(
                np.max(
                    np.abs(
                        occupied_response
                        + occupied_response.swapaxes(-1, -2)
                        + overlap_occupied
                    ),
                    initial=0.0,
                )
            ),
            float(
                np.max(
                    np.abs(occupied_response + 0.5 * overlap_occupied),
                    initial=0.0,
                )
            ),
        )
        density_ground = np.asarray(self.reference.make_rdm1())
        idempotency = (
            np.einsum("...ij,jk,kl->...il", density_response, overlap, density_ground)
            + np.einsum(
                "ij,...jk,kl->...il",
                density_ground,
                overlap_derivative,
                density_ground,
            )
            + np.einsum("ij,jk,...kl->...il", density_ground, overlap, density_response)
            - 2.0 * density_response
        )
        particle_number = (
            np.einsum("...ij,ji->...", density_response, overlap)
            + np.einsum("ij,...ji->...", density_ground, overlap_derivative)
        )
        translation = (
            float(np.max(np.abs(np.sum(density_response, axis=0)), initial=0.0))
            if len(atom_indices) == self.molecule.natm
            else None
        )
        try:
            ao = self.reference._numint.eval_ao(
                self.molecule, self.reference.grids.coords, deriv=0
            )
            rho = np.einsum(
                "gp,pq,gq->g", ao, density_ground, ao, optimize=True
            )
            quadrature_electrons = float(np.dot(self.reference.grids.weights, rho))
        except Exception as error:
            raise RKSResponseError(
                f"RKS quadrature electron-count audit failed: {error}"
            ) from error
        return (
            reconstruction,
            metric,
            float(np.max(np.abs(idempotency), initial=0.0)),
            float(np.max(np.abs(particle_number), initial=0.0)),
            translation,
            quadrature_electrons,
        )

    def _solve(self, atom_indices, result_mode, objective=None):
        validate_rks_reference(self.reference)
        atom_indices = self._response_atom_indices(atom_indices)
        initial_fingerprint = rks_reference_fingerprint(self.reference)
        functional_provenance = _functional_provenance(self.reference)
        grid_provenance = _grid_provenance(self.reference)
        coefficient, energy, occupation, occupied, virtual, minimum_gap = self._state()
        response_dimension = int(np.count_nonzero(occupied)) * int(
            np.count_nonzero(virtual)
        )
        overlap = np.asarray(self.reference.get_ovlp())
        overlap_derivative = self._overlap_derivative(atom_indices)
        (
            hamiltonian_derivative,
            hamiltonian_derivative_fixed_grid,
            xc_grid_coordinate,
            xc_grid_weight,
        ) = self._hamiltonian_derivative(coefficient, occupation, atom_indices)
        occupied_coefficients = coefficient[:, occupied]
        hamiltonian_mo = np.einsum(
            "mp,...mn,ni->...pi",
            coefficient,
            hamiltonian_derivative,
            occupied_coefficients,
        )
        overlap_mo = np.einsum(
            "mp,...mn,ni->...pi",
            coefficient,
            overlap_derivative,
            occupied_coefficients,
        )
        mo_response, residual, residual_history = self._solve_orbitals(
            hamiltonian_mo,
            overlap_mo,
            coefficient,
            energy,
            occupation,
            occupied,
            virtual,
        )
        if result_mode == "partitions":
            metric_response = np.zeros_like(mo_response)
            metric_response[..., occupied, :] = mo_response[..., occupied, :]
            occupied_virtual_response = np.zeros_like(mo_response)
            occupied_virtual_response[..., virtual, :] = mo_response[..., virtual, :]
            density_metric = self._density_from_mo_response(
                metric_response, coefficient, occupation, occupied
            )
            density_occupied_virtual = self._density_from_mo_response(
                occupied_virtual_response, coefficient, occupation, occupied
            )
            density_response = density_metric + density_occupied_virtual
        else:
            density_response = self._density_from_mo_response(
                mo_response, coefficient, occupation, occupied
            )
        (
            hamiltonian_reconstruction_residual,
            metric_residual,
            idempotency_residual,
            particle_number_residual,
            translation_residual,
            quadrature_electron_count,
        ) = self._response_invariants(
            (
                hamiltonian_derivative,
                hamiltonian_derivative_fixed_grid,
                xc_grid_coordinate,
                xc_grid_weight,
            ),
            mo_response,
            density_response,
            overlap,
            overlap_derivative,
            overlap_mo,
            occupied,
            atom_indices,
        )
        diagnostics = RKSResponseDiagnostics(
            minimum_orbital_gap=minimum_gap,
            pyscf_version=pyscf.__version__,
            libxc_version=str(libxc.__version__),
            functional_components=functional_provenance.components,
            grid_point_count=grid_provenance.point_count,
            grid_coordinates_fingerprint=grid_provenance.coordinates_fingerprint,
            grid_weights_fingerprint=grid_provenance.weights_fingerprint,
            quadrature_electron_count=quadrature_electron_count,
            cphf_tolerance=self.cphf_tolerance,
            maximum_residual=float(np.max(np.abs(residual), initial=0.0)),
            residual_rms=float(np.sqrt(np.mean(np.square(residual)))),
            residual_tolerance=self.residual_tolerance,
            invariant_tolerance=self.invariant_tolerance,
            orbital_gap_tolerance=self.orbital_gap_tolerance,
            max_cycle=self.max_cycle,
            max_refinement_cycles=self.max_refinement_cycles,
            level_shift=self.level_shift,
            response_dimension=response_dimension,
            operator_is_self_adjoint=True,
            hamiltonian_reconstruction_residual=hamiltonian_reconstruction_residual,
            metric_residual=metric_residual,
            idempotency_residual=idempotency_residual,
            particle_number_residual=particle_number_residual,
            translation_residual=translation_residual,
            refinement_cycles=len(residual_history) - 1,
            residual_history=residual_history,
        )
        arrays = {
            "mo_response": mo_response,
            "density_response": density_response,
            "overlap_derivative": overlap_derivative,
            "hamiltonian_derivative": hamiltonian_derivative,
            "hamiltonian_derivative_fixed_grid": hamiltonian_derivative_fixed_grid,
            "xc_hamiltonian_derivative_grid_coordinate": xc_grid_coordinate,
            "xc_hamiltonian_derivative_grid_weight": xc_grid_weight,
            "orbital_response_residual": residual,
        }
        if not all(np.isfinite(value).all() for value in arrays.values()):
            raise RKSResponseError("the RKS response contains nonfinite arrays")
        diagnostic_values = tuple(
            value
            for value in diagnostics.__dict__.values()
            if isinstance(value, (float, int))
        ) + diagnostics.residual_history
        if not np.isfinite(diagnostic_values).all():
            raise RKSResponseError("the RKS response diagnostics are nonfinite")
        if diagnostics.maximum_residual > self.residual_tolerance:
            history = " -> ".join(f"{value:.3e}" for value in residual_history)
            raise RKSResponseError(
                "RKS response residual exceeds tolerance: "
                f"{diagnostics.maximum_residual:.3e} > {self.residual_tolerance:.3e}; "
                f"refinement history: {history}"
            )
        invariant_failures = {
            "Hamiltonian reconstruction": diagnostics.hamiltonian_reconstruction_residual,
            "metric": diagnostics.metric_residual,
            "idempotency": diagnostics.idempotency_residual,
            "particle number": diagnostics.particle_number_residual,
            "translation": diagnostics.translation_residual,
        }
        invariant_failures = {
            name: value
            for name, value in invariant_failures.items()
            if value is not None and value > self.invariant_tolerance
        }
        if invariant_failures:
            details = ", ".join(
                f"{name}={value:.3e}" for name, value in invariant_failures.items()
            )
            raise RKSResponseError(
                "RKS response invariant exceeds tolerance "
                f"{self.invariant_tolerance:.3e}: {details}"
            )
        validate_rks_reference(self.reference)
        if rks_reference_fingerprint(self.reference) != initial_fingerprint:
            raise RKSResponseError("the RKS reference changed during the response solve")
        if result_mode == "gradient":
            return diagnostics, np.einsum("ij,bxij->bx", objective, density_response)
        response = RKSResponse(
            reference_identity=id(self.reference),
            state_fingerprint=initial_fingerprint,
            integrity_fingerprint="",
            atom_indices=atom_indices,
            mo_response=_immutable_array(mo_response),
            _mo_coefficients=_immutable_array(coefficient),
            _mo_occupations=_immutable_array(occupation),
            overlap_derivative=_immutable_array(overlap_derivative),
            hamiltonian_derivative=_immutable_array(hamiltonian_derivative),
            orbital_response_residual=_immutable_array(residual),
            functional_provenance=functional_provenance,
            grid_provenance=grid_provenance,
            hamiltonian_derivative_fixed_grid=_immutable_array(
                hamiltonian_derivative_fixed_grid
            ),
            xc_hamiltonian_derivative_grid_coordinate=_immutable_array(
                xc_grid_coordinate
            ),
            xc_hamiltonian_derivative_grid_weight=_immutable_array(xc_grid_weight),
            diagnostics=diagnostics,
        )
        response = replace(
            response,
            integrity_fingerprint=rks_response_integrity_fingerprint(response),
        )
        return (
            (response, (density_response, density_metric, density_occupied_virtual))
            if result_mode == "partitions"
            else response
        )

    def audit_response_equations(self, response: RKSResponse) -> None:
        from .audits.rks_response import audit_response_equations as audit
        return audit(self, response)
