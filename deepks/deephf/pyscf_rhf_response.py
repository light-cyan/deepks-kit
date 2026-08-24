"""Internal implementation extracted from pyscf_rhf.py."""

from dataclasses import replace
import operator
import numpy as np
import pyscf
from pyscf.hessian import rhf as rhf_hessian
from pyscf.scf import cphf, hf as scf_hf
from .capabilities import DeePHFCapabilityError
from .pyscf_rhf_reference import (
    RHFResponse,
    RHFResponseDiagnostics,
    RHFResponseError,
    _immutable_array,
    reference_fingerprint,
    validate_pyscf_version,
    validate_reference,
)
from .scanner import (
    _cycle_limit,
    _validated_float64_array,
    response_integrity_fingerprint,
)
from .restricted_response import RestrictedResponseAlgebra


class _RHFLinearResponseCore(RestrictedResponseAlgebra):
    """Shared PySCF 2.14 molecular RHF linear-response operations."""

    def __init__(
        self,
        reference,
        *,
        cphf_tolerance: float = 1.0e-11,
        residual_tolerance: float = 1.0e-9,
        invariant_tolerance: float = 1.0e-9,
        orbital_gap_tolerance: float = 1.0e-7,
        max_cycle: int = 100,
        max_refinement_cycles: int = 3,
        level_shift: float = 0.0,
        operator_stability_tolerance: float = 1.0e-6,
        operator_condition_tolerance: float = 1.0e8,
        operator_symmetry_tolerance: float = 1.0e-10,
        operator_dimension_limit: int = 512,
    ):
        self._operation_hook = None
        validate_pyscf_version()
        self.reference = validate_reference(reference)
        self.cphf_tolerance = float(cphf_tolerance)
        self.residual_tolerance = float(residual_tolerance)
        self.invariant_tolerance = float(invariant_tolerance)
        self.orbital_gap_tolerance = float(orbital_gap_tolerance)
        self.max_cycle = _cycle_limit(max_cycle, "max_cycle")
        self.max_refinement_cycles = _cycle_limit(
            max_refinement_cycles,
            "max_refinement_cycles",
        )
        self.level_shift = float(level_shift)
        self.operator_stability_tolerance = float(
            operator_stability_tolerance
        )
        self.operator_condition_tolerance = float(
            operator_condition_tolerance
        )
        self.operator_symmetry_tolerance = float(
            operator_symmetry_tolerance
        )
        self.operator_dimension_limit = _cycle_limit(
            operator_dimension_limit,
            "operator_dimension_limit",
        )
        tolerance_values = (
            self.cphf_tolerance,
            self.residual_tolerance,
            self.invariant_tolerance,
            self.orbital_gap_tolerance,
            self.operator_stability_tolerance,
            self.operator_condition_tolerance,
            self.operator_symmetry_tolerance,
        )
        if not np.isfinite(tolerance_values).all():
            raise ValueError("response tolerances must be finite")
        if not np.isfinite(self.level_shift):
            raise ValueError("response level_shift must be finite")
        if self.cphf_tolerance <= 0 or self.residual_tolerance <= 0:
            raise ValueError("response tolerances must be positive")
        if self.invariant_tolerance <= 0 or self.orbital_gap_tolerance <= 0:
            raise ValueError("response tolerances must be positive")
        if (
            self.operator_stability_tolerance <= 0
            or self.operator_condition_tolerance <= 1
            or self.operator_symmetry_tolerance <= 0
        ):
            raise ValueError("response operator tolerances are invalid")
        if self.max_cycle <= 0 or self.max_refinement_cycles < 0:
            raise ValueError("response cycle limits are invalid")
        if self.operator_dimension_limit <= 0:
            raise ValueError("response operator_dimension_limit must be positive")

    def _count_operation(self, name: str) -> None:
        if self._operation_hook is not None:
            self._operation_hook(name)

    @property
    def molecule(self):
        return self.reference.mol

    def _response_atom_indices(self, atom_indices) -> tuple[int, ...]:
        if atom_indices is None:
            return tuple(range(self.molecule.natm))
        try:
            indices = tuple(atom_indices)
        except TypeError as error:
            raise TypeError("response atom_indices must be iterable") from error
        if not indices:
            raise ValueError("response atom_indices must not be empty")
        validated_indices = []
        for atom_index in indices:
            if isinstance(atom_index, (bool, np.bool_)):
                raise TypeError("response atom indices must be integers")
            try:
                validated = operator.index(atom_index)
            except TypeError as error:
                raise TypeError("response atom indices must be integers") from error
            if validated < 0 or validated >= self.molecule.natm:
                raise IndexError("response atom index is outside the molecule")
            if validated in validated_indices:
                raise ValueError(
                    "response atom_indices must not contain duplicates"
                )
            validated_indices.append(validated)
        return tuple(validated_indices)

    def _state(self):
        coefficient = np.asarray(self.reference.mo_coeff)
        energy = np.asarray(self.reference.mo_energy)
        occupation = np.asarray(self.reference.mo_occ)
        occupied = occupation > 0
        virtual = occupation == 0
        if not np.any(occupied) or not np.any(virtual):
            raise DeePHFCapabilityError(
                "RHF response requires occupied and virtual orbitals"
            )
        gaps = energy[virtual, None] - energy[occupied]
        minimum_gap = float(np.min(gaps))
        if not np.isfinite(minimum_gap) or minimum_gap <= self.orbital_gap_tolerance:
            raise DeePHFCapabilityError(
                "RHF occupied-virtual gap is outside the strict response domain: "
                f"{minimum_gap:.3e} <= {self.orbital_gap_tolerance:.3e}"
            )
        return coefficient, energy, occupation, occupied, virtual, minimum_gap

    def _overlap_derivative(self, atom_indices=None) -> np.ndarray:
        molecule = self.molecule
        atom_indices = self._response_atom_indices(atom_indices)
        nao = molecule.nao
        try:
            integral = -molecule.intor("int1e_ipovlp", comp=3)
        except Exception as error:
            raise RHFResponseError(
                f"PySCF overlap-derivative construction failed: {error}"
            ) from error
        integral = _validated_float64_array(
            integral,
            (3, nao, nao),
            "overlap-derivative integral",
        )
        result = np.zeros((len(atom_indices), 3, nao, nao))
        atom_slices = molecule.aoslice_by_atom()
        for result_index, atom_index in enumerate(atom_indices):
            atom_slice = atom_slices[atom_index]
            ao_start, ao_stop = atom_slice[2:]
            result[result_index, :, ao_start:ao_stop] += integral[
                :, ao_start:ao_stop
            ]
            result[result_index, :, :, ao_start:ao_stop] += integral[
                :, ao_start:ao_stop
            ].transpose(0, 2, 1)
        return result

    def _hamiltonian_derivative(
        self,
        coefficient: np.ndarray,
        occupation: np.ndarray,
        atom_indices=None,
    ) -> np.ndarray:
        atom_indices = self._response_atom_indices(atom_indices)
        try:
            hessian = rhf_hessian.Hessian(self.reference)
            derivatives = hessian.make_h1(
                coefficient,
                occupation,
                atmlst=atom_indices,
            )
            derivatives = [derivatives[index] for index in atom_indices]
        except Exception as error:
            raise RHFResponseError(
                f"PySCF RHF Hamiltonian derivative construction failed: {error}"
            ) from error
        if derivatives is None:
            raise RHFResponseError(
                "PySCF RHF Hamiltonian derivative is incomplete"
            )
        expected = (len(atom_indices), 3, self.molecule.nao, self.molecule.nao)
        return _validated_float64_array(
            derivatives,
            expected,
            "Hamiltonian derivative",
        )

    def _induced_potential(self, density_response: np.ndarray) -> np.ndarray:
        flat_density = np.asarray(density_response).reshape(
            -1,
            self.molecule.nao,
            self.molecule.nao,
        )
        try:
            coulomb, exchange = scf_hf.get_jk(
                self.molecule,
                flat_density,
                hermi=1,
            )
            coulomb = _validated_float64_array(
                coulomb,
                flat_density.shape,
                "induced Coulomb response",
            )
            exchange = _validated_float64_array(
                exchange,
                flat_density.shape,
                "induced exchange response",
            )
        except RHFResponseError:
            raise
        except Exception as error:
            raise RHFResponseError(
                f"PySCF induced-potential construction failed: {error}"
            ) from error
        return (coulomb - 0.5 * exchange).reshape(density_response.shape)

    def _response_operator_matrix_and_diagnostics(
        self,
        coefficient: np.ndarray,
        energy: np.ndarray,
        occupation: np.ndarray,
        occupied: np.ndarray,
        virtual: np.ndarray,
    ) -> tuple[np.ndarray, int, float, float, float, float]:
        from .audits.rhf_response_audit import _response_operator_matrix_and_diagnostics as audit
        return audit(self, coefficient, energy, occupation, occupied, virtual)

    def validate_response_operator_exact(self) -> tuple[int, float, float, float, float]:
        from .audits.rhf_response_audit import validate_response_operator_exact as audit
        return audit(self)

    def _apply_occupied_virtual_operator(
        self,
        occupied_virtual_response: np.ndarray,
        coefficient: np.ndarray,
        energy: np.ndarray,
        occupation: np.ndarray,
        occupied: np.ndarray,
        virtual: np.ndarray,
    ) -> np.ndarray:
        """Apply the physical unshifted RHF operator to virtual-occupied amplitudes."""
        self._count_operation("response_operator_actions")
        nocc = int(np.count_nonzero(occupied))
        nvir = int(np.count_nonzero(virtual))
        response = np.asarray(occupied_virtual_response)
        if response.shape[-2:] != (nvir, nocc):
            raise RHFResponseError(
                "the RHF occupied-virtual response has an invalid shape"
            )
        full_response = np.zeros(
            (*response.shape[:-2], coefficient.shape[1], nocc),
            dtype=np.float64,
        )
        full_response[..., virtual, :] = response
        induced = self._induced_mo_potential(
            full_response,
            coefficient,
            occupation,
            occupied,
        )[..., virtual, :]
        orbital_gap = energy[virtual, None] - energy[occupied]
        return orbital_gap * response + induced


class RHFResponseAdapter(_RHFLinearResponseCore):
    """Solve and audit molecular RHF nuclear CPHF through PySCF 2.14."""

    def coordinate_blocks(
        self,
        block_size: int,
        atom_indices=None,
        result_mode="response",
        objective=None,
    ):
        """Yield audited responses while retaining at most one atom block."""
        block_size = _cycle_limit(block_size, "coordinate_block_size")
        if block_size <= 0:
            raise ValueError("coordinate_block_size must be positive")
        selected_atoms = self._response_atom_indices(atom_indices)
        for start in range(0, len(selected_atoms), block_size):
            block_atoms = selected_atoms[start : start + block_size]
            result = (
                self._solve(block_atoms, result_mode, objective)
                if result_mode == "gradient"
                else self._solve(block_atoms, result_mode)
            )
            yield block_atoms, result

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
        induced_mo = self._induced_mo_potential(
            mo_response,
            coefficient,
            occupation,
            occupied,
        )
        residual = (
            hamiltonian_mo
            + induced_mo
            - overlap_mo * energy[occupied]
            + (energy[:, None] - energy[occupied]) * mo_response
        )
        return residual[..., virtual, :]

    def audit_response_equations(self, response: RHFResponse) -> None:
        from .audits.rhf_response_audit import audit_response_equations as audit
        return audit(self, response)

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
        flattened_hamiltonian = hamiltonian_mo.reshape(-1, nmo, nocc)
        flattened_overlap = overlap_mo.reshape(-1, nmo, nocc)

        def induced_full(response):
            self._count_operation("response_operator_actions")
            response = np.asarray(response).reshape(-1, nmo, nocc)
            return self._induced_mo_potential(
                response,
                coefficient,
                occupation,
                occupied,
            )

        try:
            response, _ = cphf.solve(
                induced_full,
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
            raise RHFResponseError(f"PySCF RHF CPHF solve failed: {error}") from error
        response = _validated_float64_array(
            response,
            flattened_hamiltonian.shape,
            "PySCF RHF CPHF response",
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
            flat_residual = residual.reshape(-1, int(np.count_nonzero(virtual)), nocc)
            root_scales = np.linalg.norm(flat_residual.reshape(len(flat_residual), -1), axis=1)
            active = root_scales > np.finfo(float).eps
            correction = np.zeros_like(flat_residual)

            def induced_virtual(virtual_response):
                self._count_operation("response_operator_actions")
                virtual_response = np.asarray(virtual_response).reshape(
                    -1,
                    int(np.count_nonzero(virtual)),
                    nocc,
                )
                full_response = np.zeros(
                    (len(virtual_response), nmo, nocc),
                    dtype=virtual_response.dtype,
                )
                full_response[:, virtual] = virtual_response
                return induced_full(full_response)[:, virtual]

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
                    raise RHFResponseError(
                        f"PySCF RHF CPHF residual refinement failed: {error}"
                    ) from error
                normalized_correction = _validated_float64_array(
                    normalized_correction,
                    flat_residual[active].shape,
                    "PySCF RHF CPHF refinement response",
                )
                correction[active] = (
                    normalized_correction * root_scales[active, None, None]
                )
                response[..., virtual, :] += correction.reshape(
                    *perturbation_shape,
                    int(np.count_nonzero(virtual)),
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
            residual_history.append(
                float(np.max(np.abs(residual), initial=0.0))
            )
        return response, residual, tuple(residual_history)

    def solve(self, atom_indices=None) -> RHFResponse:
        """Return an audited first-order response for selected atoms."""
        return self._solve(atom_indices, "response")

    def _solve_with_density_partitions(self, atom_indices=None):
        """Return a response and its transient AO density work arrays."""
        return self._solve(atom_indices, "partitions")

    def _solve_for_gradient(self, objective, atom_indices=None):
        """Return compact diagnostics and the final density contraction."""
        return self._solve(atom_indices, "gradient", objective)

    def _solve(self, atom_indices, result_mode, objective=None):
        validate_reference(self.reference)
        atom_indices = self._response_atom_indices(atom_indices)
        (
            coefficient,
            energy,
            occupation,
            occupied,
            virtual,
            minimum_gap,
        ) = self._state()
        response_dimension = int(np.count_nonzero(occupied)) * int(
            np.count_nonzero(virtual)
        )
        overlap = np.asarray(self.reference.get_ovlp())
        overlap_derivative = self._overlap_derivative(atom_indices)
        hamiltonian_derivative = self._hamiltonian_derivative(
            coefficient,
            occupation,
            atom_indices,
        )
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
        density_ground = self.reference.make_rdm1()
        overlap_occupied = overlap_mo[..., occupied, :]
        metric_residual = np.max(
            np.abs(
                mo_response[..., occupied, :]
                + mo_response[..., occupied, :].swapaxes(-1, -2)
                + overlap_occupied
            ),
            initial=0.0,
        )
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
        diagnostics = RHFResponseDiagnostics(
            minimum_orbital_gap=minimum_gap,
            pyscf_version=pyscf.__version__,
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
            metric_residual=float(metric_residual),
            idempotency_residual=float(np.max(np.abs(idempotency), initial=0.0)),
            particle_number_residual=float(
                np.max(np.abs(particle_number), initial=0.0)
            ),
            refinement_cycles=len(residual_history) - 1,
            residual_history=residual_history,
        )
        arrays = {
            "mo_response": mo_response,
            "density_response": density_response,
            "overlap_derivative": overlap_derivative,
            "hamiltonian_derivative": hamiltonian_derivative,
            "orbital_response_residual": residual,
        }
        nonfinite = [name for name, value in arrays.items() if not np.isfinite(value).all()]
        if nonfinite:
            raise RHFResponseError(
                f"nonfinite RHF response quantities: {', '.join(nonfinite)}"
            )
        diagnostic_values = (
            diagnostics.minimum_orbital_gap,
            diagnostics.maximum_residual,
            diagnostics.residual_rms,
            diagnostics.metric_residual,
            diagnostics.idempotency_residual,
            diagnostics.particle_number_residual,
            *diagnostics.residual_history,
        )
        if not np.isfinite(diagnostic_values).all():
            raise RHFResponseError("nonfinite RHF response diagnostics")
        if diagnostics.maximum_residual > self.residual_tolerance:
            history = " -> ".join(
                f"{value:.3e}" for value in diagnostics.residual_history
            )
            raise RHFResponseError(
                "RHF response residual exceeds tolerance: "
                f"{diagnostics.maximum_residual:.3e} > {self.residual_tolerance:.3e}; "
                f"refinement history: {history}"
            )
        invariant_failures = {
            "metric": diagnostics.metric_residual,
            "idempotency": diagnostics.idempotency_residual,
            "particle number": diagnostics.particle_number_residual,
        }
        invariant_failures = {
            name: value
            for name, value in invariant_failures.items()
            if value > self.invariant_tolerance
        }
        if invariant_failures:
            details = ", ".join(
                f"{name}={value:.3e}" for name, value in invariant_failures.items()
            )
            raise RHFResponseError(
                "RHF response invariant exceeds tolerance "
                f"{self.invariant_tolerance:.3e}: {details}"
            )
        if result_mode == "gradient":
            return diagnostics, np.einsum("ij,bxij->bx", objective, density_response)
        response = RHFResponse(
            reference_identity=id(self.reference),
            state_fingerprint=reference_fingerprint(self.reference),
            integrity_fingerprint="",
            atom_indices=atom_indices,
            mo_response=_immutable_array(mo_response),
            _mo_coefficients=_immutable_array(coefficient),
            _mo_occupations=_immutable_array(occupation),
            overlap_derivative=_immutable_array(overlap_derivative),
            hamiltonian_derivative=_immutable_array(
                hamiltonian_derivative
            ),
            orbital_response_residual=_immutable_array(residual),
            diagnostics=diagnostics,
        )
        response = replace(
            response,
            integrity_fingerprint=response_integrity_fingerprint(response),
        )
        return (
            (response, (density_response, density_metric, density_occupied_virtual))
            if result_mode == "partitions"
            else response
        )
