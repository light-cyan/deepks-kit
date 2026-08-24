"""Internal implementation extracted from pyscf_uhf.py."""

from dataclasses import replace
import hashlib
import numpy as np
import pyscf
from .capabilities import DeePHFCapabilityError
from .contracts import array_fingerprint
from .adjoint import AdjointError, solve_scalar_adjoint
from .unrestricted_reference import (
    UHFAdjoint,
    UHFAdjointDiagnostics,
    UHFAdjointError,
    UHFResponseError,
    _cycle_limit,
    _immutable_array,
    _response_real_control,
    _validated_float64_array,
    uhf_adjoint_integrity_fingerprint,
    validate_pyscf_version,
)
from .pyscf_uhf_response_core import _UHFLinearResponseCore

class _UHFScalarAdjointProblem:
    """Bind one coupled action-only UHF operator to the adjoint protocol."""

    is_self_adjoint = True

    def __init__(
        self,
        adapter: "UHFAdjointAdapter",
        coefficient: np.ndarray,
        energy: np.ndarray,
        occupied: np.ndarray,
        virtual: np.ndarray,
    ):
        self._adapter = adapter
        self._coefficient = coefficient
        self._energy = energy
        self._occupied = occupied
        self._virtual = virtual
        dimensions = adapter._dimensions(occupied, virtual)
        self._dimension = dimensions[-2] + dimensions[-1]

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def operator_fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"pyscf-2.14-coupled-uhf-occupied-virtual-operator-v1")
        digest.update(
            self._adapter._reference_fingerprint(
                self._adapter.reference
            ).encode("ascii")
        )
        return digest.hexdigest()

    def apply(self, vector: np.ndarray) -> np.ndarray:
        return np.asarray(
            self._adapter._apply_occupied_virtual_operator(
                np.asarray(vector).reshape(self.dimension),
                self._coefficient,
                self._energy,
                self._occupied,
                self._virtual,
            )
        ).reshape(self.dimension)

    def apply_transpose(self, vector: np.ndarray) -> np.ndarray:
        # The real coupled UHF orbital Hessian is symmetric in this space.
        return self.apply(vector)

    def precondition(self, vector: np.ndarray) -> np.ndarray:
        self._adapter._count_operation("preconditioner_actions")
        alpha_gaps = (
            self._energy[0, self._virtual[0], None]
            - self._energy[0, self._occupied[0]]
        ).reshape(-1)
        beta_gaps = (
            self._energy[1, self._virtual[1], None]
            - self._energy[1, self._occupied[1]]
        ).reshape(-1)
        gaps = np.concatenate((alpha_gaps, beta_gaps))
        return np.asarray(vector).reshape(self.dimension) / gaps


class UHFAdjointAdapter(_UHFLinearResponseCore):
    """Solve one correction-specific coupled UHF scalar adjoint."""

    def __init__(
        self,
        reference,
        *,
        residual_tolerance: float = 1.0e-9,
        invariant_tolerance: float = 1.0e-9,
        orbital_gap_tolerance: float = 1.0e-7,
        operator_stability_tolerance: float = 1.0e-6,
        operator_condition_tolerance: float = 1.0e8,
        operator_symmetry_tolerance: float = 1.0e-10,
        operator_dimension_limit: int = 512,
        objective_symmetry_tolerance: float = 1.0e-10,
        max_cycle: int = 100,
        krylov_restart: int = 50,
    ):
        self._operation_hook = None
        validate_pyscf_version()
        self.reference = self._validate_reference(reference)
        self.residual_tolerance = _response_real_control(
            residual_tolerance,
            "adjoint residual_tolerance",
        )
        self.invariant_tolerance = _response_real_control(
            invariant_tolerance,
            "adjoint invariant_tolerance",
        )
        self.orbital_gap_tolerance = _response_real_control(
            orbital_gap_tolerance,
            "adjoint orbital_gap_tolerance",
        )
        self.operator_stability_tolerance = _response_real_control(
            operator_stability_tolerance,
            "adjoint operator_stability_tolerance",
        )
        self.operator_condition_tolerance = _response_real_control(
            operator_condition_tolerance,
            "adjoint operator_condition_tolerance",
        )
        self.operator_symmetry_tolerance = _response_real_control(
            operator_symmetry_tolerance,
            "adjoint operator_symmetry_tolerance",
        )
        self.operator_dimension_limit = _cycle_limit(
            operator_dimension_limit,
            "adjoint operator_dimension_limit",
        )
        self.objective_symmetry_tolerance = _response_real_control(
            objective_symmetry_tolerance,
            "adjoint objective_symmetry_tolerance",
        )
        self.max_cycle = _cycle_limit(max_cycle, "adjoint max_cycle")
        self.krylov_restart = _cycle_limit(
            krylov_restart,
            "adjoint krylov_restart",
        )
        if (
            self.residual_tolerance <= 0
            or self.invariant_tolerance <= 0
            or self.orbital_gap_tolerance <= 0
            or self.operator_stability_tolerance <= 0
            or self.operator_condition_tolerance <= 1
            or self.operator_symmetry_tolerance <= 0
            or self.objective_symmetry_tolerance <= 0
        ):
            raise ValueError("UHF adjoint tolerances are invalid")
        if self.operator_dimension_limit <= 0:
            raise ValueError("UHF adjoint operator_dimension_limit must be positive")
        if self.max_cycle <= 0 or self.krylov_restart <= 0:
            raise ValueError("UHF adjoint Krylov cycle limits must be positive")

    @staticmethod
    def _matrix_fingerprint(matrix: np.ndarray) -> str:
        return array_fingerprint(matrix)

    def _validated_objective_potential(self, value) -> np.ndarray:
        potential = _validated_float64_array(
            value,
            (self.molecule.nao, self.molecule.nao),
            "UHF correction AO objective potential",
        )
        symmetry_residual = float(
            np.max(np.abs(potential - potential.T), initial=0.0)
        )
        if symmetry_residual > self.objective_symmetry_tolerance:
            raise UHFAdjointError(
                "the UHF correction AO objective potential violates symmetry: "
                f"{symmetry_residual:.3e} > "
                f"{self.objective_symmetry_tolerance:.3e}"
            )
        return potential

    @staticmethod
    def _audited_array(value, expected_shape, name: str) -> np.ndarray:
        if type(value) is not np.ndarray:
            raise UHFAdjointError(
                f"the supplied UHF adjoint field {name} has an invalid type"
            )
        if value.shape != expected_shape:
            raise UHFAdjointError(
                f"the supplied UHF adjoint field {name} has shape {value.shape}; "
                f"expected {expected_shape}"
            )
        if value.dtype != np.dtype(np.float64) or np.iscomplexobj(value):
            raise UHFAdjointError(
                f"the supplied UHF adjoint field {name} must use real numpy.float64"
            )
        if not np.isfinite(value).all():
            raise UHFAdjointError(
                f"the supplied UHF adjoint field {name} must be finite"
            )
        if value.flags.writeable:
            raise UHFAdjointError(
                f"the supplied UHF adjoint field {name} must be immutable"
            )
        return value

    @staticmethod
    def _require_close(stored, expected, name: str) -> None:
        if not np.allclose(stored, expected, rtol=1.0e-11, atol=1.0e-12):
            raise UHFAdjointError(
                f"the supplied UHF adjoint {name} is inconsistent"
            )

    def _objective_gradients(
        self,
        objective_ao_potential: np.ndarray,
        coefficient: np.ndarray,
        occupation: np.ndarray,
        occupied: np.ndarray,
        virtual: np.ndarray,
    ) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
        objective_mo = tuple(
            coefficient[spin].T
            @ objective_ao_potential
            @ coefficient[spin]
            for spin in range(2)
        )
        gradients = tuple(
            (
                objective_mo[spin][virtual[spin]][:, occupied[spin]]
                + objective_mo[spin].T[virtual[spin]][:, occupied[spin]]
            )
            * occupation[spin, occupied[spin]]
            for spin in range(2)
        )
        return objective_mo, gradients

    def _adjoint_densities_and_potentials(
        self,
        zvector,
        coefficient,
        occupation,
        occupied,
        virtual,
    ):
        densities = []
        density_symmetry = []
        for spin in range(2):
            occupied_coefficients = coefficient[spin][:, occupied[spin]]
            virtual_coefficients = coefficient[spin][:, virtual[spin]]
            rotated = virtual_coefficients @ zvector[spin]
            one_sided = rotated @ (
                occupied_coefficients * occupation[spin, occupied[spin]]
            ).T
            density = one_sided + one_sided.T
            density = _validated_float64_array(
                density,
                (self.molecule.nao, self.molecule.nao),
                f"UHF {('alpha', 'beta')[spin]} adjoint AO density",
            )
            symmetry = float(
                np.max(np.abs(density - density.T), initial=0.0)
            )
            if symmetry > self.objective_symmetry_tolerance:
                raise UHFAdjointError(
                    "the UHF adjoint AO density violates symmetry"
                )
            densities.append(density)
            density_symmetry.append(symmetry)
        potentials = self._induced_potential(densities[0], densities[1])
        potential_symmetry = []
        validated_potentials = []
        for spin in range(2):
            potential = _validated_float64_array(
                potentials[spin],
                (self.molecule.nao, self.molecule.nao),
                f"UHF {('alpha', 'beta')[spin]} adjoint AO potential",
            )
            symmetry = float(
                np.max(np.abs(potential - potential.T), initial=0.0)
            )
            if symmetry > self.objective_symmetry_tolerance:
                raise UHFAdjointError(
                    "the UHF adjoint AO potential violates symmetry"
                )
            validated_potentials.append(potential)
            potential_symmetry.append(symmetry)
        return (
            tuple(densities),
            tuple(validated_potentials),
            tuple(density_symmetry),
            tuple(potential_symmetry),
        )

    def _gradient_partitions(
        self,
        objective_mo,
        zvector,
        adjoint_potential,
        coefficient,
        energy,
        occupation,
        occupied,
        virtual,
        atom_indices=None,
        compact=False,
    ):
        atom_indices = self._response_atom_indices(atom_indices)
        overlap_derivative = self._overlap_derivative(atom_indices)
        hamiltonian_derivative = self._hamiltonian_derivative(
            coefficient,
            occupation,
            atom_indices,
        )
        metric_spin = []
        nuclear_spin = []
        adjoint_metric_spin = []
        response = np.zeros((len(atom_indices), 3), dtype=np.float64)
        for spin in range(2):
            occupied_coefficients = coefficient[spin][:, occupied[spin]]
            overlap_mo = np.einsum(
                "mp,...mn,ni->...pi",
                coefficient[spin],
                overlap_derivative,
                occupied_coefficients,
            )
            hamiltonian_mo = np.einsum(
                "mp,...mn,ni->...pi",
                coefficient[spin],
                hamiltonian_derivative[spin],
                occupied_coefficients,
            )
            bare_rhs = (
                hamiltonian_mo[..., virtual[spin], :]
                - overlap_mo[..., virtual[spin], :]
                * energy[spin, occupied[spin]]
            )
            nuclear = -np.einsum("ai,...ai->...", zvector[spin], bare_rhs)
            objective_occupied = objective_mo[spin][occupied[spin]][
                :, occupied[spin]
            ]
            objective_occupied = 0.5 * (
                objective_occupied + objective_occupied.T
            )
            adjoint_potential_mo = (
                coefficient[spin].T
                @ adjoint_potential[spin]
                @ coefficient[spin]
            )
            adjoint_potential_occupied = adjoint_potential_mo[occupied[spin]][
                :, occupied[spin]
            ]
            adjoint_potential_occupied = 0.5 * (
                adjoint_potential_occupied
                + adjoint_potential_occupied.T
            )
            overlap_occupied = overlap_mo[..., occupied[spin], :]
            metric = np.einsum(
                "...ij,ij->...", overlap_occupied, -objective_occupied
            )
            adjoint_metric = np.einsum(
                "...ij,ij->...",
                overlap_occupied,
                0.5 * adjoint_potential_occupied,
            )
            if compact:
                response += metric + nuclear + adjoint_metric
            else:
                metric_spin.append(metric)
                nuclear_spin.append(nuclear)
                adjoint_metric_spin.append(adjoint_metric)
        if compact:
            return _validated_float64_array(
                response,
                (len(atom_indices), 3),
                "UHF adjoint response gradient",
            )
        metric_spin = np.stack(metric_spin)
        nuclear_spin = np.stack(nuclear_spin)
        adjoint_metric_spin = np.stack(adjoint_metric_spin)
        occupied_virtual_spin = nuclear_spin + adjoint_metric_spin
        metric = metric_spin.sum(axis=0)
        nuclear = nuclear_spin.sum(axis=0)
        adjoint_metric = adjoint_metric_spin.sum(axis=0)
        occupied_virtual = occupied_virtual_spin.sum(axis=0)
        response = metric + occupied_virtual
        reconstruction_residual = max(
            float(
                np.max(
                    np.abs(occupied_virtual_spin - nuclear_spin - adjoint_metric_spin),
                    initial=0.0,
                )
            ),
            float(
                np.max(
                    np.abs(response - metric - occupied_virtual),
                    initial=0.0,
                )
            ),
        )
        return (
            metric_spin,
            metric,
            nuclear_spin,
            nuclear,
            adjoint_metric_spin,
            adjoint_metric,
            occupied_virtual_spin,
            occupied_virtual,
            response,
            reconstruction_residual,
        )

    def solve(self, objective_ao_potential: np.ndarray, atom_indices=None, compact=False):
        """Return one audited UHF adjoint and selected nuclear contractions."""
        try:
            return self._solve(objective_ao_potential, atom_indices, compact=compact)
        except DeePHFCapabilityError:
            raise
        except UHFAdjointError:
            raise
        except (AdjointError, UHFResponseError) as error:
            raise UHFAdjointError(f"UHF adjoint evaluation failed: {error}") from error

    def _solve(self, objective_ao_potential: np.ndarray, atom_indices=None, compact=False):
        self._validate_reference(self.reference)
        atom_indices = self._response_atom_indices(atom_indices)
        objective = self._validated_objective_potential(objective_ao_potential)
        objective_symmetry_residual = float(
            np.max(np.abs(objective - objective.T), initial=0.0)
        )
        coefficient, energy, occupation, occupied, virtual, minimum_gaps = (
            self._state()
        )
        *_, alpha_dimension, beta_dimension = self._dimensions(occupied, virtual)
        response_dimension = alpha_dimension + beta_dimension
        objective_mo, objective_gradients = self._objective_gradients(
            objective,
            coefficient,
            occupation,
            occupied,
            virtual,
        )
        objective_vector = np.concatenate(
            tuple(value.reshape(-1) for value in objective_gradients)
        )
        problem = _UHFScalarAdjointProblem(
            self,
            coefficient,
            energy,
            occupied,
            virtual,
        )
        linear_result = solve_scalar_adjoint(
            problem,
            objective_vector,
            residual_tolerance=self.residual_tolerance,
            require_physical_residual=True,
            solver="gmres",
            max_cycle=self.max_cycle,
            restart=self.krylov_restart,
        )
        zvector = self._split_occupied_virtual(
            linear_result.solution,
            occupied,
            virtual,
        )
        residual = self._split_occupied_virtual(
            linear_result.residual,
            occupied,
            virtual,
        )
        (
            adjoint_density,
            adjoint_potential,
            density_symmetry,
            potential_symmetry,
        ) = self._adjoint_densities_and_potentials(
            zvector,
            coefficient,
            occupation,
            occupied,
            virtual,
        )
        gradient_data = self._gradient_partitions(
            objective_mo,
            zvector,
            adjoint_potential,
            coefficient,
            energy,
            occupation,
            occupied,
            virtual,
            atom_indices,
            compact=compact,
        )
        reconstruction_residual = 0.0 if compact else gradient_data[-1]
        if reconstruction_residual > self.invariant_tolerance:
            raise UHFAdjointError(
                "the UHF adjoint gradient reconstruction residual exceeds tolerance"
            )
        if compact:
            correction_gradient_response = gradient_data
        else:
            partitions = gradient_data
            for name, value in zip(
                (
                    "metric spin gradient",
                    "metric gradient",
                    "nuclear spin gradient",
                    "nuclear gradient",
                    "adjoint metric spin gradient",
                    "adjoint metric gradient",
                    "occupied-virtual spin gradient",
                    "occupied-virtual gradient",
                    "response gradient",
                ),
                partitions[:-1],
                strict=True,
            ):
                expected_shape = (
                    (2, len(atom_indices), 3)
                    if "spin" in name
                    else (len(atom_indices), 3)
                )
                _validated_float64_array(value, expected_shape, f"UHF adjoint {name}")
        self._validate_reference(self.reference)
        linear_diagnostics = linear_result.diagnostics
        diagnostics = UHFAdjointDiagnostics(
            minimum_alpha_orbital_gap=minimum_gaps[0],
            minimum_beta_orbital_gap=minimum_gaps[1],
            pyscf_version=pyscf.__version__,
            residual_tolerance=self.residual_tolerance,
            invariant_tolerance=self.invariant_tolerance,
            orbital_gap_tolerance=self.orbital_gap_tolerance,
            response_dimension=response_dimension,
            alpha_response_dimension=alpha_dimension,
            beta_response_dimension=beta_dimension,
            operator_is_self_adjoint=True,
            objective_symmetry_tolerance=self.objective_symmetry_tolerance,
            objective_symmetry_residual=objective_symmetry_residual,
            alpha_adjoint_density_symmetry_residual=density_symmetry[0],
            beta_adjoint_density_symmetry_residual=density_symmetry[1],
            alpha_adjoint_potential_symmetry_residual=potential_symmetry[0],
            beta_adjoint_potential_symmetry_residual=potential_symmetry[1],
            gradient_reconstruction_residual=reconstruction_residual,
            solver=linear_diagnostics.solver,
            solve_count=linear_diagnostics.solve_count,
            objective_gradient_norm=linear_diagnostics.objective_gradient_norm,
            solution_norm=linear_diagnostics.solution_norm,
            maximum_residual=linear_diagnostics.maximum_residual,
            residual_rms=linear_diagnostics.residual_rms,
            max_cycle=self.max_cycle,
            krylov_restart=self.krylov_restart,
            iteration_count=linear_diagnostics.iteration_count,
        )
        if compact:
            return diagnostics, correction_gradient_response
        adjoint = UHFAdjoint(
            reference_identity=id(self.reference),
            state_fingerprint=self._reference_fingerprint(self.reference),
            integrity_fingerprint="",
            operator_fingerprint=linear_result.operator_fingerprint,
            atom_indices=atom_indices,
            objective_ao_potential=_immutable_array(objective),
            alpha_objective_orbital_gradient=_immutable_array(
                objective_gradients[0]
            ),
            beta_objective_orbital_gradient=_immutable_array(
                objective_gradients[1]
            ),
            alpha_zvector=_immutable_array(zvector[0]),
            beta_zvector=_immutable_array(zvector[1]),
            alpha_residual=_immutable_array(residual[0]),
            beta_residual=_immutable_array(residual[1]),
            alpha_adjoint_ao_density=_immutable_array(adjoint_density[0]),
            beta_adjoint_ao_density=_immutable_array(adjoint_density[1]),
            alpha_adjoint_ao_potential=_immutable_array(adjoint_potential[0]),
            beta_adjoint_ao_potential=_immutable_array(adjoint_potential[1]),
            correction_gradient_metric_spin=_immutable_array(partitions[0]),
            correction_gradient_metric=_immutable_array(partitions[1]),
            correction_gradient_adjoint_nuclear_spin=_immutable_array(
                partitions[2]
            ),
            correction_gradient_adjoint_nuclear=_immutable_array(partitions[3]),
            correction_gradient_adjoint_metric_spin=_immutable_array(
                partitions[4]
            ),
            correction_gradient_adjoint_metric=_immutable_array(partitions[5]),
            correction_gradient_occupied_virtual_spin=_immutable_array(
                partitions[6]
            ),
            correction_gradient_occupied_virtual=_immutable_array(partitions[7]),
            correction_gradient_response=_immutable_array(partitions[8]),
            diagnostics=diagnostics,
        )
        return replace(
            adjoint,
            integrity_fingerprint=uhf_adjoint_integrity_fingerprint(adjoint),
        )

    def audit_adjoint(
        self,
        adjoint: UHFAdjoint,
        expected_objective_ao_potential: np.ndarray,
    ) -> None:
        from .audits.unrestricted_adjoint import audit_adjoint as audit
        return audit(self, adjoint, expected_objective_ao_potential)
