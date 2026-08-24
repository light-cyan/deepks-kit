"""Internal implementation extracted from pyscf_rhf.py."""

from dataclasses import dataclass, replace
import hashlib
from typing import Any
import numpy as np
import pyscf
from .adjoint import AdjointError, solve_scalar_adjoint
from .capabilities import DeePHFCapabilityError
from .contracts import update_digest
from .pyscf_rhf_reference import (
    RHFAdjoint,
    RHFAdjointDiagnostics,
    RHFAdjointError,
    RHFResponseError,
    _immutable_array,
    reference_fingerprint,
    validate_pyscf_version,
    validate_reference,
)
from .pyscf_rhf_response import _RHFLinearResponseCore
from .pyscf_rhf_scanner import (
    _adjoint_real_control,
    _cycle_limit,
    _validated_float64_array,
    adjoint_integrity_fingerprint,
)

class _RHFScalarAdjointProblem:
    """Bind the RHF occupied-virtual action to the adjoint protocol."""

    is_self_adjoint = True

    def __init__(
        self,
        adapter: "RHFAdjointAdapter",
        coefficient: np.ndarray,
        energy: np.ndarray,
        occupation: np.ndarray,
        occupied: np.ndarray,
        virtual: np.ndarray,
    ):
        self._adapter = adapter
        self._coefficient = coefficient
        self._energy = energy
        self._occupation = occupation
        self._occupied = occupied
        self._virtual = virtual
        self._nocc = int(np.count_nonzero(occupied))
        self._nvir = int(np.count_nonzero(virtual))

    @property
    def dimension(self) -> int:
        return self._nocc * self._nvir

    @property
    def operator_fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"pyscf-2.14-rhf-occupied-virtual-operator-v1")
        for value in (
            self._coefficient,
            self._energy,
            self._occupation,
            self._occupied,
            self._virtual,
        ):
            update_digest(digest, np.asarray(value))
        return digest.hexdigest()

    def apply(self, vector: np.ndarray) -> np.ndarray:
        amplitudes = np.asarray(vector).reshape(self._nvir, self._nocc)
        image = self._adapter._apply_occupied_virtual_operator(
            amplitudes,
            self._coefficient,
            self._energy,
            self._occupation,
            self._occupied,
            self._virtual,
        )
        return np.asarray(image).reshape(self.dimension)

    def apply_transpose(self, vector: np.ndarray) -> np.ndarray:
        # The real closed-shell orbital Hessian is symmetric in this space.
        return self.apply(vector)

    def precondition(self, vector: np.ndarray) -> np.ndarray:
        self._adapter._count_operation("preconditioner_actions")
        gaps = (
            self._energy[self._virtual, None]
            - self._energy[self._occupied]
        )
        return np.asarray(vector).reshape(self.dimension) / gaps.reshape(
            self.dimension
        )


@dataclass(frozen=True)
class _RHFOrbitalAdjoint:
    objective_symmetry_residual: float
    objective_mo: np.ndarray
    objective_orbital_gradient: np.ndarray
    linear_result: Any
    zvector: np.ndarray
    occupied_coefficients: np.ndarray
    adjoint_ao_density: np.ndarray
    adjoint_ao_potential: np.ndarray
    density_symmetry_residual: float
    potential_symmetry_residual: float


class RHFAdjointAdapter(_RHFLinearResponseCore):
    """Solve one correction-specific RHF Z-vector through PySCF 2.14."""

    def __init__(
        self,
        reference,
        *,
        residual_tolerance: float = 1.0e-9,
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
        self.reference = validate_reference(reference)
        self.residual_tolerance = _adjoint_real_control(
            residual_tolerance,
            "residual_tolerance",
        )
        self.orbital_gap_tolerance = _adjoint_real_control(
            orbital_gap_tolerance,
            "orbital_gap_tolerance",
        )
        self.operator_stability_tolerance = _adjoint_real_control(
            operator_stability_tolerance,
            "operator_stability_tolerance",
        )
        self.operator_condition_tolerance = _adjoint_real_control(
            operator_condition_tolerance,
            "operator_condition_tolerance",
        )
        self.operator_symmetry_tolerance = _adjoint_real_control(
            operator_symmetry_tolerance,
            "operator_symmetry_tolerance",
        )
        self.operator_dimension_limit = _cycle_limit(
            operator_dimension_limit,
            "operator_dimension_limit",
        )
        self.objective_symmetry_tolerance = _adjoint_real_control(
            objective_symmetry_tolerance,
            "objective_symmetry_tolerance",
        )
        self.max_cycle = _cycle_limit(max_cycle, "max_cycle")
        self.krylov_restart = _cycle_limit(
            krylov_restart,
            "krylov_restart",
        )
        if (
            self.residual_tolerance <= 0
            or self.orbital_gap_tolerance <= 0
            or self.operator_stability_tolerance <= 0
            or self.operator_condition_tolerance <= 1
            or self.operator_symmetry_tolerance <= 0
            or self.objective_symmetry_tolerance <= 0
        ):
            raise ValueError("adjoint tolerances are invalid")
        if self.operator_dimension_limit <= 0:
            raise ValueError("adjoint operator_dimension_limit must be positive")
        if self.max_cycle <= 0 or self.krylov_restart <= 0:
            raise ValueError("adjoint Krylov cycle limits must be positive")

    def _validated_objective_potential(self, value) -> np.ndarray:
        potential = _validated_float64_array(
            value,
            (self.molecule.nao, self.molecule.nao),
            "correction AO objective potential",
        )
        symmetry_residual = float(
            np.max(np.abs(potential - potential.T), initial=0.0)
        )
        if symmetry_residual > self.objective_symmetry_tolerance:
            raise RHFAdjointError(
                "the correction AO objective potential violates symmetry: "
                f"{symmetry_residual:.3e} > "
                f"{self.objective_symmetry_tolerance:.3e}"
            )
        return potential

    @staticmethod
    def _audited_array(value, expected_shape, name: str) -> np.ndarray:
        if type(value) is not np.ndarray:
            raise RHFAdjointError(
                f"the supplied RHF adjoint field {name} has an invalid type"
            )
        if value.shape != expected_shape:
            raise RHFAdjointError(
                f"the supplied RHF adjoint field {name} has shape {value.shape}; "
                f"expected {expected_shape}"
            )
        if value.dtype != np.dtype(np.float64) or np.iscomplexobj(value):
            raise RHFAdjointError(
                f"the supplied RHF adjoint field {name} must use real numpy.float64"
            )
        if not np.isfinite(value).all():
            raise RHFAdjointError(
                f"the supplied RHF adjoint field {name} must be finite"
            )
        if value.flags.writeable:
            raise RHFAdjointError(
                f"the supplied RHF adjoint field {name} must be immutable"
            )
        return value

    @staticmethod
    def _require_close(stored, expected, name: str) -> None:
        if not np.allclose(stored, expected, rtol=1.0e-11, atol=1.0e-12):
            raise RHFAdjointError(
                f"the supplied RHF adjoint {name} is inconsistent"
            )

    def _orbital_adjoint(
        self,
        objective_ao_potential,
        coefficient,
        energy,
        occupation,
        occupied,
        virtual,
    ) -> _RHFOrbitalAdjoint:
        objective_symmetry_residual = float(
            np.max(
                np.abs(objective_ao_potential - objective_ao_potential.T),
                initial=0.0,
            )
        )
        occupied_coefficients = coefficient[:, occupied]
        virtual_coefficients = coefficient[:, virtual]
        objective_mo = coefficient.T @ objective_ao_potential @ coefficient
        objective_orbital_gradient = (
            objective_mo[virtual][:, occupied]
            + objective_mo.T[virtual][:, occupied]
        ) * occupation[occupied]
        nocc = int(np.count_nonzero(occupied))
        nvir = int(np.count_nonzero(virtual))
        objective_orbital_gradient = _validated_float64_array(
            objective_orbital_gradient,
            (nvir, nocc),
            "correction occupied-virtual objective gradient",
        )
        problem = _RHFScalarAdjointProblem(
            self,
            coefficient,
            energy,
            occupation,
            occupied,
            virtual,
        )
        linear_result = solve_scalar_adjoint(
            problem,
            objective_orbital_gradient.reshape(nocc * nvir),
            residual_tolerance=self.residual_tolerance,
            require_physical_residual=True,
            solver="gmres",
            max_cycle=self.max_cycle,
            restart=self.krylov_restart,
        )
        zvector = linear_result.solution.reshape(nvir, nocc)
        rotated_occupied = virtual_coefficients @ zvector
        adjoint_density = (
            rotated_occupied @ (occupied_coefficients * occupation[occupied]).T
        )
        adjoint_density = adjoint_density + adjoint_density.T
        adjoint_density = _validated_float64_array(
            adjoint_density,
            (self.molecule.nao, self.molecule.nao),
            "RHF adjoint AO density",
        )
        density_symmetry_residual = float(
            np.max(np.abs(adjoint_density - adjoint_density.T), initial=0.0)
        )
        if density_symmetry_residual > self.objective_symmetry_tolerance:
            raise RHFAdjointError(
                "the RHF adjoint AO density violates symmetry: "
                f"{density_symmetry_residual:.3e} > "
                f"{self.objective_symmetry_tolerance:.3e}"
            )
        adjoint_potential = _validated_float64_array(
            self._induced_potential(adjoint_density),
            (self.molecule.nao, self.molecule.nao),
            "RHF adjoint AO potential",
        )
        potential_symmetry_residual = float(
            np.max(np.abs(adjoint_potential - adjoint_potential.T), initial=0.0)
        )
        if potential_symmetry_residual > self.objective_symmetry_tolerance:
            raise RHFAdjointError(
                "the RHF adjoint AO potential violates symmetry: "
                f"{potential_symmetry_residual:.3e} > "
                f"{self.objective_symmetry_tolerance:.3e}"
            )
        return _RHFOrbitalAdjoint(
            objective_symmetry_residual,
            objective_mo,
            objective_orbital_gradient,
            linear_result,
            zvector,
            occupied_coefficients,
            adjoint_density,
            adjoint_potential,
            density_symmetry_residual,
            potential_symmetry_residual,
        )

    def audit_adjoint(
        self,
        adjoint: RHFAdjoint,
        expected_objective_ao_potential: np.ndarray,
    ) -> None:
        from .audits.rhf_adjoint import audit_adjoint as audit
        return audit(self, adjoint, expected_objective_ao_potential)

    def solve(self, objective_ao_potential: np.ndarray, atom_indices=None, compact=False):
        """Return one audited Z-vector and selected nuclear contractions."""
        try:
            return self._solve(objective_ao_potential, atom_indices, compact=compact)
        except DeePHFCapabilityError:
            raise
        except RHFAdjointError:
            raise
        except (AdjointError, RHFResponseError) as error:
            raise RHFAdjointError(f"RHF adjoint evaluation failed: {error}") from error

    def _solve(self, objective_ao_potential: np.ndarray, atom_indices=None, compact=False):
        validate_reference(self.reference)
        atom_indices = self._response_atom_indices(atom_indices)
        objective_ao_potential = self._validated_objective_potential(
            objective_ao_potential
        )
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
        orbital = self._orbital_adjoint(
            objective_ao_potential,
            coefficient,
            energy,
            occupation,
            occupied,
            virtual,
        )
        objective_symmetry_residual = orbital.objective_symmetry_residual
        objective_mo = orbital.objective_mo
        objective_orbital_gradient = orbital.objective_orbital_gradient
        linear_result = orbital.linear_result
        zvector = orbital.zvector
        occupied_coefficients = orbital.occupied_coefficients
        adjoint_ao_density = orbital.adjoint_ao_density
        adjoint_ao_potential = orbital.adjoint_ao_potential
        adjoint_density_symmetry_residual = orbital.density_symmetry_residual
        adjoint_potential_symmetry_residual = orbital.potential_symmetry_residual
        nocc = int(np.count_nonzero(occupied))
        nvir = int(np.count_nonzero(virtual))
        overlap_derivative = self._overlap_derivative(atom_indices)
        hamiltonian_derivative = self._hamiltonian_derivative(
            coefficient,
            occupation,
            atom_indices,
        )
        overlap_mo = np.einsum(
            "mp,...mn,ni->...pi",
            coefficient,
            overlap_derivative,
            occupied_coefficients,
        )
        hamiltonian_mo = np.einsum(
            "mp,...mn,ni->...pi",
            coefficient,
            hamiltonian_derivative,
            occupied_coefficients,
        )
        bare_nuclear_rhs = (
            hamiltonian_mo[..., virtual, :]
            - overlap_mo[..., virtual, :] * energy[occupied]
        )
        correction_gradient_adjoint_nuclear = -np.einsum(
            "ai,...ai->...",
            zvector,
            bare_nuclear_rhs,
        )
        objective_occupied = objective_mo[occupied][:, occupied]
        objective_occupied = 0.5 * (
            objective_occupied + objective_occupied.T
        )
        adjoint_potential_mo = (
            coefficient.T @ adjoint_ao_potential @ coefficient
        )
        adjoint_potential_occupied = adjoint_potential_mo[occupied][
            :, occupied
        ]
        adjoint_potential_occupied = 0.5 * (
            adjoint_potential_occupied + adjoint_potential_occupied.T
        )
        overlap_occupied = overlap_mo[..., occupied, :]
        correction_gradient_metric = np.einsum(
            "...ij,ij->...",
            overlap_occupied,
            -2.0 * objective_occupied,
        )
        correction_gradient_adjoint_metric = np.einsum(
            "...ij,ij->...",
            overlap_occupied,
            0.5 * adjoint_potential_occupied,
        )
        correction_gradient_response = (
            correction_gradient_metric
            + correction_gradient_adjoint_nuclear
            + correction_gradient_adjoint_metric
        )
        gradient_fields = {
            "RHF adjoint response gradient": correction_gradient_response,
        }
        if not compact:
            correction_gradient_occupied_virtual = (
                correction_gradient_adjoint_nuclear
                + correction_gradient_adjoint_metric
            )
            gradient_fields.update(
                {
                    "RHF objective metric gradient": correction_gradient_metric,
                    "RHF adjoint nuclear gradient": correction_gradient_adjoint_nuclear,
                    "RHF adjoint metric gradient": correction_gradient_adjoint_metric,
                    "RHF occupied-virtual gradient": correction_gradient_occupied_virtual,
                }
            )
        for name, value in gradient_fields.items():
            _validated_float64_array(
                value,
                (len(atom_indices), 3),
                name,
            )
        validate_reference(self.reference)
        state_fingerprint = reference_fingerprint(self.reference)
        linear_diagnostics = linear_result.diagnostics
        diagnostics = RHFAdjointDiagnostics(
            minimum_orbital_gap=minimum_gap,
            pyscf_version=pyscf.__version__,
            residual_tolerance=self.residual_tolerance,
            orbital_gap_tolerance=self.orbital_gap_tolerance,
            response_dimension=response_dimension,
            operator_is_self_adjoint=True,
            objective_symmetry_tolerance=self.objective_symmetry_tolerance,
            objective_symmetry_residual=objective_symmetry_residual,
            adjoint_density_symmetry_residual=(
                adjoint_density_symmetry_residual
            ),
            adjoint_potential_symmetry_residual=(
                adjoint_potential_symmetry_residual
            ),
            solver=linear_diagnostics.solver,
            solve_count=linear_diagnostics.solve_count,
            objective_gradient_norm=(
                linear_diagnostics.objective_gradient_norm
            ),
            solution_norm=linear_diagnostics.solution_norm,
            maximum_residual=linear_diagnostics.maximum_residual,
            residual_rms=linear_diagnostics.residual_rms,
            max_cycle=self.max_cycle,
            krylov_restart=self.krylov_restart,
            iteration_count=linear_diagnostics.iteration_count,
        )
        if compact:
            return diagnostics, correction_gradient_response
        adjoint = RHFAdjoint(
            reference_identity=id(self.reference),
            state_fingerprint=state_fingerprint,
            integrity_fingerprint="",
            operator_fingerprint=linear_result.operator_fingerprint,
            atom_indices=atom_indices,
            objective_ao_potential=_immutable_array(objective_ao_potential),
            objective_orbital_gradient=_immutable_array(
                objective_orbital_gradient
            ),
            zvector=_immutable_array(zvector),
            residual=_immutable_array(
                linear_result.residual.reshape(nvir, nocc)
            ),
            adjoint_ao_density=_immutable_array(adjoint_ao_density),
            adjoint_ao_potential=_immutable_array(adjoint_ao_potential),
            correction_gradient_metric=_immutable_array(
                correction_gradient_metric
            ),
            correction_gradient_adjoint_nuclear=_immutable_array(
                correction_gradient_adjoint_nuclear
            ),
            correction_gradient_adjoint_metric=_immutable_array(
                correction_gradient_adjoint_metric
            ),
            correction_gradient_occupied_virtual=_immutable_array(
                correction_gradient_occupied_virtual
            ),
            correction_gradient_response=_immutable_array(
                correction_gradient_response
            ),
            diagnostics=diagnostics,
        )
        return replace(
            adjoint,
            integrity_fingerprint=adjoint_integrity_fingerprint(adjoint),
        )
