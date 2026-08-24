"""Internal implementation extracted from pyscf_rks.py."""

from dataclasses import replace
import numpy as np
import pyscf
from pyscf.dft import libxc
from .adjoint import AdjointError, ScalarAdjointProblem, solve_scalar_adjoint
from .capabilities import DeePHFCapabilityError
from .contracts import immutable_array as _immutable_array
from .pyscf_dft_provenance import (
    RKSAdjoint,
    RKSAdjointDiagnostics,
    RKSAdjointError,
    RKSResponseError,
    _cycle_limit,
    _functional_provenance,
    _grid_provenance,
    _response_real_control,
    _validated_float64_array,
)
from .pyscf_rks_reference import (
    rks_adjoint_integrity_fingerprint,
    rks_reference_fingerprint,
    validate_rks_reference,
)
from .pyscf_rks_response_core import (
    _RKSLinearResponseProblem,
    _RKSLinearResponseCore,
)

class RKSAdjointAdapter(_RKSLinearResponseCore):
    """Solve one correction-specific pure-LDA RKS scalar adjoint."""

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
        super().__init__(
            reference,
            residual_tolerance=residual_tolerance,
            invariant_tolerance=invariant_tolerance,
            orbital_gap_tolerance=orbital_gap_tolerance,
            operator_stability_tolerance=operator_stability_tolerance,
            operator_condition_tolerance=operator_condition_tolerance,
            operator_symmetry_tolerance=operator_symmetry_tolerance,
            operator_dimension_limit=operator_dimension_limit,
            max_cycle=max_cycle,
        )
        self.krylov_restart = _cycle_limit(
            krylov_restart,
            "krylov_restart",
        )
        self.objective_symmetry_tolerance = _response_real_control(
            objective_symmetry_tolerance,
            "objective_symmetry_tolerance",
        )
        if self.objective_symmetry_tolerance <= 0.0:
            raise ValueError("adjoint objective_symmetry_tolerance must be positive")
        if self.krylov_restart <= 0:
            raise ValueError("adjoint krylov_restart must be positive")

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
            raise RKSAdjointError(
                "the correction AO objective potential violates symmetry: "
                f"{symmetry_residual:.3e} > "
                f"{self.objective_symmetry_tolerance:.3e}"
            )
        return potential

    @staticmethod
    def _audited_array(value, expected_shape, name: str) -> np.ndarray:
        if type(value) is not np.ndarray:
            raise RKSAdjointError(
                f"the supplied RKS adjoint field {name} has an invalid type"
            )
        if value.shape != expected_shape:
            raise RKSAdjointError(
                f"the supplied RKS adjoint field {name} has shape {value.shape}; "
                f"expected {expected_shape}"
            )
        if value.dtype != np.dtype(np.float64) or np.iscomplexobj(value):
            raise RKSAdjointError(
                f"the supplied RKS adjoint field {name} must use real numpy.float64"
            )
        if not np.isfinite(value).all():
            raise RKSAdjointError(
                f"the supplied RKS adjoint field {name} must be finite"
            )
        if value.flags.writeable:
            raise RKSAdjointError(
                f"the supplied RKS adjoint field {name} must be immutable"
            )
        return value

    @staticmethod
    def _require_close(stored, expected, name: str) -> None:
        if not np.allclose(stored, expected, rtol=1.0e-11, atol=1.0e-12):
            maximum_residual = float(
                np.max(np.abs(np.asarray(stored) - np.asarray(expected)), initial=0.0)
            )
            raise RKSAdjointError(
                f"the supplied RKS adjoint {name} is inconsistent: "
                f"residual {maximum_residual:.3e}"
            )

    @staticmethod
    def _residual_statistics(value: np.ndarray) -> tuple[float, float]:
        return (
            float(np.max(np.abs(value), initial=0.0)),
            float(np.sqrt(np.mean(np.square(value)))),
        )

    def _adjoint_density(
        self,
        zvector: np.ndarray,
        coefficient: np.ndarray,
        occupation: np.ndarray,
        occupied: np.ndarray,
        virtual: np.ndarray,
    ) -> np.ndarray:
        occupied_coefficients = coefficient[:, occupied]
        virtual_coefficients = coefficient[:, virtual]
        rotated_occupied = virtual_coefficients @ zvector
        one_sided = rotated_occupied @ (
            occupied_coefficients * occupation[occupied]
        ).T
        density = one_sided + one_sided.T
        return _validated_float64_array(
            density,
            (self.molecule.nao, self.molecule.nao),
            "RKS adjoint AO density",
        )

    def _gradient_partitions(
        self,
        objective_ao_potential: np.ndarray,
        zvector: np.ndarray,
        adjoint_ao_potential: np.ndarray,
        coefficient: np.ndarray,
        energy: np.ndarray,
        occupation: np.ndarray,
        occupied: np.ndarray,
        virtual: np.ndarray,
        atom_indices=None,
        compact=False,
    ):
        atom_indices = self._response_atom_indices(atom_indices)
        overlap_derivative = self._overlap_derivative(atom_indices)
        (
            hamiltonian_derivative,
            hamiltonian_fixed_grid,
            hamiltonian_grid_coordinate,
            hamiltonian_grid_weight,
        ) = self._hamiltonian_derivative(coefficient, occupation, atom_indices)
        hamiltonian_reconstruction_residual = float(
            np.max(
                np.abs(
                    hamiltonian_derivative
                    - hamiltonian_fixed_grid
                    - hamiltonian_grid_coordinate
                    - hamiltonian_grid_weight
                ),
                initial=0.0,
            )
        )
        occupied_coefficients = coefficient[:, occupied]

        def occupied_mo(value):
            return np.einsum(
                "mp,...mn,ni->...pi",
                coefficient,
                value,
                occupied_coefficients,
            )

        overlap_mo = occupied_mo(overlap_derivative)
        if compact:
            full_rhs = (
                occupied_mo(hamiltonian_derivative)[..., virtual, :]
                - overlap_mo[..., virtual, :] * energy[occupied]
            )
            correction_gradient_adjoint_nuclear = -np.einsum(
                "ai,...ai->...", zvector, full_rhs
            )
        else:
            fixed_grid_mo = occupied_mo(hamiltonian_fixed_grid)
            grid_coordinate_mo = occupied_mo(hamiltonian_grid_coordinate)
            grid_weight_mo = occupied_mo(hamiltonian_grid_weight)
            fixed_grid_rhs = (
                fixed_grid_mo[..., virtual, :]
                - overlap_mo[..., virtual, :] * energy[occupied]
            )
            correction_gradient_adjoint_fixed_grid = -np.einsum(
                "ai,...ai->...", zvector, fixed_grid_rhs
            )
            correction_gradient_adjoint_grid_coordinate = -np.einsum(
                "ai,...ai->...", zvector, grid_coordinate_mo[..., virtual, :]
            )
            correction_gradient_adjoint_grid_weight = -np.einsum(
                "ai,...ai->...", zvector, grid_weight_mo[..., virtual, :]
            )
            correction_gradient_adjoint_nuclear = (
                correction_gradient_adjoint_fixed_grid
                + correction_gradient_adjoint_grid_coordinate
                + correction_gradient_adjoint_grid_weight
            )
        objective_mo = coefficient.T @ objective_ao_potential @ coefficient
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
            adjoint_potential_occupied
            + adjoint_potential_occupied.T
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
        if compact:
            _validated_float64_array(
                correction_gradient_response,
                (len(atom_indices), 3),
                "RKS correction_gradient_response",
            )
            return correction_gradient_response, hamiltonian_reconstruction_residual
        correction_gradient_occupied_virtual = (
            correction_gradient_adjoint_nuclear
            + correction_gradient_adjoint_metric
        )
        partitions = {
            "correction_gradient_metric": correction_gradient_metric,
            "correction_gradient_adjoint_fixed_grid": (
                correction_gradient_adjoint_fixed_grid
            ),
            "correction_gradient_adjoint_grid_coordinate": (
                correction_gradient_adjoint_grid_coordinate
            ),
            "correction_gradient_adjoint_grid_weight": (
                correction_gradient_adjoint_grid_weight
            ),
            "correction_gradient_adjoint_nuclear": (
                correction_gradient_adjoint_nuclear
            ),
            "correction_gradient_adjoint_metric": (
                correction_gradient_adjoint_metric
            ),
            "correction_gradient_occupied_virtual": (
                correction_gradient_occupied_virtual
            ),
            "correction_gradient_response": correction_gradient_response,
        }
        expected_shape = (len(atom_indices), 3)
        for name, value in partitions.items():
            _validated_float64_array(value, expected_shape, f"RKS {name}")
        return (
            partitions,
            hamiltonian_reconstruction_residual,
        )

    def _expected_objective_gradient(
        self,
        objective_ao_potential: np.ndarray,
        coefficient: np.ndarray,
        occupation: np.ndarray,
        occupied: np.ndarray,
        virtual: np.ndarray,
    ) -> np.ndarray:
        objective_mo = coefficient.T @ objective_ao_potential @ coefficient
        objective_gradient = (
            objective_mo[virtual][:, occupied]
            + objective_mo.T[virtual][:, occupied]
        ) * occupation[occupied]
        return _validated_float64_array(
            objective_gradient,
            (
                int(np.count_nonzero(virtual)),
                int(np.count_nonzero(occupied)),
            ),
            "correction occupied-virtual objective gradient",
        )

    def solve(self, objective_ao_potential: np.ndarray, atom_indices=None, compact=False):
        """Return one audited RKS Z-vector and selected nuclear contractions."""
        try:
            return self._solve(objective_ao_potential, atom_indices, compact=compact)
        except DeePHFCapabilityError:
            raise
        except RKSAdjointError:
            raise
        except (AdjointError, RKSResponseError) as error:
            raise RKSAdjointError(
                f"RKS adjoint evaluation failed: {error}"
            ) from error

    def _solve(self, objective_ao_potential: np.ndarray, atom_indices=None, compact=False):
        validate_rks_reference(self.reference)
        atom_indices = self._response_atom_indices(atom_indices)
        initial_fingerprint = rks_reference_fingerprint(self.reference)
        functional_provenance = _functional_provenance(self.reference)
        grid_provenance = _grid_provenance(self.reference)
        objective_ao_potential = self._validated_objective_potential(
            objective_ao_potential
        )
        objective_symmetry_residual = float(
            np.max(
                np.abs(objective_ao_potential - objective_ao_potential.T),
                initial=0.0,
            )
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
        objective_orbital_gradient = self._expected_objective_gradient(
            objective_ao_potential,
            coefficient,
            occupation,
            occupied,
            virtual,
        )
        problem = _RKSLinearResponseProblem(self)
        if not isinstance(problem, ScalarAdjointProblem):
            raise RKSAdjointError(
                "the RKS adjoint operator violates the neutral scalar protocol"
            )
        linear_result = solve_scalar_adjoint(
            problem,
            objective_orbital_gradient.reshape(response_dimension),
            residual_tolerance=self.residual_tolerance,
            require_physical_residual=True,
            solver="gmres",
            max_cycle=self.max_cycle,
            restart=self.krylov_restart,
        )
        nocc = int(np.count_nonzero(occupied))
        nvir = int(np.count_nonzero(virtual))
        zvector = linear_result.solution.reshape(nvir, nocc)
        adjoint_ao_density = self._adjoint_density(
            zvector,
            coefficient,
            occupation,
            occupied,
            virtual,
        )
        adjoint_density_symmetry_residual = float(
            np.max(
                np.abs(adjoint_ao_density - adjoint_ao_density.T),
                initial=0.0,
            )
        )
        if adjoint_density_symmetry_residual > self.objective_symmetry_tolerance:
            raise RKSAdjointError(
                "the RKS adjoint AO density violates symmetry: "
                f"{adjoint_density_symmetry_residual:.3e} > "
                f"{self.objective_symmetry_tolerance:.3e}"
            )
        adjoint_ao_potential = _validated_float64_array(
            self._induced_potential(adjoint_ao_density),
            (self.molecule.nao, self.molecule.nao),
            "RKS adjoint AO potential",
        )
        adjoint_potential_symmetry_residual = float(
            np.max(
                np.abs(adjoint_ao_potential - adjoint_ao_potential.T),
                initial=0.0,
            )
        )
        if adjoint_potential_symmetry_residual > self.objective_symmetry_tolerance:
            raise RKSAdjointError(
                "the RKS adjoint AO potential violates symmetry: "
                f"{adjoint_potential_symmetry_residual:.3e} > "
                f"{self.objective_symmetry_tolerance:.3e}"
            )
        (
            gradient_data,
            hamiltonian_reconstruction_residual,
        ) = self._gradient_partitions(
            objective_ao_potential,
            zvector,
            adjoint_ao_potential,
            coefficient,
            energy,
            occupation,
            occupied,
            virtual,
            atom_indices,
            compact=compact,
        )
        if compact:
            correction_gradient_response = gradient_data
        else:
            partitions = gradient_data
            self._require_close(
                partitions["correction_gradient_adjoint_nuclear"],
                partitions["correction_gradient_adjoint_fixed_grid"]
                + partitions["correction_gradient_adjoint_grid_coordinate"]
                + partitions["correction_gradient_adjoint_grid_weight"],
                "nuclear gradient partition",
            )
            self._require_close(
                partitions["correction_gradient_occupied_virtual"],
                partitions["correction_gradient_adjoint_nuclear"]
                + partitions["correction_gradient_adjoint_metric"],
                "occupied-virtual gradient partition",
            )
            self._require_close(
                partitions["correction_gradient_response"],
                partitions["correction_gradient_metric"]
                + partitions["correction_gradient_occupied_virtual"],
                "response gradient partition",
            )
        if hamiltonian_reconstruction_residual > self.invariant_tolerance:
            raise RKSAdjointError(
                "the RKS adjoint Hamiltonian reconstruction exceeds tolerance"
            )
        validate_rks_reference(self.reference)
        if rks_reference_fingerprint(self.reference) != initial_fingerprint:
            raise RKSAdjointError(
                "the RKS reference changed during the scalar-adjoint evaluation"
            )
        linear_diagnostics = linear_result.diagnostics
        diagnostics = RKSAdjointDiagnostics(
            minimum_orbital_gap=minimum_gap,
            pyscf_version=pyscf.__version__,
            libxc_version=str(libxc.__version__),
            functional_components=functional_provenance.components,
            grid_point_count=grid_provenance.point_count,
            grid_coordinates_fingerprint=grid_provenance.coordinates_fingerprint,
            grid_weights_fingerprint=grid_provenance.weights_fingerprint,
            residual_tolerance=self.residual_tolerance,
            invariant_tolerance=self.invariant_tolerance,
            orbital_gap_tolerance=self.orbital_gap_tolerance,
            response_dimension=response_dimension,
            operator_is_self_adjoint=True,
            hamiltonian_reconstruction_residual=(
                hamiltonian_reconstruction_residual
            ),
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
        adjoint = RKSAdjoint(
            reference_identity=id(self.reference),
            state_fingerprint=initial_fingerprint,
            integrity_fingerprint="",
            operator_fingerprint=linear_result.operator_fingerprint,
            atom_indices=atom_indices,
            functional_provenance=functional_provenance,
            grid_provenance=grid_provenance,
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
            **{
                name: _immutable_array(value)
                for name, value in partitions.items()
            },
            diagnostics=diagnostics,
        )
        return replace(
            adjoint,
            integrity_fingerprint=rks_adjoint_integrity_fingerprint(adjoint),
        )

    def audit_adjoint(
        self,
        adjoint: RKSAdjoint,
        expected_objective_ao_potential: np.ndarray,
    ) -> None:
        """Independently audit one consumed RKS adjoint without another solve."""
        try:
            self._audit_adjoint(adjoint, expected_objective_ao_potential)
        except DeePHFCapabilityError:
            raise
        except RKSAdjointError:
            raise
        except (AdjointError, RKSResponseError) as error:
            raise RKSAdjointError(
                f"RKS adjoint audit failed: {error}"
            ) from error

    def _audit_adjoint(
        self,
        adjoint: RKSAdjoint,
        expected_objective_ao_potential: np.ndarray,
    ) -> None:
        from .audits.rks_adjoint import _audit_adjoint as audit
        return audit(self, adjoint, expected_objective_ao_potential)
