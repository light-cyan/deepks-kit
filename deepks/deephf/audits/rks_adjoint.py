"""Bounded dense audits separated from production solver assembly."""

from __future__ import annotations

from ..pyscf_dft_provenance import RKSAdjoint
from ..pyscf_dft_provenance import RKSAdjointDiagnostics
from ..pyscf_dft_provenance import RKSAdjointError
from ..pyscf_dft_provenance import RKSFunctionalProvenance
from ..pyscf_dft_provenance import RKSGridProvenance
from numbers import Real
from ..pyscf_rks_response_core import _RKSLinearResponseProblem
from ..pyscf_dft_provenance import _functional_provenance
from ..pyscf_dft_provenance import _grid_provenance
from ..pyscf_dft_provenance import _validated_float64_array
from pyscf.dft import libxc
import numpy as np
import pyscf
from ..pyscf_rks_reference import rks_adjoint_integrity_fingerprint
from ..pyscf_rks_reference import rks_reference_fingerprint
from ..adjoint import scalar_operator_fingerprint
from ..pyscf_rks_reference import validate_rks_reference


def _audit_adjoint(
    self,
    adjoint: RKSAdjoint,
    expected_objective_ao_potential: np.ndarray,
) -> None:
    validate_rks_reference(self.reference)
    if type(adjoint) is not RKSAdjoint:
        raise RKSAdjointError("the supplied RKS adjoint has an invalid type")
    diagnostics = adjoint.diagnostics
    if type(diagnostics) is not RKSAdjointDiagnostics:
        raise RKSAdjointError(
            "the supplied RKS adjoint diagnostics have an invalid type"
        )
    current_fingerprint = rks_reference_fingerprint(self.reference)
    if adjoint.reference_identity != id(self.reference):
        raise RKSAdjointError(
            "the supplied RKS adjoint belongs to another reference"
        )
    if adjoint.state_fingerprint != current_fingerprint:
        raise RKSAdjointError(
            "the supplied RKS adjoint does not match the current RKS state"
        )
    if adjoint.integrity_fingerprint != rks_adjoint_integrity_fingerprint(
        adjoint
    ):
        raise RKSAdjointError(
            "the supplied RKS adjoint failed its integrity check"
        )
    provenance_values = (
        adjoint.reference_identity,
        adjoint.state_fingerprint,
        adjoint.integrity_fingerprint,
        adjoint.operator_fingerprint,
    )
    if (
        type(provenance_values[0]) is not int
        or any(type(value) is not str for value in provenance_values[1:])
    ):
        raise RKSAdjointError(
            "the supplied RKS adjoint provenance fields have invalid types"
        )
    functional_provenance = _functional_provenance(self.reference)
    grid_provenance = _grid_provenance(self.reference)
    if (
        type(adjoint.functional_provenance) is not RKSFunctionalProvenance
        or adjoint.functional_provenance != functional_provenance
    ):
        raise RKSAdjointError(
            "the supplied RKS adjoint functional provenance is invalid"
        )
    if (
        type(adjoint.grid_provenance) is not RKSGridProvenance
        or adjoint.grid_provenance != grid_provenance
    ):
        raise RKSAdjointError(
            "the supplied RKS adjoint grid provenance is invalid"
        )
    if diagnostics.solver != "scipy.sparse.linalg.gmres(A.T, b)":
        raise RKSAdjointError(
            "the supplied RKS adjoint solver convention is invalid"
        )
    if type(diagnostics.solve_count) is not int or diagnostics.solve_count != 1:
        raise RKSAdjointError(
            "the supplied RKS adjoint must contain exactly one scalar solve"
        )
    integer_diagnostics = (
        diagnostics.grid_point_count,
        diagnostics.response_dimension,
        diagnostics.max_cycle,
        diagnostics.krylov_restart,
    )
    if any(type(value) is not int or value <= 0 for value in integer_diagnostics):
        raise RKSAdjointError(
            "the supplied RKS adjoint integer diagnostics are invalid"
        )
    if (
        type(diagnostics.iteration_count) is not int
        or diagnostics.iteration_count < 0
    ):
        raise RKSAdjointError(
            "the supplied RKS adjoint iteration count is invalid"
        )
    if diagnostics.operator_is_self_adjoint is not True:
        raise RKSAdjointError("the supplied RKS adjoint operator contract is invalid")
    diagnostic_reals = (
        diagnostics.minimum_orbital_gap,
        diagnostics.residual_tolerance,
        diagnostics.invariant_tolerance,
        diagnostics.orbital_gap_tolerance,
        diagnostics.hamiltonian_reconstruction_residual,
        diagnostics.objective_symmetry_tolerance,
        diagnostics.objective_symmetry_residual,
        diagnostics.adjoint_density_symmetry_residual,
        diagnostics.adjoint_potential_symmetry_residual,
        diagnostics.objective_gradient_norm,
        diagnostics.solution_norm,
        diagnostics.maximum_residual,
        diagnostics.residual_rms,
    )
    if any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
        for value in diagnostic_reals
    ) or not np.isfinite(diagnostic_reals).all():
        raise RKSAdjointError(
            "the supplied RKS adjoint diagnostics must be finite real scalars"
        )
    controls = {
        "residual_tolerance": self.residual_tolerance,
        "invariant_tolerance": self.invariant_tolerance,
        "orbital_gap_tolerance": self.orbital_gap_tolerance,
        "objective_symmetry_tolerance": self.objective_symmetry_tolerance,
        "max_cycle": self.max_cycle,
        "krylov_restart": self.krylov_restart,
    }
    for name, expected in controls.items():
        if getattr(diagnostics, name) != expected:
            raise RKSAdjointError(
                f"the supplied RKS adjoint {name} control is inconsistent"
            )
    if (
        diagnostics.residual_tolerance <= 0.0
        or diagnostics.invariant_tolerance <= 0.0
        or diagnostics.orbital_gap_tolerance <= 0.0
        or diagnostics.objective_symmetry_tolerance <= 0.0
    ):
        raise RKSAdjointError(
            "the supplied RKS adjoint controls are invalid"
        )
    exact_diagnostics = {
        "pyscf_version": pyscf.__version__,
        "libxc_version": str(libxc.__version__),
        "functional_components": functional_provenance.components,
        "grid_point_count": grid_provenance.point_count,
        "grid_coordinates_fingerprint": (
            grid_provenance.coordinates_fingerprint
        ),
        "grid_weights_fingerprint": grid_provenance.weights_fingerprint,
    }
    for name, expected in exact_diagnostics.items():
        if getattr(diagnostics, name) != expected:
            raise RKSAdjointError(
                f"the supplied RKS adjoint diagnostic {name} is invalid"
            )
    (
        coefficient,
        energy,
        occupation,
        occupied,
        virtual,
        minimum_gap,
    ) = self._state()
    nocc = int(np.count_nonzero(occupied))
    nvir = int(np.count_nonzero(virtual))
    dimension = nocc * nvir
    atom_indices = self._response_atom_indices(adjoint.atom_indices)
    if atom_indices != adjoint.atom_indices:
        raise RKSAdjointError("the supplied RKS adjoint atom selection is invalid")
    natm = len(atom_indices)
    nao = int(self.molecule.nao)
    array_shapes = {
        "objective_ao_potential": (nao, nao),
        "objective_orbital_gradient": (nvir, nocc),
        "zvector": (nvir, nocc),
        "residual": (nvir, nocc),
        "adjoint_ao_density": (nao, nao),
        "adjoint_ao_potential": (nao, nao),
        "correction_gradient_metric": (natm, 3),
        "correction_gradient_adjoint_fixed_grid": (natm, 3),
        "correction_gradient_adjoint_grid_coordinate": (natm, 3),
        "correction_gradient_adjoint_grid_weight": (natm, 3),
        "correction_gradient_adjoint_nuclear": (natm, 3),
        "correction_gradient_adjoint_metric": (natm, 3),
        "correction_gradient_occupied_virtual": (natm, 3),
        "correction_gradient_response": (natm, 3),
    }
    for name, shape in array_shapes.items():
        self._audited_array(getattr(adjoint, name), shape, name)
    expected_objective_ao_potential = self._validated_objective_potential(
        expected_objective_ao_potential
    )
    self._require_close(
        adjoint.objective_ao_potential,
        expected_objective_ao_potential,
        "objective AO potential",
    )
    expected_objective_gradient = self._expected_objective_gradient(
        expected_objective_ao_potential,
        coefficient,
        occupation,
        occupied,
        virtual,
    )
    self._require_close(
        adjoint.objective_orbital_gradient,
        expected_objective_gradient,
        "bilateral occupied-virtual objective gradient",
    )
    response_dimension = dimension
    problem = _RKSLinearResponseProblem(self)
    expected_operator_fingerprint = scalar_operator_fingerprint(
        problem,
        solver="gmres",
    )
    if adjoint.operator_fingerprint != expected_operator_fingerprint:
        raise RKSAdjointError(
            "the supplied RKS adjoint response operator is inconsistent"
        )
    objective_vector = expected_objective_gradient.reshape(dimension)
    zvector = adjoint.zvector
    residual = (
        self._apply_occupied_virtual_operator(
            zvector,
            coefficient,
            energy,
            occupation,
            occupied,
            virtual,
        ).reshape(dimension)
        - objective_vector
    ).reshape(nvir, nocc)
    self._require_close(
        adjoint.residual,
        residual,
        "independent residual",
    )
    expected_adjoint_density = self._adjoint_density(
        zvector,
        coefficient,
        occupation,
        occupied,
        virtual,
    )
    self._require_close(
        adjoint.adjoint_ao_density,
        expected_adjoint_density,
        "AO density",
    )
    expected_adjoint_potential = _validated_float64_array(
        self._induced_potential(expected_adjoint_density),
        (nao, nao),
        "independently rebuilt RKS adjoint AO potential",
    )
    self._require_close(
        adjoint.adjoint_ao_potential,
        expected_adjoint_potential,
        "AO potential",
    )
    (
        expected_partitions,
        hamiltonian_reconstruction_residual,
    ) = self._gradient_partitions(
        expected_objective_ao_potential,
        zvector,
        expected_adjoint_potential,
        coefficient,
        energy,
        occupation,
        occupied,
        virtual,
        atom_indices,
    )
    for name, expected in expected_partitions.items():
        self._require_close(getattr(adjoint, name), expected, name)
    self._require_close(
        adjoint.correction_gradient_adjoint_nuclear,
        adjoint.correction_gradient_adjoint_fixed_grid
        + adjoint.correction_gradient_adjoint_grid_coordinate
        + adjoint.correction_gradient_adjoint_grid_weight,
        "nuclear gradient partition",
    )
    self._require_close(
        adjoint.correction_gradient_occupied_virtual,
        adjoint.correction_gradient_adjoint_nuclear
        + adjoint.correction_gradient_adjoint_metric,
        "occupied-virtual gradient partition",
    )
    self._require_close(
        adjoint.correction_gradient_response,
        adjoint.correction_gradient_metric
        + adjoint.correction_gradient_occupied_virtual,
        "response gradient partition",
    )
    maximum_residual, residual_rms = self._residual_statistics(residual)
    objective_symmetry_residual = float(
        np.max(
            np.abs(
                expected_objective_ao_potential
                - expected_objective_ao_potential.T
            ),
            initial=0.0,
        )
    )
    density_symmetry_residual = float(
        np.max(
            np.abs(expected_adjoint_density - expected_adjoint_density.T),
            initial=0.0,
        )
    )
    potential_symmetry_residual = float(
        np.max(
            np.abs(
                expected_adjoint_potential
                - expected_adjoint_potential.T
            ),
            initial=0.0,
        )
    )
    measured = {
        "minimum_orbital_gap": minimum_gap,
        "response_dimension": response_dimension,
        "hamiltonian_reconstruction_residual": (
            hamiltonian_reconstruction_residual
        ),
        "objective_symmetry_residual": objective_symmetry_residual,
        "adjoint_density_symmetry_residual": density_symmetry_residual,
        "adjoint_potential_symmetry_residual": potential_symmetry_residual,
        "objective_gradient_norm": float(
            np.linalg.norm(expected_objective_gradient)
        ),
        "solution_norm": float(np.linalg.norm(zvector)),
        "maximum_residual": maximum_residual,
        "residual_rms": residual_rms,
    }
    for name, expected in measured.items():
        stored = getattr(diagnostics, name)
        if isinstance(expected, int):
            matches = stored == expected
        else:
            matches = np.isclose(
                stored,
                expected,
                rtol=1.0e-10,
                atol=1.0e-12,
            )
        if not matches:
            raise RKSAdjointError(
                f"the supplied RKS adjoint {name} diagnostic is inconsistent"
            )
    if (
        maximum_residual > diagnostics.residual_tolerance
        or minimum_gap <= diagnostics.orbital_gap_tolerance
        or hamiltonian_reconstruction_residual
        > diagnostics.invariant_tolerance
        or objective_symmetry_residual
        > diagnostics.objective_symmetry_tolerance
        or density_symmetry_residual
        > diagnostics.objective_symmetry_tolerance
        or potential_symmetry_residual
        > diagnostics.objective_symmetry_tolerance
    ):
        raise RKSAdjointError(
            "the supplied RKS adjoint exceeds an accepted control"
        )
    validate_rks_reference(self.reference)
    if rks_reference_fingerprint(self.reference) != current_fingerprint:
        raise RKSAdjointError(
            "the RKS reference changed during the scalar-adjoint audit"
        )


__all__ = ['_audit_adjoint']
