"""Bounded dense audits separated from production solver assembly."""

from __future__ import annotations

from ..pyscf_rhf_reference import RHFAdjoint
from ..pyscf_rhf_reference import RHFAdjointDiagnostics
from ..pyscf_rhf_reference import RHFAdjointError
from numbers import Real
from ..pyscf_rhf_scanner import _validated_float64_array
from ..pyscf_rhf_scanner import adjoint_integrity_fingerprint
import numpy as np
import pyscf
from ..pyscf_rhf_reference import reference_fingerprint
from ..adjoint import scalar_operator_fingerprint
from ..pyscf_rhf_reference import validate_reference
from ..pyscf_rhf_adjoint import (
    _RHFScalarAdjointProblem,
)


def audit_adjoint(
    self,
    adjoint: RHFAdjoint,
    expected_objective_ao_potential: np.ndarray,
) -> None:
    """Independently audit one consumed RHF adjoint without another solve."""
    validate_reference(self.reference)
    if type(adjoint) is not RHFAdjoint:
        raise RHFAdjointError("the supplied RHF adjoint has an invalid type")
    diagnostics = adjoint.diagnostics
    if type(diagnostics) is not RHFAdjointDiagnostics:
        raise RHFAdjointError(
            "the supplied RHF adjoint diagnostics have an invalid type"
        )
    if adjoint.reference_identity != id(self.reference):
        raise RHFAdjointError(
            "the supplied RHF adjoint belongs to another reference"
        )
    if adjoint.state_fingerprint != reference_fingerprint(self.reference):
        raise RHFAdjointError(
            "the supplied RHF adjoint does not match the current RHF state"
        )
    if adjoint.integrity_fingerprint != adjoint_integrity_fingerprint(adjoint):
        raise RHFAdjointError(
            "the supplied RHF adjoint failed its integrity check"
        )
    if (
        type(adjoint.reference_identity) is not int
        or type(adjoint.state_fingerprint) is not str
        or type(adjoint.integrity_fingerprint) is not str
        or type(adjoint.operator_fingerprint) is not str
    ):
        raise RHFAdjointError(
            "the supplied RHF adjoint provenance fields have invalid types"
        )
    if diagnostics.pyscf_version != pyscf.__version__:
        raise RHFAdjointError(
            "the supplied RHF adjoint PySCF version does not match the runtime"
        )
    if diagnostics.solver != "scipy.sparse.linalg.gmres(A.T, b)":
        raise RHFAdjointError(
            "the supplied RHF adjoint solver convention is invalid"
        )
    if type(diagnostics.solve_count) is not int or diagnostics.solve_count != 1:
        raise RHFAdjointError(
            "the supplied RHF adjoint must contain exactly one scalar solve"
        )
    if type(diagnostics.response_dimension) is not int:
        raise RHFAdjointError(
            "the supplied RHF adjoint response dimension has an invalid type"
        )
    if diagnostics.operator_is_self_adjoint is not True:
        raise RHFAdjointError("the supplied RHF adjoint operator contract is invalid")
    if (
        type(diagnostics.max_cycle) is not int
        or type(diagnostics.krylov_restart) is not int
        or type(diagnostics.iteration_count) is not int
    ):
        raise RHFAdjointError(
            "the supplied RHF adjoint Krylov diagnostics have invalid types"
        )
    diagnostic_reals = (
        diagnostics.minimum_orbital_gap,
        diagnostics.residual_tolerance,
        diagnostics.orbital_gap_tolerance,
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
        raise RHFAdjointError(
            "the supplied RHF adjoint diagnostics must be finite real scalars"
        )
    if (
        diagnostics.residual_tolerance <= 0
        or diagnostics.orbital_gap_tolerance <= 0
        or diagnostics.objective_symmetry_tolerance <= 0
        or diagnostics.response_dimension <= 0
        or diagnostics.max_cycle <= 0
        or diagnostics.krylov_restart <= 0
        or diagnostics.iteration_count < 0
    ):
        raise RHFAdjointError(
            "the supplied RHF adjoint controls are invalid"
        )
    accepted_controls = {
        "residual_tolerance": self.residual_tolerance,
        "orbital_gap_tolerance": self.orbital_gap_tolerance,
        "objective_symmetry_tolerance": (
            self.objective_symmetry_tolerance
        ),
        "max_cycle": self.max_cycle,
        "krylov_restart": self.krylov_restart,
    }
    for name, expected in accepted_controls.items():
        if getattr(diagnostics, name) != expected:
            raise RHFAdjointError(
                f"the supplied RHF adjoint {name} control is inconsistent"
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
        raise RHFAdjointError("the supplied RHF adjoint atom selection is invalid")
    natm = len(atom_indices)
    nao = int(self.molecule.nao)
    arrays = {
        "objective_ao_potential": (nao, nao),
        "objective_orbital_gradient": (nvir, nocc),
        "zvector": (nvir, nocc),
        "residual": (nvir, nocc),
        "adjoint_ao_density": (nao, nao),
        "adjoint_ao_potential": (nao, nao),
        "correction_gradient_metric": (natm, 3),
        "correction_gradient_adjoint_nuclear": (natm, 3),
        "correction_gradient_adjoint_metric": (natm, 3),
        "correction_gradient_occupied_virtual": (natm, 3),
        "correction_gradient_response": (natm, 3),
    }
    for name, shape in arrays.items():
        self._audited_array(getattr(adjoint, name), shape, name)
    expected_objective_ao_potential = _validated_float64_array(
        expected_objective_ao_potential,
        (nao, nao),
        "expected correction AO objective potential",
    )
    self._require_close(
        adjoint.objective_ao_potential,
        expected_objective_ao_potential,
        "objective AO potential",
    )
    objective_mo = (
        coefficient.T @ expected_objective_ao_potential @ coefficient
    )
    expected_objective_gradient = (
        objective_mo[virtual][:, occupied]
        + objective_mo.T[virtual][:, occupied]
    ) * occupation[occupied]
    self._require_close(
        adjoint.objective_orbital_gradient,
        expected_objective_gradient,
        "bilateral occupied-virtual objective gradient",
    )
    response_dimension = dimension
    problem = _RHFScalarAdjointProblem(
        self,
        coefficient,
        energy,
        occupation,
        occupied,
        virtual,
    )
    expected_operator_fingerprint = scalar_operator_fingerprint(
        problem,
        solver="gmres",
    )
    if adjoint.operator_fingerprint != expected_operator_fingerprint:
        raise RHFAdjointError(
            "the supplied RHF adjoint response operator is inconsistent"
        )
    zvector = adjoint.zvector
    objective_vector = expected_objective_gradient.reshape(dimension)
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
    occupied_coefficients = coefficient[:, occupied]
    virtual_coefficients = coefficient[:, virtual]
    rotated_occupied = virtual_coefficients @ zvector
    expected_adjoint_density = (
        rotated_occupied
        @ (occupied_coefficients * occupation[occupied]).T
    )
    expected_adjoint_density = (
        expected_adjoint_density + expected_adjoint_density.T
    )
    self._require_close(
        adjoint.adjoint_ao_density,
        expected_adjoint_density,
        "AO density",
    )
    expected_adjoint_potential = self._induced_potential(
        expected_adjoint_density
    )
    self._require_close(
        adjoint.adjoint_ao_potential,
        expected_adjoint_potential,
        "AO potential",
    )
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
    expected_adjoint_nuclear = -np.einsum(
        "ai,...ai->...",
        zvector,
        bare_nuclear_rhs,
    )
    objective_occupied = objective_mo[occupied][:, occupied]
    objective_occupied = 0.5 * (
        objective_occupied + objective_occupied.T
    )
    adjoint_potential_mo = (
        coefficient.T @ expected_adjoint_potential @ coefficient
    )
    adjoint_potential_occupied = adjoint_potential_mo[occupied][
        :, occupied
    ]
    adjoint_potential_occupied = 0.5 * (
        adjoint_potential_occupied + adjoint_potential_occupied.T
    )
    overlap_occupied = overlap_mo[..., occupied, :]
    expected_metric = np.einsum(
        "...ij,ij->...",
        overlap_occupied,
        -2.0 * objective_occupied,
    )
    expected_adjoint_metric = np.einsum(
        "...ij,ij->...",
        overlap_occupied,
        0.5 * adjoint_potential_occupied,
    )
    expected_occupied_virtual = (
        expected_adjoint_nuclear + expected_adjoint_metric
    )
    expected_response = expected_metric + expected_occupied_virtual
    expected_gradients = {
        "correction_gradient_metric": expected_metric,
        "correction_gradient_adjoint_nuclear": expected_adjoint_nuclear,
        "correction_gradient_adjoint_metric": expected_adjoint_metric,
        "correction_gradient_occupied_virtual": expected_occupied_virtual,
        "correction_gradient_response": expected_response,
    }
    for name, expected in expected_gradients.items():
        self._require_close(
            getattr(adjoint, name),
            expected,
            name,
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

    def residual_statistics(value):
        return (
            float(np.max(np.abs(value), initial=0.0)),
            float(np.sqrt(np.mean(np.square(value)))),
        )

    maximum_residual, residual_rms = residual_statistics(residual)
    measured = {
        "minimum_orbital_gap": minimum_gap,
        "response_dimension": response_dimension,
        "objective_symmetry_residual": float(
            np.max(
                np.abs(
                    expected_objective_ao_potential
                    - expected_objective_ao_potential.T
                ),
                initial=0.0,
            )
        ),
        "adjoint_density_symmetry_residual": float(
            np.max(
                np.abs(
                    expected_adjoint_density - expected_adjoint_density.T
                ),
                initial=0.0,
            )
        ),
        "adjoint_potential_symmetry_residual": float(
            np.max(
                np.abs(
                    expected_adjoint_potential
                    - expected_adjoint_potential.T
                ),
                initial=0.0,
            )
        ),
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
            raise RHFAdjointError(
                f"the supplied RHF adjoint {name} diagnostic is inconsistent"
            )
    if (
        maximum_residual > diagnostics.residual_tolerance
        or minimum_gap <= diagnostics.orbital_gap_tolerance
        or measured["objective_symmetry_residual"]
        > diagnostics.objective_symmetry_tolerance
        or measured["adjoint_density_symmetry_residual"]
        > diagnostics.objective_symmetry_tolerance
        or measured["adjoint_potential_symmetry_residual"]
        > diagnostics.objective_symmetry_tolerance
    ):
        raise RHFAdjointError(
            "the supplied RHF adjoint exceeds an accepted control"
        )
    validate_reference(self.reference)


__all__ = ['audit_adjoint']
