"""Bounded dense audits separated from production solver assembly."""

from __future__ import annotations

from numbers import Real
from ..pyscf_uhf_reference import UHFAdjoint
from ..pyscf_uhf_reference import UHFAdjointDiagnostics
from ..pyscf_uhf_reference import UHFAdjointError
from dataclasses import fields
import numpy as np
import pyscf
from ..adjoint import scalar_operator_fingerprint
from ..pyscf_uhf_reference import uhf_adjoint_integrity_fingerprint
from ..pyscf_uhf_adjoint import (
    _UHFScalarAdjointProblem,
)


def _validated_adjoint_contract(self, adjoint):
    """Independently audit one consumed UHF adjoint without another solve."""
    self._validate_reference(self.reference)
    if type(adjoint) is not UHFAdjoint:
        raise UHFAdjointError("the supplied UHF adjoint has an invalid type")
    diagnostics = adjoint.diagnostics
    if type(diagnostics) is not UHFAdjointDiagnostics:
        raise UHFAdjointError(
            "the supplied UHF adjoint diagnostics have an invalid type"
        )
    if adjoint.reference_identity != id(self.reference):
        raise UHFAdjointError(
            "the supplied UHF adjoint belongs to another reference"
        )
    if adjoint.state_fingerprint != self._reference_fingerprint(self.reference):
        raise UHFAdjointError(
            "the supplied UHF adjoint does not match the current UHF state"
        )
    if adjoint.integrity_fingerprint != uhf_adjoint_integrity_fingerprint(
        adjoint
    ):
        raise UHFAdjointError(
            "the supplied UHF adjoint failed its integrity check"
        )
    provenance = (
        adjoint.reference_identity,
        adjoint.state_fingerprint,
        adjoint.integrity_fingerprint,
        adjoint.operator_fingerprint,
    )
    if type(provenance[0]) is not int or any(
        type(value) is not str for value in provenance[1:]
    ):
        raise UHFAdjointError(
            "the supplied UHF adjoint provenance fields have invalid types"
        )
    if diagnostics.pyscf_version != pyscf.__version__:
        raise UHFAdjointError(
            "the supplied UHF adjoint PySCF version does not match the runtime"
        )
    if diagnostics.solver != "scipy.sparse.linalg.gmres(A.T, b)":
        raise UHFAdjointError(
            "the supplied UHF adjoint solver convention is invalid"
        )
    integer_diagnostics = (
        diagnostics.response_dimension,
        diagnostics.alpha_response_dimension,
        diagnostics.beta_response_dimension,
        diagnostics.solve_count,
        diagnostics.max_cycle,
        diagnostics.krylov_restart,
        diagnostics.iteration_count,
    )
    if any(type(value) is not int for value in integer_diagnostics):
        raise UHFAdjointError(
            "the supplied UHF adjoint integer diagnostics are invalid"
        )
    if diagnostics.solve_count != 1:
        raise UHFAdjointError(
            "the supplied UHF adjoint must contain exactly one scalar solve"
        )
    if diagnostics.operator_is_self_adjoint is not True:
        raise UHFAdjointError("the supplied UHF adjoint operator contract is invalid")
    real_names = tuple(
        field.name
        for field in fields(diagnostics)
        if field.name
        not in {
            "pyscf_version",
            "response_dimension",
            "alpha_response_dimension",
            "beta_response_dimension",
            "operator_is_self_adjoint",
            "solver",
            "solve_count",
            "max_cycle",
            "krylov_restart",
            "iteration_count",
        }
    )
    real_values = tuple(getattr(diagnostics, name) for name in real_names)
    if any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
        for value in real_values
    ) or not np.isfinite(real_values).all():
        raise UHFAdjointError(
            "the supplied UHF adjoint diagnostics must be finite real scalars"
        )
    accepted_controls = {
        "residual_tolerance": self.residual_tolerance,
        "invariant_tolerance": self.invariant_tolerance,
        "orbital_gap_tolerance": self.orbital_gap_tolerance,
        "objective_symmetry_tolerance": self.objective_symmetry_tolerance,
        "max_cycle": self.max_cycle,
        "krylov_restart": self.krylov_restart,
    }
    for name, expected in accepted_controls.items():
        if getattr(diagnostics, name) != expected:
            raise UHFAdjointError(
                f"the supplied UHF adjoint {name} control is inconsistent"
            )
    return diagnostics


def _validated_adjoint_state(self, adjoint, expected_objective_ao_potential):
    coefficient, energy, occupation, occupied, virtual, minimum_gaps = (
        self._state()
    )
    dimensions = self._dimensions(occupied, virtual)
    alpha_nocc, beta_nocc, alpha_nvir, beta_nvir = dimensions[:4]
    alpha_dimension, beta_dimension = dimensions[-2:]
    dimension = alpha_dimension + beta_dimension
    atom_indices = self._response_atom_indices(adjoint.atom_indices)
    if atom_indices != adjoint.atom_indices:
        raise UHFAdjointError("the supplied UHF adjoint atom selection is invalid")
    natm = len(atom_indices)
    nao = int(self.molecule.nao)
    alpha_shape = (alpha_nvir, alpha_nocc)
    beta_shape = (beta_nvir, beta_nocc)
    arrays = {
        "objective_ao_potential": (nao, nao),
        "alpha_objective_orbital_gradient": alpha_shape,
        "beta_objective_orbital_gradient": beta_shape,
        "alpha_zvector": alpha_shape,
        "beta_zvector": beta_shape,
        "alpha_residual": alpha_shape,
        "beta_residual": beta_shape,
        "alpha_adjoint_ao_density": (nao, nao),
        "beta_adjoint_ao_density": (nao, nao),
        "alpha_adjoint_ao_potential": (nao, nao),
        "beta_adjoint_ao_potential": (nao, nao),
        "correction_gradient_metric_spin": (2, natm, 3),
        "correction_gradient_metric": (natm, 3),
        "correction_gradient_adjoint_nuclear_spin": (2, natm, 3),
        "correction_gradient_adjoint_nuclear": (natm, 3),
        "correction_gradient_adjoint_metric_spin": (2, natm, 3),
        "correction_gradient_adjoint_metric": (natm, 3),
        "correction_gradient_occupied_virtual_spin": (2, natm, 3),
        "correction_gradient_occupied_virtual": (natm, 3),
        "correction_gradient_response": (natm, 3),
    }
    for name, shape in arrays.items():
        self._audited_array(getattr(adjoint, name), shape, name)
    expected_objective = self._validated_objective_potential(
        expected_objective_ao_potential
    )
    self._require_close(
        adjoint.objective_ao_potential,
        expected_objective,
        "objective AO potential",
    )
    objective_mo, objective_gradients = self._objective_gradients(
        expected_objective,
        coefficient,
        occupation,
        occupied,
        virtual,
    )
    self._require_close(
        adjoint.alpha_objective_orbital_gradient,
        objective_gradients[0],
        "alpha bilateral occupied-virtual objective gradient",
    )
    self._require_close(
        adjoint.beta_objective_orbital_gradient,
        objective_gradients[1],
        "beta bilateral occupied-virtual objective gradient",
    )
    problem = _UHFScalarAdjointProblem(
        self,
        coefficient,
        energy,
        occupied,
        virtual,
    )
    expected_operator_fingerprint = scalar_operator_fingerprint(
        problem,
        solver="gmres",
    )
    if adjoint.operator_fingerprint != expected_operator_fingerprint:
        raise UHFAdjointError(
            "the supplied UHF adjoint response operator is inconsistent"
        )
    return (
        coefficient, energy, occupation, occupied, virtual, minimum_gaps,
        alpha_dimension, beta_dimension, dimension, atom_indices,
        expected_objective, objective_mo, objective_gradients,
    )


def _audit_adjoint_result(self, adjoint, diagnostics, state):
    (
        coefficient, energy, occupation, occupied, virtual, minimum_gaps,
        alpha_dimension, beta_dimension, dimension, atom_indices,
        expected_objective, objective_mo, objective_gradients,
    ) = state
    zvector = (adjoint.alpha_zvector, adjoint.beta_zvector)
    zflat = np.concatenate(tuple(value.reshape(-1) for value in zvector))
    objective_flat = np.concatenate(
        tuple(value.reshape(-1) for value in objective_gradients)
    )
    residual = self._split_occupied_virtual(
        self._apply_occupied_virtual_operator(
            zflat,
            coefficient,
            energy,
            occupied,
            virtual,
        )
        - objective_flat,
        occupied,
        virtual,
    )
    for spin, spin_name in enumerate(("alpha", "beta")):
        self._require_close(
            getattr(adjoint, f"{spin_name}_residual"),
            residual[spin],
            f"{spin_name} independent residual",
        )
    (
        expected_density,
        expected_potential,
        density_symmetry,
        potential_symmetry,
    ) = self._adjoint_densities_and_potentials(
        zvector,
        coefficient,
        occupation,
        occupied,
        virtual,
    )
    for spin, spin_name in enumerate(("alpha", "beta")):
        self._require_close(
            getattr(adjoint, f"{spin_name}_adjoint_ao_density"),
            expected_density[spin],
            f"{spin_name} AO density",
        )
        self._require_close(
            getattr(adjoint, f"{spin_name}_adjoint_ao_potential"),
            expected_potential[spin],
            f"{spin_name} AO potential",
        )
    partitions = self._gradient_partitions(
        objective_mo,
        zvector,
        expected_potential,
        coefficient,
        energy,
        occupation,
        occupied,
        virtual,
        atom_indices,
    )
    partition_fields = (
        "correction_gradient_metric_spin",
        "correction_gradient_metric",
        "correction_gradient_adjoint_nuclear_spin",
        "correction_gradient_adjoint_nuclear",
        "correction_gradient_adjoint_metric_spin",
        "correction_gradient_adjoint_metric",
        "correction_gradient_occupied_virtual_spin",
        "correction_gradient_occupied_virtual",
        "correction_gradient_response",
    )
    for name, expected in zip(partition_fields, partitions[:-1], strict=True):
        self._require_close(getattr(adjoint, name), expected, name)
    combined_residual = np.concatenate(
        tuple(value.reshape(-1) for value in residual)
    )
    measured = {
        "minimum_alpha_orbital_gap": minimum_gaps[0],
        "minimum_beta_orbital_gap": minimum_gaps[1],
        "response_dimension": dimension,
        "alpha_response_dimension": alpha_dimension,
        "beta_response_dimension": beta_dimension,
        "objective_symmetry_residual": float(
            np.max(np.abs(expected_objective - expected_objective.T), initial=0.0)
        ),
        "alpha_adjoint_density_symmetry_residual": density_symmetry[0],
        "beta_adjoint_density_symmetry_residual": density_symmetry[1],
        "alpha_adjoint_potential_symmetry_residual": potential_symmetry[0],
        "beta_adjoint_potential_symmetry_residual": potential_symmetry[1],
        "gradient_reconstruction_residual": partitions[-1],
        "objective_gradient_norm": float(np.linalg.norm(objective_flat)),
        "solution_norm": float(np.linalg.norm(zflat)),
        "maximum_residual": float(
            np.max(np.abs(combined_residual), initial=0.0)
        ),
        "residual_rms": float(
            np.sqrt(np.mean(np.square(combined_residual)))
        ),
    }
    for name, expected in measured.items():
        if not np.isclose(
            getattr(diagnostics, name),
            expected,
            rtol=1.0e-10,
            atol=1.0e-12,
        ):
            raise UHFAdjointError(
                f"the supplied UHF adjoint {name} diagnostic is inconsistent"
            )
    if measured["maximum_residual"] > self.residual_tolerance:
        raise UHFAdjointError(
            "the supplied UHF adjoint residual exceeds its tolerance"
        )
    if measured["gradient_reconstruction_residual"] > self.invariant_tolerance:
        raise UHFAdjointError(
            "the supplied UHF adjoint invariant exceeds its tolerance"
        )
    self._validate_reference(self.reference)


def audit_adjoint(
    self,
    adjoint: UHFAdjoint,
    expected_objective_ao_potential: np.ndarray,
) -> None:
    """Independently audit one consumed UHF adjoint without another solve."""
    diagnostics = _validated_adjoint_contract(self, adjoint)
    state = _validated_adjoint_state(
        self, adjoint, expected_objective_ao_potential
    )
    _audit_adjoint_result(self, adjoint, diagnostics, state)


__all__ = ['audit_adjoint']
