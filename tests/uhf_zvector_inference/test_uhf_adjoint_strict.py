from dataclasses import replace

import numpy as np
import pytest

import deepks.deephf.adjoint as neutral_adjoint
import deepks.deephf.pyscf_uhf as pyscf_uhf
from pyscf import scf
from deepks.deephf import (
    DeePHFCapabilityError,
    UHFAdjointAdapter,
    UHFAdjointError,
)

@pytest.mark.parametrize(
    ("option", "value"),
    [
        pytest.param("residual_tolerance", True, id="residual-bool"),
        pytest.param("residual_tolerance", np.nan, id="residual-nan"),
        pytest.param("residual_tolerance", 0.0, id="residual-zero"),
        pytest.param("invariant_tolerance", np.inf, id="invariant-inf"),
        pytest.param("invariant_tolerance", -1.0, id="invariant-negative"),
        pytest.param("orbital_gap_tolerance", "1e-7", id="gap-string"),
        pytest.param("operator_stability_tolerance", 0.0, id="stability-zero"),
        pytest.param("operator_condition_tolerance", 1.0, id="condition-one"),
        pytest.param("operator_symmetry_tolerance", -1.0, id="symmetry-negative"),
        pytest.param("operator_dimension_limit", False, id="dimension-bool"),
        pytest.param("operator_dimension_limit", 0, id="dimension-zero"),
        pytest.param("objective_symmetry_tolerance", 0.0, id="objective-zero"),
    ],
)
def test_adjoint_controls_reject_invalid_values(
    uhf_oracle_case,
    option,
    value,
):
    with pytest.raises((TypeError, ValueError)):
        UHFAdjointAdapter(
            uhf_oracle_case.reference,
            **{option: value},
        )


def test_adjoint_rejects_a_nonsymmetric_objective(
    uhf_oracle_case,
    independent_uhf_adjoint_oracle,
):
    objective = independent_uhf_adjoint_oracle.objective_ao_potential.copy()
    objective[0, 1] += 1.0e-3

    with pytest.raises(UHFAdjointError, match="violates symmetry"):
        UHFAdjointAdapter(uhf_oracle_case.reference).solve(objective)


def test_adjoint_operator_dimension_stability_and_condition_gates(
    uhf_oracle_case,
    independent_uhf_adjoint_oracle,
):
    objective = independent_uhf_adjoint_oracle.objective_ao_potential
    dimension = uhf_oracle_case.coupled_operator.shape[0]
    minimum = float(np.linalg.eigvalsh(uhf_oracle_case.coupled_operator)[0])
    condition = float(np.linalg.cond(uhf_oracle_case.coupled_operator))

    with pytest.raises(DeePHFCapabilityError, match="dimension exceeds"):
        UHFAdjointAdapter(
            uhf_oracle_case.reference,
            operator_dimension_limit=dimension - 1,
        ).solve(objective)
    with pytest.raises(DeePHFCapabilityError, match="unstable or singular"):
        UHFAdjointAdapter(
            uhf_oracle_case.reference,
            operator_stability_tolerance=minimum * 1.01,
        ).solve(objective)
    with pytest.raises(DeePHFCapabilityError, match="ill conditioned"):
        UHFAdjointAdapter(
            uhf_oracle_case.reference,
            operator_condition_tolerance=condition * 0.99,
        ).solve(objective)


def test_adjoint_rejects_operator_asymmetry_before_solve(
    uhf_oracle_case,
    independent_uhf_adjoint_oracle,
    monkeypatch,
):
    objective = independent_uhf_adjoint_oracle.objective_ao_potential
    original = UHFAdjointAdapter._apply_occupied_virtual_operator

    def asymmetric(self, vectors, *args, **kwargs):
        result = np.asarray(original(self, vectors, *args, **kwargs)).copy()
        array = np.asarray(vectors)
        if array.ndim == 2 and array.shape[0] > 1:
            result[0, 1] += 1.0e-3
        return result

    monkeypatch.setattr(
        UHFAdjointAdapter,
        "_apply_occupied_virtual_operator",
        asymmetric,
    )
    with pytest.raises(UHFAdjointError, match="violates symmetry"):
        UHFAdjointAdapter(uhf_oracle_case.reference).solve(objective)


def test_adjoint_solver_and_independent_residual_faults_are_explicit(
    uhf_oracle_case,
    independent_uhf_adjoint_oracle,
    monkeypatch,
):
    objective = independent_uhf_adjoint_oracle.objective_ao_potential

    def failed_solve(*args, **kwargs):
        raise np.linalg.LinAlgError("injected transpose failure")

    monkeypatch.setattr(neutral_adjoint.np.linalg, "solve", failed_solve)
    with pytest.raises(UHFAdjointError, match="adjoint evaluation failed"):
        UHFAdjointAdapter(uhf_oracle_case.reference).solve(objective)


def test_adjoint_physical_residual_fault_is_not_hidden(
    uhf_oracle_case,
    independent_uhf_adjoint_oracle,
    monkeypatch,
):
    objective = independent_uhf_adjoint_oracle.objective_ao_potential
    original = pyscf_uhf._UHFScalarAdjointProblem.apply

    def corrupted(self, vector):
        result = np.asarray(original(self, vector)).copy()
        result[0] += 1.0e-4
        return result

    monkeypatch.setattr(
        pyscf_uhf._UHFScalarAdjointProblem,
        "apply",
        corrupted,
    )
    with pytest.raises(
        UHFAdjointError,
        match="transpose adjoint residual|physical adjoint residual",
    ):
        UHFAdjointAdapter(uhf_oracle_case.reference).solve(objective)


def test_adjoint_arrays_are_immutable_and_audit_does_not_resolve(
    uhf_oracle_case,
    independent_uhf_adjoint_oracle,
    monkeypatch,
):
    objective = independent_uhf_adjoint_oracle.objective_ao_potential
    adapter = UHFAdjointAdapter(uhf_oracle_case.reference)
    adjoint = adapter.solve(objective)

    for field in adjoint.__dataclass_fields__.values():
        value = getattr(adjoint, field.name)
        if isinstance(value, np.ndarray):
            assert value.dtype == np.dtype(np.float64)
            assert not value.flags.writeable
            assert np.isfinite(value).all()

    def forbidden_solve(*args, **kwargs):
        raise AssertionError("audit must not solve another adjoint")

    monkeypatch.setattr(pyscf_uhf, "solve_scalar_adjoint", forbidden_solve)
    adapter.audit_adjoint(adjoint, objective)


def test_adjoint_audit_rejects_foreign_stale_and_resealed_results(
    uhf_oracle_case,
    independent_uhf_adjoint_oracle,
):
    objective = independent_uhf_adjoint_oracle.objective_ao_potential
    adapter = UHFAdjointAdapter(uhf_oracle_case.reference)
    adjoint = adapter.solve(objective)
    foreign_reference = scf.UHF(uhf_oracle_case.reference.mol)
    foreign_reference.conv_tol = 1.0e-13
    foreign_reference.conv_tol_grad = 1.0e-10
    foreign_reference.conv_tol_cpscf = 1.0e-12
    foreign_reference.max_cycle = 100
    foreign_reference.kernel()
    assert foreign_reference.converged

    with pytest.raises(UHFAdjointError, match="another reference"):
        UHFAdjointAdapter(foreign_reference).audit_adjoint(adjoint, objective)

    mutable = adjoint.alpha_zvector.copy()
    forged = replace(adjoint, alpha_zvector=mutable)
    forged = replace(
        forged,
        integrity_fingerprint=pyscf_uhf.uhf_adjoint_integrity_fingerprint(
            forged
        ),
    )
    with pytest.raises(UHFAdjointError, match="must be immutable"):
        adapter.audit_adjoint(forged, objective)

    shifted = np.frombuffer(
        (adjoint.alpha_zvector + 1.0e-3).tobytes(),
        dtype=np.float64,
    ).reshape(adjoint.alpha_zvector.shape)
    forged = replace(adjoint, alpha_zvector=shifted)
    forged = replace(
        forged,
        integrity_fingerprint=pyscf_uhf.uhf_adjoint_integrity_fingerprint(
            forged
        ),
    )
    with pytest.raises(UHFAdjointError, match="residual|density|inconsistent"):
        adapter.audit_adjoint(forged, objective)


def test_adjoint_audit_rejects_coordinated_control_forgery(
    uhf_oracle_case,
    independent_uhf_adjoint_oracle,
):
    objective = independent_uhf_adjoint_oracle.objective_ao_potential
    adapter = UHFAdjointAdapter(uhf_oracle_case.reference)
    adjoint = adapter.solve(objective)
    forged = replace(
        adjoint,
        diagnostics=replace(
            adjoint.diagnostics,
            residual_tolerance=1.0,
        ),
    )
    forged = replace(
        forged,
        integrity_fingerprint=pyscf_uhf.uhf_adjoint_integrity_fingerprint(
            forged
        ),
    )

    with pytest.raises(UHFAdjointError, match="control is inconsistent"):
        adapter.audit_adjoint(forged, objective)
