from dataclasses import replace
import sys

import numpy as np
import pytest

import deepks.deephf.adjoint as adjoint_module
import deepks.deephf.pyscf_rks as rks_adapter_module
from deepks.deephf.capabilities import DeePHFCapabilityError
from deepks.deephf.pyscf_rks import (
    RKSAdjointAdapter,
    RKSAdjointError,
    RKSResponseError,
    rks_adjoint_integrity_fingerprint,
)
_P4B_FIXTURES = sys.modules["_deepks_p4b_rks_oracle_fixtures"]


@pytest.fixture(scope="module")
def rks_reference():
    return _P4B_FIXTURES.run_fresh_rks()


@pytest.fixture(scope="module")
def objective_ao_potential(rks_reference):
    nao = rks_reference.mol.nao
    random = np.random.default_rng(9304)
    objective = random.normal(size=(nao, nao)).astype(np.float64)
    return 0.5 * (objective + objective.T)


@pytest.fixture(scope="module")
def rks_adjoint_adapter(rks_reference):
    return RKSAdjointAdapter(rks_reference)


@pytest.fixture(scope="module")
def rks_adjoint(rks_adjoint_adapter, objective_ao_potential):
    return rks_adjoint_adapter.solve(objective_ao_potential)


def _reseal(adjoint, **changes):
    changed = replace(adjoint, integrity_fingerprint="", **changes)
    return replace(
        changed,
        integrity_fingerprint=rks_adjoint_integrity_fingerprint(changed),
    )


@pytest.mark.parametrize(
    "name",
    [
        "residual_tolerance",
        "invariant_tolerance",
        "orbital_gap_tolerance",
        "operator_stability_tolerance",
        "operator_condition_tolerance",
        "operator_symmetry_tolerance",
        "objective_symmetry_tolerance",
    ],
)
@pytest.mark.parametrize("value", [True, "1e-9", 1.0e-9 + 0.0j, np.array(1.0e-9)])
def test_adjoint_rejects_non_real_scalar_controls(
    rks_reference,
    monkeypatch,
    name,
    value,
):
    monkeypatch.setattr(
        rks_adapter_module,
        "validate_rks_reference",
        lambda reference: reference,
    )

    with pytest.raises(ValueError, match="must be a real numeric scalar"):
        RKSAdjointAdapter(rks_reference, **{name: value})


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"residual_tolerance": 0.0}, "tolerances must be positive"),
        ({"invariant_tolerance": -1.0}, "tolerances must be positive"),
        ({"orbital_gap_tolerance": 0.0}, "tolerances must be positive"),
        ({"operator_stability_tolerance": 0.0}, "tolerances must be positive"),
        ({"operator_condition_tolerance": 1.0}, "must exceed one"),
        ({"operator_symmetry_tolerance": 0.0}, "tolerances must be positive"),
        ({"objective_symmetry_tolerance": 0.0}, "must be positive"),
        ({"operator_dimension_limit": 0}, "must be positive"),
        ({"operator_dimension_limit": True}, "must be an integer"),
    ],
)
def test_adjoint_rejects_invalid_numeric_controls(
    rks_reference,
    monkeypatch,
    options,
    message,
):
    monkeypatch.setattr(
        rks_adapter_module,
        "validate_rks_reference",
        lambda reference: reference,
    )

    with pytest.raises(ValueError, match=message):
        RKSAdjointAdapter(rks_reference, **options)


def test_adjoint_rejects_asymmetric_objective(
    rks_adjoint_adapter,
    objective_ao_potential,
):
    objective = objective_ao_potential.copy()
    objective[0, 1] += 1.0e-4

    with pytest.raises(RKSAdjointError, match="violates symmetry"):
        rks_adjoint_adapter.solve(objective)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (
            {"operator_stability_tolerance": 1.0e6},
            "unstable or singular",
        ),
        (
            {"operator_condition_tolerance": 2.0},
            "ill conditioned",
        ),
        (
            {"operator_symmetry_tolerance": 1.0e-18},
            "violates symmetry",
        ),
    ],
)
def test_adjoint_enforces_operator_gates(
    rks_reference,
    objective_ao_potential,
    monkeypatch,
    options,
    message,
):
    monkeypatch.setattr(
        rks_adapter_module,
        "validate_rks_reference",
        lambda reference: reference,
    )
    adapter = RKSAdjointAdapter(rks_reference, **options)

    with pytest.raises(
        (DeePHFCapabilityError, RKSResponseError),
        match=message,
    ):
        adapter.validate_response_operator_exact()


def test_adjoint_production_solve_does_not_use_dense_debug_audit(
    rks_reference,
    objective_ao_potential,
    monkeypatch,
):
    def forbidden_dense_audit(*_args, **_kwargs):
        raise AssertionError("the RKS response matrix was materialized")

    monkeypatch.setattr(
        rks_adapter_module._RKSLinearResponseCore,
        "_response_operator_matrix_and_diagnostics",
        forbidden_dense_audit,
    )
    adjoint = RKSAdjointAdapter(
        rks_reference,
        operator_dimension_limit=1,
    ).solve(objective_ao_potential)

    assert adjoint.diagnostics.response_dimension > 1
    assert adjoint.diagnostics.iteration_count > 0


def test_adjoint_wraps_dense_solver_failure(
    rks_adjoint_adapter,
    objective_ao_potential,
    monkeypatch,
):
    def failed_solve(*_args, **_kwargs):
        raise np.linalg.LinAlgError("injected singular transpose")

    monkeypatch.setattr(adjoint_module, "gmres", failed_solve)

    with pytest.raises(RKSAdjointError, match="GMRES adjoint solver raised"):
        rks_adjoint_adapter.solve(objective_ao_potential)


def test_adjoint_rejects_nonfinite_solver_result(
    rks_adjoint_adapter,
    objective_ao_potential,
    monkeypatch,
):
    def nonfinite_solve(_operator, right_hand_side, **_options):
        return np.full_like(right_hand_side, np.nan), 0

    monkeypatch.setattr(adjoint_module, "gmres", nonfinite_solve)

    with pytest.raises(RKSAdjointError, match="adjoint solution must be finite"):
        rks_adjoint_adapter.solve(objective_ao_potential)


def test_adjoint_rejects_literal_transpose_residual(
    rks_adjoint_adapter,
    objective_ao_potential,
    monkeypatch,
):
    original_solve = adjoint_module.gmres

    def corrupted_solve(matrix, right_hand_side, **options):
        solution, info = original_solve(matrix, right_hand_side, **options)
        return solution + 1.0e-3, info

    monkeypatch.setattr(adjoint_module, "gmres", corrupted_solve)

    with pytest.raises(RKSAdjointError, match="solver residual exceeds tolerance"):
        rks_adjoint_adapter.solve(objective_ao_potential)


def test_adjoint_audit_rejects_foreign_and_stale_provenance(
    rks_adjoint_adapter,
    rks_adjoint,
    objective_ao_potential,
):
    foreign = _reseal(
        rks_adjoint,
        reference_identity=rks_adjoint.reference_identity + 1,
    )
    stale = _reseal(
        rks_adjoint,
        state_fingerprint="0" * 64,
    )

    with pytest.raises(RKSAdjointError, match="another reference"):
        rks_adjoint_adapter.audit_adjoint(foreign, objective_ao_potential)
    with pytest.raises(RKSAdjointError, match="current RKS state"):
        rks_adjoint_adapter.audit_adjoint(stale, objective_ao_potential)


def test_adjoint_audit_rejects_mutable_and_coordinated_resealed_arrays(
    rks_adjoint_adapter,
    rks_adjoint,
    objective_ao_potential,
):
    mutable = _reseal(
        rks_adjoint,
        zvector=np.array(rks_adjoint.zvector, copy=True),
    )
    changed_zvector = np.array(rks_adjoint.zvector, copy=True)
    changed_zvector[0, 0] += 1.0e-5
    changed_zvector.setflags(write=False)
    resealed = _reseal(rks_adjoint, zvector=changed_zvector)

    with pytest.raises(RKSAdjointError, match="must be immutable"):
        rks_adjoint_adapter.audit_adjoint(mutable, objective_ao_potential)
    with pytest.raises(RKSAdjointError, match="independent residual"):
        rks_adjoint_adapter.audit_adjoint(resealed, objective_ao_potential)


@pytest.mark.parametrize(
    ("field", "value_factory", "message"),
    [
        pytest.param(
            "operator_fingerprint",
            lambda adjoint: "0" * len(adjoint.operator_fingerprint),
            "response operator is inconsistent",
            id="operator",
        ),
        pytest.param(
            "functional_provenance",
            lambda adjoint: replace(
                adjoint.functional_provenance,
                backend_version="forged",
            ),
            "functional provenance is invalid",
            id="functional-provenance",
        ),
        pytest.param(
            "grid_provenance",
            lambda adjoint: replace(
                adjoint.grid_provenance,
                weights_fingerprint="0" * 64,
            ),
            "grid provenance is invalid",
            id="grid-provenance",
        ),
        pytest.param(
            "diagnostics",
            lambda adjoint: replace(adjoint.diagnostics, solve_count=2),
            "exactly one scalar solve",
            id="solve-count",
        ),
        pytest.param(
            "diagnostics",
            lambda adjoint: replace(
                adjoint.diagnostics,
                maximum_residual=False,
            ),
            "finite real scalars",
            id="diagnostic-type",
        ),
    ],
)
def test_adjoint_audit_rejects_coordinated_resealed_provenance(
    rks_adjoint_adapter,
    rks_adjoint,
    objective_ao_potential,
    field,
    value_factory,
    message,
):
    forged = _reseal(
        rks_adjoint,
        **{field: value_factory(rks_adjoint)},
    )

    with pytest.raises(RKSAdjointError, match=message):
        rks_adjoint_adapter.audit_adjoint(
            forged,
            objective_ao_potential,
        )


def test_adjoint_audit_rejects_resealed_objective_and_gradient(
    rks_adjoint_adapter,
    rks_adjoint,
    objective_ao_potential,
):
    changed_objective = np.array(
        rks_adjoint.objective_ao_potential,
        copy=True,
    )
    changed_objective[0, 0] += 1.0e-4
    changed_objective.setflags(write=False)
    changed_gradient = np.array(
        rks_adjoint.correction_gradient_adjoint_grid_weight,
        copy=True,
    )
    changed_gradient[0, 0] += 1.0e-4
    changed_gradient.setflags(write=False)
    forged_objective = _reseal(
        rks_adjoint,
        objective_ao_potential=changed_objective,
    )
    forged_gradient = _reseal(
        rks_adjoint,
        correction_gradient_adjoint_grid_weight=changed_gradient,
    )

    with pytest.raises(RKSAdjointError, match="objective AO potential"):
        rks_adjoint_adapter.audit_adjoint(
            forged_objective,
            objective_ao_potential,
        )
    with pytest.raises(RKSAdjointError, match="adjoint_grid_weight"):
        rks_adjoint_adapter.audit_adjoint(
            forged_gradient,
            objective_ao_potential,
        )


def test_adjoint_audit_never_performs_a_second_solve(
    rks_adjoint_adapter,
    rks_adjoint,
    objective_ao_potential,
    monkeypatch,
):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("the RKS adjoint audit attempted another solve")

    monkeypatch.setattr(rks_adapter_module, "solve_scalar_adjoint", forbidden)
    monkeypatch.setattr(adjoint_module.np.linalg, "solve", forbidden)
    monkeypatch.setattr(rks_adapter_module.cphf, "solve", forbidden)

    rks_adjoint_adapter.audit_adjoint(
        rks_adjoint,
        objective_ao_potential,
    )
