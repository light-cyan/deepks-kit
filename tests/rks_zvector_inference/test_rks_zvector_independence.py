from dataclasses import dataclass

import numpy as np

import deepks.deephf.adjoint as adjoint_module
import deepks.deephf.pyscf_rks as rks_adapter_module
from deepks.deephf.adjoint import solve_scalar_adjoint


@dataclass(frozen=True)
class _NonsymmetricProblem:
    matrix: np.ndarray

    @property
    def dimension(self):
        return self.matrix.shape[0]

    def dense_operator(self):
        return self.matrix.copy()

    def apply(self, vector):
        return self.matrix @ vector

    def apply_transpose(self, vector):
        return self.matrix.T @ vector


def test_reference_neutral_adjoint_uses_literal_transpose_for_nonsymmetric_case():
    matrix = np.array(
        [
            [3.4, 0.8, -0.3],
            [0.1, 2.2, 0.6],
            [-0.5, 0.2, 1.8],
        ],
        dtype=np.float64,
    )
    bilateral_objective = np.array([0.37, -0.26, 0.19], dtype=np.float64)
    result = solve_scalar_adjoint(
        _NonsymmetricProblem(matrix),
        bilateral_objective,
    )
    expected = np.linalg.solve(matrix.T, bilateral_objective)
    incorrect_forward = np.linalg.solve(matrix, bilateral_objective)

    np.testing.assert_allclose(
        result.solution,
        expected,
        rtol=2.0e-15,
        atol=2.0e-15,
    )
    assert np.max(np.abs(result.solution - incorrect_forward)) > 1.0e-2
    assert result.diagnostics.solve_count == 1
    assert result.diagnostics.maximum_solver_residual < 1.0e-14
    assert result.diagnostics.maximum_transpose_residual < 1.0e-14
    assert result.diagnostics.maximum_physical_residual > 1.0e-2


def test_rks_zvector_never_enters_direct_or_coordinate_density_paths(
    rks_oracle_case,
    monkeypatch,
):
    from deepks.deephf.pyscf_rks import RKSResponseAdapter
    from deepks.deephf.rks_gradient import RKSDeePHFGradients
    from deepks.deephf.rks_method import RKSDeePHF

    def forbidden(*_args, **_kwargs):
        raise AssertionError("the RKS Z-vector entered a direct-response path")

    monkeypatch.setattr(RKSResponseAdapter, "solve", forbidden)
    monkeypatch.setattr(RKSResponseAdapter, "_solve_orbitals", forbidden)
    monkeypatch.setattr(RKSResponseAdapter, "audit_response_equations", forbidden)
    for name in (
        "response",
        "first_order_density",
        "dq_dR_response",
        "dq_dR_relaxed",
    ):
        monkeypatch.setattr(RKSDeePHF, name, forbidden)
    monkeypatch.setattr(RKSDeePHFGradients, "kernel", forbidden)
    monkeypatch.setattr(RKSDeePHFGradients, "_kernel", forbidden)

    driver = rks_oracle_case.method.nuc_grad_method(
        backend="zvector"
    ).run()

    assert np.isfinite(driver.de_full).all()
    assert not hasattr(driver, "dq_dR_response")
    assert not hasattr(driver, "dq_dR_relaxed")
    for name in (
        "density_response",
        "first_order_density",
        "dq_dR_response",
        "dq_dR_relaxed",
    ):
        assert not hasattr(driver.adjoint_result, name)


def test_one_rks_scalar_correction_performs_exactly_one_adjoint_solve(
    rks_oracle_case,
    monkeypatch,
):
    original_adapter_solve = rks_adapter_module.RKSAdjointAdapter.solve
    original_scalar_solve = rks_adapter_module.solve_scalar_adjoint
    original_dense_solve = adjoint_module.np.linalg.solve
    adapter_calls = []
    scalar_calls = []
    dense_calls = []

    def counted_adapter_solve(self, objective):
        adapter_calls.append(np.asarray(objective).shape)
        return original_adapter_solve(self, objective)

    def counted_scalar_solve(problem, objective, **options):
        scalar_calls.append((problem.dimension, np.asarray(objective).shape))
        return original_scalar_solve(problem, objective, **options)

    def counted_dense_solve(matrix, right_hand_side):
        dense_calls.append(
            (np.asarray(matrix).shape, np.asarray(right_hand_side).shape)
        )
        return original_dense_solve(matrix, right_hand_side)

    monkeypatch.setattr(
        rks_adapter_module.RKSAdjointAdapter,
        "solve",
        counted_adapter_solve,
    )
    monkeypatch.setattr(
        rks_adapter_module,
        "solve_scalar_adjoint",
        counted_scalar_solve,
    )
    monkeypatch.setattr(
        adjoint_module.np.linalg,
        "solve",
        counted_dense_solve,
    )

    driver = rks_oracle_case.method.nuc_grad_method(
        backend="zvector"
    ).run()

    assert adapter_calls == [(rks_oracle_case.reference.mol.nao,) * 2]
    assert scalar_calls == [(10, (10,))]
    assert dense_calls == [((10, 10), (10,))]
    assert driver.adjoint_result.diagnostics.solve_count == 1


def test_rks_zvector_result_cannot_masquerade_as_force_training_jacobian(
    rks_oracle_case,
):
    method = rks_oracle_case.method
    driver = method.nuc_grad_method(backend="zvector").run()
    adjoint = driver.adjoint_result

    assert driver.backend == "zvector"
    assert hasattr(driver, "dq_dR_explicit")
    assert not hasattr(driver, "dq_dR_response")
    assert not hasattr(driver, "dq_dR_relaxed")
    assert not hasattr(adjoint, "descriptor_response")
    assert not hasattr(adjoint, "descriptor_jacobian")
    assert not hasattr(adjoint, "density_response")
    assert not hasattr(adjoint, "force_data")
