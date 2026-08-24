from dataclasses import dataclass
import hashlib

import numpy as np
import pytest
import torch

import deepks.deephf.adjoint as adjoint_module
from deepks.deephf.adjoint import solve_scalar_adjoint
from deepks.deephf.pyscf_rhf import RHFAdjointAdapter
from deepks.model.model import SCALE_EPS


@dataclass(frozen=True)
class _NonsymmetricAdjointProblem:
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


def test_self_adjoint_matrix_free_solve_has_one_postsolve_action(monkeypatch):
    calls = {"forward": 0, "transpose": 0}
    freeze_count = 0

    class IdentityProblem:
        dimension = 2
        is_self_adjoint = True
        operator_fingerprint = hashlib.sha256(b"identity").hexdigest()

        def apply(self, vector):
            calls["forward"] += 1
            return vector

        def apply_transpose(self, vector):
            calls["transpose"] += 1
            return vector

    def exact_gmres(_operator, objective, **_controls):
        return objective.copy(), 0

    original_freeze = adjoint_module._immutable_array

    def counted_freeze(value):
        nonlocal freeze_count
        freeze_count += 1
        return original_freeze(value)

    monkeypatch.setattr(adjoint_module, "gmres", exact_gmres)
    monkeypatch.setattr(adjoint_module, "_immutable_array", counted_freeze)
    result = solve_scalar_adjoint(
        IdentityProblem(),
        np.array([0.3, -0.2], dtype=np.float64),
        require_physical_residual=True,
    )
    assert calls == {"forward": 1, "transpose": 0}
    assert freeze_count == 5
    assert tuple(result.__dataclass_fields__) == (
        "operator_fingerprint",
        "integrity_fingerprint",
        "objective_gradient",
        "solution",
        "residual",
        "diagnostics",
    )


def test_false_self_adjoint_claim_cannot_skip_the_physical_residual(monkeypatch):
    class FalseClaim:
        dimension = 1
        is_self_adjoint = True
        operator_fingerprint = hashlib.sha256(b"false-claim").hexdigest()

        def apply(self, vector):
            return 2.0 * vector

        def apply_transpose(self, vector):
            return vector

    monkeypatch.setattr(
        adjoint_module,
        "gmres",
        lambda _operator, objective, **_controls: (objective.copy(), 0),
    )
    with pytest.raises(adjoint_module.AdjointError, match="residual exceeds tolerance"):
        solve_scalar_adjoint(FalseClaim(), np.ones(1), require_physical_residual=True)


def _independent_objective_ao_potential(case):
    density = torch.tensor(
        case.method.ao_density(),
        dtype=torch.float64,
        requires_grad=True,
    )
    descriptor = case.method._descriptor.torch_descriptor(density)
    energy = case.model(descriptor).sum()
    (potential,) = torch.autograd.grad(energy, density)
    return potential.detach().cpu().numpy()


@pytest.fixture(scope="session")
def objective_ao_potential(zvector_algebra_case):
    return _independent_objective_ao_potential(zvector_algebra_case)


@pytest.fixture(scope="session")
def rhf_adjoint(zvector_algebra_case, objective_ao_potential):
    return RHFAdjointAdapter(zvector_algebra_case.reference).solve(
        objective_ao_potential
    )


def test_generic_adjoint_solves_the_literal_transpose():
    matrix = np.array(
        [
            [3.2, 0.7, -0.2],
            [0.1, 2.4, 0.5],
            [-0.4, 0.2, 1.7],
        ],
        dtype=np.float64,
    )
    objective_gradient = np.array([0.31, -0.27, 0.19], dtype=np.float64)
    problem = _NonsymmetricAdjointProblem(matrix)

    result = solve_scalar_adjoint(problem, objective_gradient)
    expected_transpose = np.linalg.solve(matrix.T, objective_gradient)
    incorrect_forward = np.linalg.solve(matrix, objective_gradient)

    np.testing.assert_allclose(
        result.solution,
        expected_transpose,
        rtol=2.0e-15,
        atol=2.0e-15,
    )
    assert np.max(np.abs(result.solution - incorrect_forward)) > 1.0e-2
    assert result.diagnostics.solve_count == 1
    assert result.diagnostics.maximum_residual < 1.0e-14
    assert not result.solution.flags.writeable


@pytest.mark.parametrize(
    "residual_tolerance",
    [True, np.bool_(False), "1e-9", 1.0e-9 + 0.0j, np.array(1.0e-9)],
)
def test_generic_adjoint_rejects_non_real_scalar_tolerances(
    residual_tolerance,
):
    problem = _NonsymmetricAdjointProblem(np.eye(2, dtype=np.float64))
    with pytest.raises(
        TypeError,
        match="residual_tolerance must be a real number",
    ):
        solve_scalar_adjoint(
            problem,
            np.ones(2, dtype=np.float64),
            residual_tolerance=residual_tolerance,
        )


@pytest.mark.parametrize("residual_tolerance", [0.0, -1.0, np.nan, np.inf])
def test_generic_adjoint_rejects_invalid_real_tolerances(residual_tolerance):
    problem = _NonsymmetricAdjointProblem(np.eye(2, dtype=np.float64))
    with pytest.raises(
        ValueError,
        match="residual_tolerance must be finite and positive",
    ):
        solve_scalar_adjoint(
            problem,
            np.ones(2, dtype=np.float64),
            residual_tolerance=residual_tolerance,
        )


def test_objective_ao_potential_matches_independent_torch_autograd(
    zvector_algebra_case,
    objective_ao_potential,
    rhf_adjoint,
):
    case = zvector_algebra_case
    sensitivity = case.method.correction_sensitivity()
    chain_rule_potential = np.einsum(
        "ap,apij->ij",
        sensitivity,
        case.method.dq_dP(),
    )
    linear_sensitivity = (
        case.model.linear.weight.detach().cpu().numpy()
        / (case.model.input_scale.detach().cpu().numpy() + SCALE_EPS)
    )
    linear_sensitivity = np.broadcast_to(
        linear_sensitivity,
        sensitivity.shape,
    )

    assert objective_ao_potential.dtype == np.dtype(np.float64)
    assert np.max(np.abs(linear_sensitivity)) > 1.0e-3
    assert np.max(np.abs(sensitivity - linear_sensitivity)) > 1.0e-3
    np.testing.assert_allclose(
        chain_rule_potential,
        objective_ao_potential,
        rtol=3.0e-13,
        atol=3.0e-13,
    )
    np.testing.assert_allclose(
        rhf_adjoint.objective_ao_potential,
        objective_ao_potential,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        objective_ao_potential,
        objective_ao_potential.T,
        rtol=0.0,
        atol=2.0e-14,
    )


def test_bilateral_occupied_virtual_rhs_matches_density_variation(
    zvector_algebra_case,
    objective_ao_potential,
    rhf_adjoint,
):
    reference = zvector_algebra_case.reference
    coefficient = np.asarray(reference.mo_coeff)
    occupations = np.asarray(reference.mo_occ)
    occupied = occupations > 0
    virtual = occupations == 0
    occupied_coefficients = coefficient[:, occupied]
    virtual_coefficients = coefficient[:, virtual]
    occupied_occupations = occupations[occupied]
    expected_rhs = np.einsum(
        "pa,qi,pq,i->ai",
        virtual_coefficients,
        occupied_coefficients,
        objective_ao_potential + objective_ao_potential.T,
        occupied_occupations,
    )
    generator = np.random.default_rng(20260820)
    trial_amplitude = generator.normal(size=expected_rhs.shape)
    coefficient_variation = virtual_coefficients @ trial_amplitude
    density_variation = np.einsum(
        "pi,qi,i->pq",
        coefficient_variation,
        occupied_coefficients,
        occupied_occupations,
    )
    density_variation = density_variation + density_variation.T
    density_contraction = np.einsum(
        "pq,pq->",
        objective_ao_potential,
        density_variation,
    )
    orbital_contraction = np.einsum(
        "ai,ai->",
        expected_rhs,
        trial_amplitude,
    )

    np.testing.assert_allclose(
        rhf_adjoint.objective_orbital_gradient,
        expected_rhs,
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        orbital_contraction,
        density_contraction,
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    one_sided_rhs = np.einsum(
        "pa,qi,pq,i->ai",
        virtual_coefficients,
        occupied_coefficients,
        objective_ao_potential,
        occupied_occupations,
    )
    assert np.max(np.abs(expected_rhs - one_sided_rhs)) > 1.0e-3


def test_rhf_zvector_matches_an_independent_ao2mo_transpose_solve(
    zvector_algebra_case,
    rhf_adjoint,
):
    operator = zvector_algebra_case.independent_operator
    objective_gradient = rhf_adjoint.objective_orbital_gradient.reshape(-1)
    expected = np.linalg.solve(operator.T, objective_gradient).reshape(
        rhf_adjoint.objective_orbital_gradient.shape
    )

    np.testing.assert_allclose(
        operator,
        operator.T,
        rtol=0.0,
        atol=3.0e-14,
    )
    np.testing.assert_allclose(
        rhf_adjoint.zvector,
        expected,
        rtol=3.0e-13,
        atol=3.0e-13,
    )
    assert rhf_adjoint.diagnostics.solve_count == 1
    assert rhf_adjoint.diagnostics.response_dimension == operator.shape[0]
    assert rhf_adjoint.diagnostics.maximum_residual < 1.0e-10


def test_metric_formula_and_response_partition_match_the_direct_oracle(
    zvector_algebra_case,
    objective_ao_potential,
    rhf_adjoint,
):
    case = zvector_algebra_case
    reference = case.reference
    response = case.direct_response
    coefficient = np.asarray(reference.mo_coeff)
    occupations = np.asarray(reference.mo_occ)
    occupied = occupations > 0
    occupied_coefficients = coefficient[:, occupied]
    occupied_occupations = occupations[occupied]
    overlap_occupied = np.einsum(
        "pi,bxpq,qj->bxij",
        occupied_coefficients,
        response.overlap_derivative,
        occupied_coefficients,
    )
    objective_occupied = (
        occupied_coefficients.T
        @ objective_ao_potential
        @ occupied_coefficients
    )
    direct_metric_contraction = np.einsum(
        "pq,bxpq->bx",
        objective_ao_potential,
        response.density_response_metric,
    )
    manual_objective_metric = -0.5 * np.einsum(
        "i,bxji,ji->bx",
        occupied_occupations,
        overlap_occupied,
        objective_occupied + objective_occupied.T,
    )
    adjoint_potential_occupied = (
        occupied_coefficients.T
        @ rhf_adjoint.adjoint_ao_potential
        @ occupied_coefficients
    )
    manual_adjoint_metric = np.einsum(
        "bxij,ij->bx",
        overlap_occupied,
        0.5 * adjoint_potential_occupied,
    )
    manual_complete_metric = manual_objective_metric + manual_adjoint_metric
    direct_occupied_virtual = np.einsum(
        "pq,bxpq->bx",
        objective_ao_potential,
        response.density_response_occupied_virtual,
    )
    direct_complete_response = np.einsum(
        "pq,bxpq->bx",
        objective_ao_potential,
        response.density_response,
    )

    assert np.max(np.abs(direct_metric_contraction)) > 1.0e-3
    np.testing.assert_allclose(
        manual_objective_metric,
        direct_metric_contraction,
        rtol=3.0e-13,
        atol=3.0e-13,
    )
    np.testing.assert_allclose(
        rhf_adjoint.correction_gradient_metric,
        manual_objective_metric,
        rtol=3.0e-13,
        atol=3.0e-13,
    )
    np.testing.assert_allclose(
        rhf_adjoint.correction_gradient_adjoint_metric,
        manual_adjoint_metric,
        rtol=3.0e-13,
        atol=3.0e-13,
    )
    np.testing.assert_allclose(
        rhf_adjoint.correction_gradient_occupied_virtual,
        rhf_adjoint.correction_gradient_adjoint_nuclear
        + rhf_adjoint.correction_gradient_adjoint_metric,
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        rhf_adjoint.correction_gradient_occupied_virtual,
        direct_occupied_virtual,
        rtol=3.0e-12,
        atol=3.0e-12,
    )
    np.testing.assert_allclose(
        rhf_adjoint.correction_gradient_metric
        + rhf_adjoint.correction_gradient_adjoint_metric,
        manual_complete_metric,
        rtol=3.0e-13,
        atol=3.0e-13,
    )
    np.testing.assert_allclose(
        rhf_adjoint.correction_gradient_response,
        rhf_adjoint.correction_gradient_metric
        + rhf_adjoint.correction_gradient_occupied_virtual,
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        rhf_adjoint.correction_gradient_response,
        direct_complete_response,
        rtol=3.0e-12,
        atol=3.0e-12,
    )
