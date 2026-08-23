import numpy as np
import pytest
import torch
from pyscf import gto, scf

from deepks.deephf import DeePHF, RHFBlockedResponseSummary
from deepks.deephf.adjoint import (
    scalar_operator_fingerprint,
    solve_scalar_adjoint,
    symmetric_operator_telemetry,
)
from deepks.deephf.pyscf_rhf import RHFResponseAdapter, _RHFLinearResponseCore
from deepks.model.model import CorrNet


PROJECTOR_BASIS = [[0, [0.8, 1.0]]]


class _MatrixFreeProblem:
    def __init__(self, matrix):
        self.matrix = matrix

    @property
    def dimension(self):
        return self.matrix.shape[0]

    def apply(self, vector):
        return self.matrix @ vector

    def apply_transpose(self, vector):
        return self.matrix.T @ vector

    def precondition(self, vector):
        return vector / np.diag(self.matrix)


class _StateFingerprintProblem(_MatrixFreeProblem):
    operator_fingerprint = "1" * 64


class _DiagonalProblem:
    def __init__(self, diagonal):
        self.diagonal = diagonal

    @property
    def dimension(self):
        return self.diagonal.size

    def apply(self, vector):
        return self.diagonal * vector

    def apply_transpose(self, vector):
        return self.apply(vector)


@pytest.fixture(scope="module")
def scalable_rhf_method():
    molecule = gto.M(
        atom=(
            "H 0.0 0.0 0.0; H 0.0 0.0 1.3; "
            "H 0.2 0.1 2.8; H -0.1 0.3 4.4"
        ),
        basis="sto-3g",
        unit="Bohr",
        verbose=0,
    )
    reference = scf.RHF(molecule)
    reference.conv_tol = 1.0e-12
    reference.conv_tol_cpscf = 1.0e-12
    reference.kernel()
    assert reference.converged
    model = CorrNet(
        input_dim=1,
        hidden_sizes=(2,),
        proj_basis=PROJECTOR_BASIS,
    ).double()
    with torch.no_grad():
        model.linear.weight.fill_(0.04)
        model.linear.bias.fill_(0.01)
        for parameter in model.densenet.parameters():
            parameter.zero_()
    method = DeePHF(
        reference,
        model.eval(),
        projector_basis=PROJECTOR_BASIS,
    )
    method.kernel()
    return method


def test_generic_gmres_adjoint_never_requests_a_dense_operator():
    matrix = np.array(
        [[3.1, 0.2, -0.1], [0.4, 2.7, 0.3], [-0.2, 0.1, 1.9]],
        dtype=np.float64,
    )
    objective = np.array([0.3, -0.2, 0.1], dtype=np.float64)
    result = solve_scalar_adjoint(
        _MatrixFreeProblem(matrix),
        objective,
        solver="gmres",
        residual_tolerance=1.0e-11,
    )
    expected = np.linalg.solve(matrix.T, objective)
    np.testing.assert_allclose(result.solution, expected, rtol=0.0, atol=1.0e-11)
    assert result.diagnostics.solver == "scipy.sparse.linalg.gmres(A.T, b)"
    assert result.diagnostics.iteration_count > 0


def test_operator_fingerprint_combines_state_and_action_evidence():
    problem = _StateFingerprintProblem(np.eye(3, dtype=np.float64))
    initial = scalar_operator_fingerprint(problem)
    problem.matrix = np.diag(np.array([1.0, 1.0, 1.5]))
    changed = scalar_operator_fingerprint(problem)

    assert initial != changed


def test_short_lanczos_values_are_explicitly_only_telemetry():
    diagonal = np.concatenate(
        (np.array([-1.0e-8]), np.linspace(1.0, 10.0, 999))
    )
    telemetry = symmetric_operator_telemetry(
        _DiagonalProblem(diagonal)
    )

    assert telemetry.minimum_ritz_value > 0.0
    assert diagonal.min() < 0.0


def test_rhf_zvector_never_uses_the_dense_debug_audit(
    scalable_rhf_method,
    monkeypatch,
):
    def forbidden_dense_operator(*_args, **_kwargs):
        raise AssertionError("the RHF response matrix was materialized")

    monkeypatch.setattr(
        _RHFLinearResponseCore,
        "_response_operator_matrix_and_diagnostics",
        forbidden_dense_operator,
    )
    adjoint = scalable_rhf_method.adjoint(operator_dimension_limit=1)
    assert adjoint.diagnostics.response_dimension == 4
    assert adjoint.diagnostics.response_dimension > 1
    assert adjoint.diagnostics.operator_diagnostics_are_estimates is True
    assert adjoint.diagnostics.iteration_count > 0
    assert adjoint.diagnostics.maximum_solver_residual < 1.0e-9


def test_direct_coordinate_blocks_match_full_response_without_retaining_it(
    scalable_rhf_method,
):
    full = scalable_rhf_method.nuc_grad_method(
        backend="direct",
        operator_dimension_limit=1,
    ).run()
    blocked = scalable_rhf_method.nuc_grad_method(
        backend="direct",
        operator_dimension_limit=1,
        coordinate_block_size=1,
    ).run()
    np.testing.assert_allclose(blocked.de, full.de, rtol=0.0, atol=1.0e-11)
    np.testing.assert_allclose(
        blocked.dq_dR_response,
        full.dq_dR_response,
        rtol=0.0,
        atol=1.0e-11,
    )
    assert type(blocked.response_result) is RHFBlockedResponseSummary
    assert blocked.response_result.coordinate_block_size == 1
    assert blocked.response_result.block_count == scalable_rhf_method.mol.natm
    assert not hasattr(blocked.response_result, "density_response")


def test_selected_atoms_limit_coordinate_response_work(
    scalable_rhf_method,
    monkeypatch,
):
    solved_atoms = []
    original_solve = RHFResponseAdapter.solve

    def counted_solve(self, atom_indices=None):
        solved_atoms.append(tuple(atom_indices))
        return original_solve(self, atom_indices=atom_indices)

    monkeypatch.setattr(RHFResponseAdapter, "solve", counted_solve)
    driver = scalable_rhf_method.nuc_grad_method(
        backend="direct",
        coordinate_block_size=1,
    )
    selected = driver.kernel(atmlst=(3, 1))

    assert solved_atoms == [(3,), (1,)]
    assert selected.shape == (2, 3)
    assert driver.dq_dR_response.shape[:2] == (2, 3)
