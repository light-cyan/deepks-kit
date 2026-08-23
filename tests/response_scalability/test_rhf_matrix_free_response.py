import numpy as np
import pytest
import torch
from pyscf import gto, scf

from deepks.deephf import DeePHF, RHFBlockedResponseSummary
from deepks.deephf.adjoint import solve_scalar_adjoint
from deepks.deephf.pyscf_rhf import _RHFLinearResponseCore
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


def test_rhf_zvector_uses_matrix_free_audit_above_the_dense_debug_limit(
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
