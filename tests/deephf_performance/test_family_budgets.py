import numpy as np
import pytest
import torch
from pyscf import gto

from deepks.deephf import build_reference, make_deephf
from deepks.deephf.capabilities import DeePHFCapabilityError
from deepks.model.model import CorrNet


PROJECTOR_BASIS = [[0, [0.8, 1.0]]]


@pytest.fixture(scope="module", params=("rhf", "uhf", "rks", "uks"))
def family_reference(request):
    family = request.param
    unrestricted = family in {"uhf", "uks"}
    molecule = gto.M(
        atom="Li 0 0 0" if unrestricted else "H 0 0 0; H 0 0 1.4",
        basis="sto-3g",
        unit="Bohr",
        spin=1 if unrestricted else 0,
        verbose=0,
    )
    return family, build_reference(molecule, family)


def _model(*, constant):
    model = CorrNet(
        input_dim=1,
        hidden_sizes=(2,),
        proj_basis=PROJECTOR_BASIS,
    ).double()
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.zero_()
        model.linear.bias.fill_(0.17)
        if not constant:
            model.linear.weight.fill_(0.23)
    return model.eval()


def _run(family_reference, backend, *, constant):
    _family, reference = family_reference
    method = make_deephf(
        reference,
        _model(constant=constant),
        projector_basis=PROJECTOR_BASIS,
    )
    with method.calculation():
        method.kernel()
        gradient = method.nuc_grad_method(
            backend=backend,
            retain_details=False,
        ).kernel()
        descriptor = method.descriptor()
    assert np.isfinite(gradient).all()
    assert np.isfinite(descriptor).all()
    return method.operation_counts


def test_constructor_performs_static_validation_without_model_forward(family_reference):
    class CountingModel(torch.nn.Module):
        input_dim = 1

        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(0.2, dtype=torch.float64))
            self.forward_count = 0

        def forward(self, values):
            self.forward_count += 1
            return self.scale * values.sum()

    _family, reference = family_reference
    model = CountingModel()
    method = make_deephf(
        reference,
        model,
        projector_basis=PROJECTOR_BASIS,
    )
    assert model.forward_count == 0
    assert np.isfinite(method.kernel())
    assert model.forward_count == 1


def test_static_float32_model_incompatibility_fails_at_construction(family_reference):
    _family, reference = family_reference
    with pytest.raises(DeePHFCapabilityError, match="must use torch.float64"):
        make_deephf(
            reference,
            _model(constant=False).float(),
            projector_basis=PROJECTOR_BASIS,
        )


@pytest.mark.parametrize("backend", ("direct", "zvector"))
@pytest.mark.parametrize("constant", (False, True), ids=("nonzero", "zero"))
def test_compact_workflow_evaluation_and_solve_budget(
    family_reference, backend, constant
):
    counts = _run(family_reference, backend, constant=constant)
    assert counts["descriptor_evaluations"] == 1
    assert counts["model_forwards"] == 1
    selected_solve = "direct_response_solves" if backend == "direct" else "adjoint_solves"
    other_solve = "adjoint_solves" if backend == "direct" else "direct_response_solves"
    assert counts.get(selected_solve, 0) == (0 if constant else 1)
    assert counts.get(other_solve, 0) == 0
    if not constant:
        return
    assert counts.get("shell_eigenvalue_jacobian_constructions", 0) == 0
    assert counts.get("derivative_overlap_integral_evaluations", 0) == 0
    assert counts.get("response_operator_actions", 0) == 0
    assert counts.get("preconditioner_actions", 0) == 0
