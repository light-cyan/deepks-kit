from copy import deepcopy

import numpy as np
import pytest
import torch
from pyscf import gto, scf

import deepks.deephf.adjoint as adjoint_module
from deepks.deephf import DeePHF, DeePHFCapabilityError
from deepks.deephf.pyscf_rhf import RHFAdjointAdapter, RHFAdjointError
from deepks.descriptor import DescriptorDifferentiabilityError
from deepks.model.model import CorrNet


DRIVER_RESULT_FIELDS = (
    "adjoint_result",
    "descriptor_diagnostics",
    "reference_gradient",
    "dq_dR_explicit",
    "correction_gradient_explicit",
    "correction_gradient_metric",
    "correction_gradient_adjoint_nuclear",
    "correction_gradient_adjoint_metric",
    "correction_gradient_occupied_virtual",
    "correction_gradient_response",
    "correction_gradient",
    "de_full",
    "de",
)


@pytest.mark.parametrize("backend", [None, "", "Z", "rhf_zvector"])
def test_unknown_backend_is_rejected(zvector_algebra_case, backend):
    with pytest.raises(
        ValueError,
        match="gradient backend must be 'direct' or 'zvector'",
    ):
        zvector_algebra_case.method.nuc_grad_method(backend=backend)


@pytest.mark.parametrize(
    ("backend", "options", "match"),
    [
        (
            "zvector",
            {"cphf_tolerance": 1.0e-12},
            "unsupported zvector backend options: cphf_tolerance",
        ),
        (
            "direct",
            {"objective_symmetry_tolerance": 1.0e-12},
            "unsupported direct backend options: objective_symmetry_tolerance",
        ),
        (
            "zvector",
            {"fallback": "direct"},
            "unsupported zvector backend options: fallback",
        ),
    ],
)
def test_backend_specific_options_are_strictly_namespaced(
    zvector_algebra_case,
    backend,
    options,
    match,
):
    with pytest.raises(ValueError, match=match):
        zvector_algebra_case.method.nuc_grad_method(
            backend=backend,
            **options,
        )


def _corrupted_solver(kind, original_solve):
    if kind == "residual":
        def solve(matrix, rhs):
            return original_solve(matrix, rhs) + 1.0e-3

        return solve
    if kind == "nonfinite":
        def solve(_matrix, rhs):
            return np.full_like(rhs, np.nan)

        return solve

    def solve(_matrix, _rhs):
        raise np.linalg.LinAlgError("injected adjoint failure")

    return solve


@pytest.mark.parametrize(
    ("corruption", "match"),
    [
        ("residual", "adjoint residual exceeds tolerance"),
        ("nonfinite", "adjoint solution must be finite"),
        ("hard", "dense transpose adjoint solve failed"),
    ],
)
def test_corrupted_adjoint_solver_fails_without_fallback_and_clears_results(
    zvector_algebra_case,
    monkeypatch,
    corruption,
    match,
):
    method = zvector_algebra_case.method
    driver = method.nuc_grad_method(backend="zvector").run()
    assert all(getattr(driver, name) is not None for name in DRIVER_RESULT_FIELDS)
    original_solve = np.linalg.solve

    with monkeypatch.context() as patch:
        patch.setattr(
            adjoint_module.np.linalg,
            "solve",
            _corrupted_solver(corruption, original_solve),
        )
        with pytest.raises(RHFAdjointError, match=match):
            driver.kernel()

    assert all(getattr(driver, name) is None for name in DRIVER_RESULT_FIELDS)
    direct = method.nuc_grad_method(backend="direct").kernel()
    assert np.isfinite(direct).all()


def test_operator_stability_and_condition_number_gates_are_independent(
    zvector_algebra_case,
):
    method = zvector_algebra_case.method
    baseline = method.adjoint()
    diagnostics = baseline.diagnostics

    with pytest.raises(
        DeePHFCapabilityError,
        match="response operator is unstable or singular",
    ):
        method.gradient(
            backend="zvector",
            operator_stability_tolerance=(
                diagnostics.operator_minimum_eigenvalue * 1.01
            ),
        )
    condition_limit = max(
        1.0 + 1.0e-8,
        diagnostics.operator_condition_number * 0.5,
    )
    assert condition_limit < diagnostics.operator_condition_number
    with pytest.raises(
        DeePHFCapabilityError,
        match="response operator is ill conditioned",
    ):
        method.gradient(
            backend="zvector",
            operator_condition_tolerance=condition_limit,
        )


@pytest.mark.parametrize("corruption", ["nonfinite", "asymmetric"])
def test_adjoint_rejects_invalid_objective_ao_potential(
    zvector_algebra_case,
    corruption,
):
    case = zvector_algebra_case
    objective = case.method.correction_ao_potential().copy()
    if corruption == "nonfinite":
        objective[0, 0] = np.nan
        match = "objective potential must be finite"
    else:
        objective[0, 1] += 1.0e-3
        match = "objective potential violates symmetry"

    with pytest.raises(RHFAdjointError, match=match):
        RHFAdjointAdapter(case.reference).solve(objective)


def test_model_failure_propagates_and_clears_a_previously_successful_driver(
    zvector_algebra_case,
):
    case = zvector_algebra_case
    model = deepcopy(case.model)
    method = DeePHF(case.reference, model, projector_basis=model._pbas)
    driver = method.nuc_grad_method(backend="zvector").run()
    assert driver.de_full is not None

    with torch.no_grad():
        model.linear.weight[0, 0] = torch.nan
    with pytest.raises(
        DeePHFCapabilityError,
        match="model parameters and buffers must be finite",
    ):
        driver.kernel()

    assert all(getattr(driver, name) is None for name in DRIVER_RESULT_FIELDS)


def test_projector_model_metadata_mismatch_is_not_deferred_to_a_backend(
    zvector_algebra_case,
):
    case = zvector_algebra_case
    incompatible_basis = [[0, [0.7, 1.0]], [1, [0.3, 1.0]]]

    with pytest.raises(
        DeePHFCapabilityError,
        match="projector metadata does not match projector_basis",
    ):
        DeePHF(
            case.reference,
            deepcopy(case.model),
            projector_basis=incompatible_basis,
        )


def test_nondifferentiable_descriptor_failure_propagates_before_adjoint():
    molecule = gto.M(
        atom="H 0 0 0; H 0 0 1.4",
        basis="sto-3g",
        unit="Bohr",
        symmetry=False,
        cart=False,
        verbose=0,
    )
    reference = scf.RHF(molecule)
    reference.conv_tol = 1.0e-13
    reference.conv_tol_grad = 1.0e-10
    reference.kernel()
    assert reference.converged
    projector_basis = [[1, [1.0, 1.0]]]
    model = CorrNet(
        input_dim=3,
        hidden_sizes=(2,),
        proj_basis=projector_basis,
    ).double()
    with torch.no_grad():
        model.linear.weight[0] = torch.tensor(
            [0.2, 0.25, -0.3],
            dtype=torch.float64,
        )
        model.linear.bias.zero_()
        for parameter in model.densenet.parameters():
            parameter.zero_()
    method = DeePHF(
        reference,
        model.eval(),
        projector_basis=projector_basis,
    )
    driver = method.nuc_grad_method(backend="zvector")

    with pytest.raises(
        DescriptorDifferentiabilityError,
        match="structural zero block sensitivity spread",
    ):
        driver.kernel()

    assert all(getattr(driver, name) is None for name in DRIVER_RESULT_FIELDS)
