from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch
from pyscf import gto, scf

import deepks.deephf.adjoint as adjoint_module
from deepks.deephf import DeePHF, DeePHFCapabilityError
from deepks.deephf.pyscf_rhf import (
    RHFAdjointAdapter,
    RHFAdjointError,
    adjoint_integrity_fingerprint,
)
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


def test_method_direct_and_adjoint_option_namespaces_are_independent(
    zvector_algebra_case,
):
    case = zvector_algebra_case
    direct_configured = DeePHF(
        case.reference,
        deepcopy(case.model),
        projector_basis=case.model._pbas,
        response_options={"cphf_tolerance": 1.0e-11},
    )
    zvector_configured = DeePHF(
        case.reference,
        deepcopy(case.model),
        projector_basis=case.model._pbas,
        adjoint_options={"objective_symmetry_tolerance": 1.0e-10},
    )

    zvector_gradient = direct_configured.gradient(backend="zvector")
    direct_gradient = zvector_configured.gradient(backend="direct")

    assert np.isfinite(zvector_gradient).all()
    assert np.isfinite(direct_gradient).all()


def test_adjoint_options_require_a_mapping_at_construction(
    zvector_algebra_case,
):
    case = zvector_algebra_case
    with pytest.raises(TypeError, match="adjoint_options must be a mapping"):
        DeePHF(
            case.reference,
            deepcopy(case.model),
            projector_basis=case.model._pbas,
            adjoint_options=[("objective_symmetry_tolerance", 1.0e-10)],
        )


@pytest.mark.parametrize(
    ("configured_namespace", "match"),
    [
        (
            "direct",
            "unsupported direct backend options: objective_symmetry_tolerance",
        ),
        (
            "zvector",
            "unsupported zvector backend options: cphf_tolerance",
        ),
    ],
)
def test_invalid_method_options_fail_in_their_own_backend_namespace(
    zvector_algebra_case,
    configured_namespace,
    match,
):
    case = zvector_algebra_case
    constructor_options = (
        {
            "response_options": {
                "objective_symmetry_tolerance": 1.0e-10,
            }
        }
        if configured_namespace == "direct"
        else {"adjoint_options": {"cphf_tolerance": 1.0e-11}}
    )
    method = DeePHF(
        case.reference,
        deepcopy(case.model),
        projector_basis=case.model._pbas,
        **constructor_options,
    )

    with pytest.raises(ValueError, match=match):
        method.nuc_grad_method(backend=configured_namespace)


def _corrupted_solver(kind, original_solve):
    if kind == "residual":
        def solve(matrix, rhs, **options):
            solution, info = original_solve(matrix, rhs, **options)
            return solution + 1.0e-3, info

        return solve
    if kind == "nonfinite":
        def solve(_matrix, rhs, **_options):
            return np.full_like(rhs, np.nan), 0

        return solve

    def solve(_matrix, _rhs, **_options):
        raise RuntimeError("injected adjoint failure")

    return solve


@pytest.mark.parametrize(
    ("corruption", "match"),
    [
        ("residual", "adjoint solver residual exceeds tolerance"),
        ("nonfinite", "adjoint solution must be finite"),
        ("hard", "matrix-free GMRES adjoint solver raised an error"),
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
    original_solve = adjoint_module.gmres

    with monkeypatch.context() as patch:
        patch.setattr(
            adjoint_module,
            "gmres",
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
    exact = RHFAdjointAdapter(
        zvector_algebra_case.reference
    ).validate_response_operator_exact()
    _dimension, minimum, _maximum, condition, _symmetry = exact

    with pytest.raises(
        DeePHFCapabilityError,
        match="response operator is unstable or singular",
    ):
        RHFAdjointAdapter(
            zvector_algebra_case.reference,
            operator_stability_tolerance=(
                minimum * 1.01
            ),
        ).validate_response_operator_exact()
    condition_limit = max(
        1.0 + 1.0e-8,
        condition * 0.5,
    )
    assert condition_limit < condition
    with pytest.raises(
        DeePHFCapabilityError,
        match="response operator is ill conditioned",
    ):
        RHFAdjointAdapter(
            zvector_algebra_case.reference,
            operator_condition_tolerance=condition_limit,
        ).validate_response_operator_exact()


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


def test_mutating_a_shared_projector_basis_rejects_stale_energy_and_zvector(
    zvector_algebra_case,
):
    case = zvector_algebra_case
    shared_basis = deepcopy(case.model._pbas)
    model = CorrNet(
        input_dim=4,
        hidden_sizes=(3,),
        actv_fn="tanh",
        use_resnet=False,
        proj_basis=shared_basis,
    ).double()
    model.load_state_dict(case.model.state_dict())
    model.eval()
    method = DeePHF(
        case.reference,
        model,
        projector_basis=shared_basis,
    )
    baseline_energy = method.kernel()
    driver = method.nuc_grad_method(backend="zvector").run()
    baseline_gradient = driver.de_full.copy()
    descriptor_basis = deepcopy(method._descriptor.projector_basis)

    assert model._pbas is shared_basis
    assert method._descriptor.projector_basis is not shared_basis
    shared_basis[0][1][0] *= 1.6
    assert method._descriptor.projector_basis == descriptor_basis

    with pytest.raises(
        DeePHFCapabilityError,
        match="projector metadata does not match projector_basis",
    ):
        method.kernel()
    with pytest.raises(
        DeePHFCapabilityError,
        match="projector metadata does not match projector_basis",
    ):
        driver.kernel()
    assert all(getattr(driver, name) is None for name in DRIVER_RESULT_FIELDS)

    fresh_method = DeePHF(
        case.reference,
        model,
        projector_basis=shared_basis,
    )
    fresh_energy = fresh_method.kernel()
    fresh_gradient = fresh_method.gradient(backend="zvector")

    assert not np.isclose(fresh_energy, baseline_energy, rtol=0.0, atol=1.0e-10)
    assert not np.allclose(
        fresh_gradient,
        baseline_gradient,
        rtol=0.0,
        atol=1.0e-9,
    )


@pytest.mark.parametrize("state", ["projector_basis", "overlap_cache"])
def test_mutating_bound_descriptor_state_is_rejected_before_evaluation(
    zvector_algebra_case,
    state,
):
    case = zvector_algebra_case
    model = deepcopy(case.model)
    method = DeePHF(
        case.reference,
        model,
        projector_basis=model._pbas,
    )
    driver = method.nuc_grad_method(backend="zvector").run()

    if state == "projector_basis":
        method._descriptor.projector_basis[0][1][0] *= 1.1
    else:
        method._descriptor.overlap_shells[0][0, 0, 0] += 1.0e-4

    with pytest.raises(
        DeePHFCapabilityError,
        match="DeePHF scientific state changed",
    ):
        method.correction_energy()
    with pytest.raises(
        DeePHFCapabilityError,
        match="DeePHF scientific state changed",
    ):
        driver.kernel()
    assert all(getattr(driver, name) is None for name in DRIVER_RESULT_FIELDS)


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


@pytest.mark.parametrize(
    "control",
    [
        "residual_tolerance",
        "orbital_gap_tolerance",
        "operator_stability_tolerance",
        "operator_condition_tolerance",
        "operator_symmetry_tolerance",
        "objective_symmetry_tolerance",
    ],
)
@pytest.mark.parametrize(
    "invalid_value",
    [True, np.bool_(False), "1e-9", 1.0e-9 + 0.0j, np.array(1.0e-9)],
)
def test_adjoint_real_controls_reject_non_real_scalar_types(
    zvector_algebra_case,
    control,
    invalid_value,
):
    with pytest.raises(
        ValueError,
        match=rf"adjoint {control} must be a real numeric scalar",
    ):
        RHFAdjointAdapter(
            zvector_algebra_case.reference,
            **{control: invalid_value},
        )


@pytest.mark.parametrize("invalid_value", [True, np.bool_(False), 4.0, "4"])
def test_adjoint_dimension_limit_keeps_the_strict_integer_gate(
    zvector_algebra_case,
    invalid_value,
):
    with pytest.raises(ValueError, match="operator_dimension_limit must be an integer"):
        RHFAdjointAdapter(
            zvector_algebra_case.reference,
            operator_dimension_limit=invalid_value,
        )


def _fresh_reference(case, displacement=0.0):
    molecule = deepcopy(case.reference.mol)
    coordinates = np.asarray(molecule.atom_coords(unit="Bohr")).copy()
    coordinates[0, 0] += displacement
    molecule.set_geom_(coordinates, unit="Bohr")
    reference = scf.RHF(molecule)
    reference.conv_tol = 1.0e-13
    reference.conv_tol_grad = 1.0e-10
    reference.conv_tol_cpscf = 1.0e-12
    reference.max_cycle = 100
    reference.kernel(dm0=None)
    assert reference.converged
    return reference


def _fresh_method(case, displacement=0.0):
    reference = _fresh_reference(case, displacement=displacement)
    model = deepcopy(case.model)
    return DeePHF(
        reference,
        model,
        projector_basis=model._pbas,
    )


def test_reference_science_guard_ignores_common_origin_scratch(
    zvector_algebra_case,
):
    method = _fresh_method(zvector_algebra_case)
    with method.mol.with_common_origin((0.31, -0.27, 0.19)):
        gradient = method.gradient(backend="zvector")
    assert np.isfinite(gradient).all()


def test_replaced_model_forward_fails_and_clears_the_driver(
    zvector_algebra_case,
):
    case = zvector_algebra_case
    method = _fresh_method(case)
    driver = method.nuc_grad_method(backend="zvector").run()
    model = method.model
    original_forward = model.forward

    def replacement_forward(values):
        return original_forward(values)

    model.forward = replacement_forward
    with pytest.raises(
        DeePHFCapabilityError,
        match="forward implementation was replaced",
    ):
        driver.kernel()
    assert all(getattr(driver, name) is None for name in DRIVER_RESULT_FIELDS)


def test_standalone_zvector_rejects_paired_mode_restoring_hooks_before_forward(
    zvector_algebra_case,
):
    method = _fresh_method(zvector_algebra_case)
    calls = []

    def enable_training(module, _inputs):
        calls.append("pre")
        module.train()

    def restore_evaluation(module, _inputs, output):
        calls.append("post")
        module.eval()
        return output

    pre_hook = method.model.register_forward_pre_hook(enable_training)
    post_hook = method.model.register_forward_hook(restore_evaluation)
    try:
        with pytest.raises(
            DeePHFCapabilityError,
            match="cannot contain module execution hooks",
        ):
            method.gradient(backend="zvector")
    finally:
        pre_hook.remove()
        post_hook.remove()
    assert calls == []


def test_standalone_zvector_rejects_a_training_submodule(
    zvector_algebra_case,
):
    method = _fresh_method(zvector_algebra_case)
    method.model.linear.train()
    with pytest.raises(
        DeePHFCapabilityError,
        match="must remain in evaluation mode",
    ):
        method.gradient(backend="zvector")






def test_nonlinear_corrnet_descriptor_fd_audit_preserves_one_adjoint_solve(
    zvector_algebra_case,
    monkeypatch,
):
    method = _fresh_method(zvector_algebra_case)
    solve_count = 0
    original_solve = adjoint_module.gmres

    def counted_solve(matrix, rhs, **options):
        nonlocal solve_count
        solve_count += 1
        return original_solve(matrix, rhs, **options)

    monkeypatch.setattr(adjoint_module, "gmres", counted_solve)
    with torch.no_grad():
        gradient = method.gradient(backend="zvector")

    assert np.isfinite(gradient).all()
    assert solve_count == 1


@pytest.mark.parametrize(
    ("field", "replacement_value", "match"),
    [
        ("zvector", np.zeros((1, 1), dtype=np.float64), "has shape"),
        (
            "objective_ao_potential",
            np.zeros((7, 7), dtype=np.float32),
            "must use real numpy.float64",
        ),
        (
            "correction_gradient_response",
            np.zeros((3, 3), dtype=np.float64),
            "must be immutable",
        ),
    ],
)
def test_adjoint_audit_rejects_resealed_shape_dtype_and_mutability_forgery(
    zvector_algebra_case,
    field,
    replacement_value,
    match,
):
    case = zvector_algebra_case
    sensitivity = case.method.correction_sensitivity()
    expected_objective = case.method._correction_ao_potential(sensitivity)
    adjoint = RHFAdjointAdapter(case.reference).solve(expected_objective)
    forged = replace(adjoint, **{field: replacement_value})
    forged = replace(forged, integrity_fingerprint="")
    forged = replace(
        forged,
        integrity_fingerprint=adjoint_integrity_fingerprint(forged),
    )

    with pytest.raises(RHFAdjointError, match=match):
        RHFAdjointAdapter(case.reference).audit_adjoint(
            forged,
            expected_objective,
        )


def test_adjoint_audit_rejects_coordinated_resealed_gradient_forgery(
    zvector_algebra_case,
):
    case = zvector_algebra_case
    sensitivity = case.method.correction_sensitivity()
    expected_objective = case.method._correction_ao_potential(sensitivity)
    adjoint = RHFAdjointAdapter(case.reference).solve(expected_objective)
    delta = np.full((case.reference.mol.natm, 3), 2.0e-3)

    def immutable(value):
        value = np.ascontiguousarray(value)
        return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)

    forged = replace(
        adjoint,
        correction_gradient_metric=immutable(
            adjoint.correction_gradient_metric + delta
        ),
        correction_gradient_occupied_virtual=immutable(
            adjoint.correction_gradient_occupied_virtual - delta
        ),
    )
    forged = replace(forged, integrity_fingerprint="")
    forged = replace(
        forged,
        integrity_fingerprint=adjoint_integrity_fingerprint(forged),
    )

    with pytest.raises(
        RHFAdjointError,
        match="correction_gradient_metric is inconsistent",
    ):
        RHFAdjointAdapter(case.reference).audit_adjoint(
            forged,
            expected_objective,
        )


def test_zvector_driver_public_provenance_properties_are_read_only(
    zvector_algebra_case,
):
    driver = zvector_algebra_case.method.nuc_grad_method(backend="zvector")
    for name, value in (
        ("base", object()),
        ("mol", object()),
        ("backend", "direct"),
    ):
        with pytest.raises(AttributeError):
            setattr(driver, name, value)


@pytest.mark.parametrize("corruption", ["base", "mol", "backend"])
def test_zvector_driver_rejects_corrupted_internal_provenance(
    zvector_algebra_case,
    corruption,
):
    case = zvector_algebra_case
    driver = case.method.nuc_grad_method(backend="zvector")
    if corruption == "base":
        driver._base = _fresh_method(case, displacement=0.02)
    elif corruption == "mol":
        driver._mol = deepcopy(case.reference.mol)
    else:
        driver._backend = "direct"

    with pytest.raises(RHFAdjointError, match="driver binding is invalid"):
        driver.kernel()
    assert all(getattr(driver, name) is None for name in DRIVER_RESULT_FIELDS)
