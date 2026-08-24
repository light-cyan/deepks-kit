from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest
import torch

import deepks.deephf.adjoint as adjoint_module
from deepks.deephf import DeePHF
from deepks.deephf.gradient import RHFDeePHFGradients
from deepks.model.model import CorrNet


def test_rhf_compact_selected_gradient_drivers_retain_one_array(
    zvector_algebra_case,
    monkeypatch,
):
    expected = {
        backend: zvector_algebra_case.method.nuc_grad_method(
            backend=backend,
        ).kernel(atmlst=(1,))
        for backend in ("direct", "zvector")
    }
    monkeypatch.setattr(
        zvector_algebra_case.method,
        "dq_dR_explicit",
        lambda *args, **kwargs: pytest.fail("compact execution built a Jacobian"),
    )
    for backend in ("direct", "zvector"):
        driver = zvector_algebra_case.method.nuc_grad_method(
            backend=backend,
            retain_details=False,
        ).run(atmlst=(1,))
        retained_bytes = sum(
            value.nbytes
            for value in vars(driver).values()
            if isinstance(value, np.ndarray)
        )
        assert retained_bytes == driver.de.nbytes
        assert driver.de.shape == (1, 3)
        assert not hasattr(driver, "de_full")
        result_name = "response_result" if backend == "direct" else "adjoint_result"
        assert not hasattr(driver, result_name)
        np.testing.assert_allclose(driver.de, expected[backend], rtol=0.0, atol=1.0e-12)


def _make_constant_model(template, bias):
    model = deepcopy(template)
    with torch.no_grad():
        model.linear.weight.zero_()
        model.linear.bias.fill_(bias)
        for parameter in model.densenet.parameters():
            parameter.zero_()
        model.energy_const.zero_()
    return model.eval()


def _force_checkpoint_metadata(model):
    encoded_projector = json.dumps(
        model._pbas,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return {
        "schema_id": "deepks.deephf.rhf-force-data",
        "schema_version": 1,
        "compatibility_fingerprint": hashlib.sha256(
            b"deterministic P3A force-training contract"
        ).hexdigest(),
        "jacobian_semantics": "dq_dR_relaxed",
        "n_feature": 4,
        "descriptor_definition": "ordered_projected_density_eigenvalues",
        "descriptor_spin_semantics": "spin_summed",
        "descriptor_shell_sizes": [1, 3],
        "projector_sha256": hashlib.sha256(encoded_projector).hexdigest(),
        "reference_family": "RHF",
        "response_backend": "rhf_direct",
    }


def test_zvector_total_gradient_matches_three_step_fresh_total_energy_fd(
    zvector_algebra_case,
):
    case = zvector_algebra_case
    analytic = case.method.gradient(backend="zvector")

    for step, absolute_tolerance in (
        (1.0e-3, 3.0e-6),
        (3.0e-4, 4.0e-7),
        (1.0e-4, 1.0e-7),
    ):
        finite_difference = case.total_energy_finite_difference(step)
        np.testing.assert_allclose(
            analytic,
            finite_difference,
            rtol=3.0e-6,
            atol=absolute_tolerance,
            err_msg=f"central-difference step {step:.1e} Bohr",
        )


def _assert_constant_correction_has_native_zvector_gradient(case, bias):
    model = _make_constant_model(case.model, bias)
    method = DeePHF(
        case.reference,
        model,
        projector_basis=model._pbas,
    )
    method.kernel()
    driver = method.nuc_grad_method(backend="zvector").run()
    native = np.asarray(case.reference.nuc_grad_method().kernel())

    np.testing.assert_allclose(
        method.correction_sensitivity(),
        np.zeros((case.reference.mol.natm, 4)),
        rtol=0.0,
        atol=0.0,
    )
    for field in (
        "correction_gradient_explicit",
        "correction_gradient_metric",
        "correction_gradient_occupied_virtual",
        "correction_gradient_response",
        "correction_gradient",
    ):
        np.testing.assert_allclose(
            getattr(driver, field),
            np.zeros((case.reference.mol.natm, 3)),
            rtol=0.0,
            atol=2.0e-15,
            err_msg=field,
        )
    np.testing.assert_allclose(
        driver.de_full,
        native,
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    return method


def test_zero_correction_reduces_to_native_rhf_gradient(zvector_algebra_case):
    method = _assert_constant_correction_has_native_zvector_gradient(
        zvector_algebra_case,
        bias=0.0,
    )
    np.testing.assert_allclose(method.e_corr, 0.0, rtol=0.0, atol=2.0e-15)


def test_constant_correction_reduces_to_native_rhf_gradient(
    zvector_algebra_case,
):
    method = _assert_constant_correction_has_native_zvector_gradient(
        zvector_algebra_case,
        bias=0.017,
    )
    np.testing.assert_allclose(method.e_corr, 0.051, rtol=0.0, atol=2.0e-15)


def test_zvector_force_sign_and_selected_atom_semantics(zvector_algebra_case):
    method = zvector_algebra_case.method
    complete = method.gradient(backend="zvector")

    np.testing.assert_allclose(
        method.forces(backend="zvector"),
        -complete,
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    driver = method.nuc_grad_method(backend="zvector")
    selected = driver.kernel(atmlst=(np.int64(2), 0))
    np.testing.assert_allclose(
        selected,
        complete[[2, 0]],
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        driver.forces(atmlst=[1]),
        -complete[[1]],
        rtol=2.0e-12,
        atol=2.0e-12,
    )


def test_zvector_backend_never_enters_direct_or_complete_density_paths(
    zvector_algebra_case,
    monkeypatch,
):
    method = zvector_algebra_case.method

    def forbidden(*_args, **_kwargs):
        raise AssertionError("the Z-vector backend entered a direct-response path")

    for name in (
        "response",
        "first_order_density",
        "dq_dR_response",
        "dq_dR_relaxed",
    ):
        monkeypatch.setattr(DeePHF, name, forbidden)
    monkeypatch.setattr(RHFDeePHFGradients, "kernel", forbidden)

    zvector = method.nuc_grad_method(backend="zvector").kernel()

    assert np.isfinite(zvector).all()
    with np.testing.assert_raises_regex(
        AssertionError,
        "direct-response path",
    ):
        method.nuc_grad_method(backend="direct").kernel()


def test_one_scalar_objective_uses_one_matrix_free_linear_solve(
    zvector_algebra_case,
    monkeypatch,
):
    original_solve = adjoint_module.gmres
    calls = 0

    def counted_solve(matrix, rhs, **options):
        nonlocal calls
        calls += 1
        assert matrix.shape == (rhs.size, rhs.size)
        return original_solve(matrix, rhs, **options)

    monkeypatch.setattr(adjoint_module, "gmres", counted_solve)
    driver = zvector_algebra_case.method.nuc_grad_method(
        backend="zvector"
    ).run()
    adjoint = driver.adjoint_result
    occupations = np.asarray(zvector_algebra_case.reference.mo_occ)
    n_occupied = int(np.count_nonzero(occupations > 0))
    n_virtual = int(np.count_nonzero(occupations == 0))

    assert calls == 1
    assert adjoint.objective_orbital_gradient.shape == (
        n_virtual,
        n_occupied,
    )
    assert adjoint.objective_orbital_gradient.size == (
        adjoint.diagnostics.response_dimension
    )
    assert adjoint.diagnostics.solve_count == 1
    assert adjoint.diagnostics.iteration_count > 0
    assert not hasattr(driver, "dq_dR_response")
    assert not hasattr(driver, "dq_dR_relaxed")
    for name in (
        "density_response",
        "first_order_density",
        "dq_dR_response",
        "dq_dR_relaxed",
    ):
        assert not hasattr(adjoint, name)


def test_force_training_checkpoint_model_is_a_valid_zvector_objective(
    zvector_algebra_case,
    tmp_path,
):
    case = zvector_algebra_case
    checkpoint = tmp_path / "p3a-force-model.pth"
    metadata = _force_checkpoint_metadata(case.model)
    case.model.save(checkpoint, force_training=metadata)
    loaded = CorrNet.load(
        checkpoint,
        require_force_metadata=True,
        expected_force_contract_fingerprint=metadata[
            "compatibility_fingerprint"
        ],
    ).double().eval()
    method = DeePHF(case.reference, loaded)

    np.testing.assert_allclose(
        method.gradient(backend="zvector"),
        case.method.gradient(backend="zvector"),
        rtol=0.0,
        atol=0.0,
    )
    assert loaded._checkpoint_extra_info["force_training"][
        "jacobian_semantics"
    ] == "dq_dR_relaxed"
