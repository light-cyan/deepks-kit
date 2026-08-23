from dataclasses import replace

import numpy as np
import pytest

import deepks.deephf.pyscf_uhf as pyscf_uhf
from deepks.deephf import (
    UHFAdjoint,
    UHFAdjointAdapter,
    UHFAdjointDiagnostics,
    UHFAdjointError,
    UHFDeePHF,
    UHFDeePHFGradients,
    UHFDeePHFZVectorGradients,
    UHFResponseAdapter,
    ScalarAdjointProblem,
)


def _assert_driver_cleared(driver):
    for name in (
        "adjoint_result",
        "descriptor_diagnostics",
        "reference_gradient",
        "dq_dR_explicit_spin",
        "dq_dR_explicit",
        "correction_gradient_explicit_spin",
        "correction_gradient_metric_spin",
        "correction_gradient_adjoint_nuclear_spin",
        "correction_gradient_adjoint_metric_spin",
        "correction_gradient_occupied_virtual_spin",
        "correction_gradient_response_spin",
        "correction_gradient_spin",
        "correction_gradient_explicit",
        "correction_gradient_metric",
        "correction_gradient_adjoint_nuclear",
        "correction_gradient_adjoint_metric",
        "correction_gradient_occupied_virtual",
        "correction_gradient_response",
        "correction_gradient",
        "de_full",
        "de",
    ):
        assert getattr(driver, name, None) is None


def test_public_types_and_explicit_backend_dispatch(uhf_oracle_case):
    method = uhf_oracle_case.method
    direct = method.nuc_grad_method()
    zvector = method.nuc_grad_method(backend="zvector")

    assert type(direct) is UHFDeePHFGradients
    assert direct.backend == "direct"
    assert type(zvector) is UHFDeePHFZVectorGradients
    assert zvector.backend == "zvector"
    adjoint = method.adjoint()
    assert type(adjoint) is UHFAdjoint
    assert type(adjoint.diagnostics) is UHFAdjointDiagnostics


def test_direct_and_adjoint_option_namespaces_are_independent(uhf_oracle_case):
    method = UHFDeePHF(
        uhf_oracle_case.reference,
        uhf_oracle_case.model,
        projector_basis=uhf_oracle_case.method._descriptor.projector_basis,
        response_options={"cphf_tolerance": 1.0e-12},
        adjoint_options={"residual_tolerance": 2.0e-9},
    )

    assert method.nuc_grad_method().response_options == {}
    assert method.nuc_grad_method(backend="zvector").adjoint_options == {}
    assert np.isfinite(method.gradient(backend="zvector")).all()
    with pytest.raises(ValueError, match="unsupported direct backend options"):
        method.nuc_grad_method(
            backend="direct",
            objective_symmetry_tolerance=1.0e-10,
        )
    with pytest.raises(ValueError, match="unsupported zvector backend options"):
        method.nuc_grad_method(
            backend="zvector",
            cphf_tolerance=1.0e-12,
        )
    with pytest.raises(ValueError, match="must be 'direct' or 'zvector'"):
        method.nuc_grad_method(backend="automatic")


def test_zvector_uses_one_adjoint_solve_and_no_direct_response_path(
    uhf_oracle_case,
    monkeypatch,
):
    method = uhf_oracle_case.method
    solve_calls = 0
    problems = []
    original = pyscf_uhf.solve_scalar_adjoint

    def counted(*args, **kwargs):
        nonlocal solve_calls
        solve_calls += 1
        problems.append(args[0])
        return original(*args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("the UHF Z-vector path entered a direct response API")

    monkeypatch.setattr(pyscf_uhf, "solve_scalar_adjoint", counted)
    monkeypatch.setattr(pyscf_uhf.ucphf, "solve", forbidden)
    monkeypatch.setattr(UHFResponseAdapter, "solve", forbidden)
    monkeypatch.setattr(UHFDeePHF, "response", forbidden)
    monkeypatch.setattr(UHFDeePHF, "first_order_density", forbidden)
    monkeypatch.setattr(UHFDeePHF, "first_order_spin_density", forbidden)
    monkeypatch.setattr(UHFDeePHF, "dq_dR_response", forbidden)
    monkeypatch.setattr(UHFDeePHF, "dq_dR_relaxed", forbidden)
    monkeypatch.setattr(UHFDeePHFGradients, "_kernel", forbidden)

    driver = method.nuc_grad_method(backend="zvector")
    gradient = driver.kernel()

    assert solve_calls == 1
    assert len(problems) == 1
    assert isinstance(problems[0], ScalarAdjointProblem)
    assert np.isfinite(gradient).all()
    assert driver.adjoint_diagnostics.solve_count == 1
    assert not hasattr(driver, "response_result")
    assert not hasattr(driver, "dq_dR_relaxed")








def test_adjoint_failure_never_falls_back_and_clears_results(
    uhf_oracle_case,
    monkeypatch,
):
    method = uhf_oracle_case.method
    direct_calls = 0

    def failed_adjoint(*args, **kwargs):
        raise UHFAdjointError("injected coupled adjoint failure")

    def counted_direct(*args, **kwargs):
        nonlocal direct_calls
        direct_calls += 1
        raise AssertionError("direct fallback is forbidden")

    monkeypatch.setattr(UHFAdjointAdapter, "solve", failed_adjoint)
    monkeypatch.setattr(UHFResponseAdapter, "solve", counted_direct)
    driver = method.nuc_grad_method(backend="zvector")

    with pytest.raises(UHFAdjointError, match="injected coupled"):
        driver.kernel()
    assert direct_calls == 0
    _assert_driver_cleared(driver)


def test_force_sign_selected_atoms_and_scanner_rejection(uhf_oracle_case):
    method = uhf_oracle_case.method
    full = method.gradient(backend="zvector")
    selected = method.nuc_grad_method(backend="zvector").kernel(
        atmlst=(np.int64(2), 0)
    )

    np.testing.assert_allclose(
        method.forces(backend="zvector"),
        -full,
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        selected,
        full[[2, 0]],
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    with pytest.raises(UHFAdjointError, match="does not provide a gradient scanner"):
        method.nuc_grad_method(backend="zvector").as_scanner()


def test_direct_backend_remains_numerically_unchanged(uhf_oracle_case):
    method = uhf_oracle_case.method

    np.testing.assert_allclose(
        method.gradient(),
        uhf_oracle_case.gradient,
        rtol=0.0,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        method.gradient(backend="direct"),
        uhf_oracle_case.gradient,
        rtol=0.0,
        atol=2.0e-12,
    )
