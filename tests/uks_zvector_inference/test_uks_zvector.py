from dataclasses import replace
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

import deepks.deephf.pyscf_uhf as pyscf_uhf

from deepks.deephf import (
    DeePHFCapabilityError,
    UKSAdjoint,
    UKSAdjointAdapter,
    UKSAdjointError,
    UKSDeePHF,
    UKSResponseAdapter,
    UKSResponseError,
)


_FIXTURE_PATH = Path(__file__).resolve().parents[1] / "uks_analytic_forces" / "conftest.py"
_SPEC = importlib.util.spec_from_file_location("_deepks_uks_oracle_fixtures", _FIXTURE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("the UKS oracle fixture cannot be loaded")
_FIXTURES = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _FIXTURES
_SPEC.loader.exec_module(_FIXTURES)
uks_case = _FIXTURES.uks_case


def test_zvector_matches_direct_and_fresh_energy_finite_differences(uks_case):
    np.testing.assert_allclose(uks_case.zvector_gradient, uks_case.direct_gradient, rtol=0.0, atol=2.0e-10)
    for step, tolerance in ((1.0e-3, 2.0e-6), (3.0e-4, 3.0e-7), (1.0e-4, 3.0e-7)):
        expected = uks_case.finite_difference("energy", step).reshape(3, 3)
        np.testing.assert_allclose(uks_case.zvector_gradient, expected, rtol=3.0e-6, atol=tolerance)


def test_zvector_has_one_solve_and_complete_grid_partitions(uks_case):
    driver = uks_case.method.nuc_grad_method(backend="zvector")
    driver.kernel()
    adjoint = driver.adjoint_result
    assert type(adjoint) is UKSAdjoint
    assert adjoint.diagnostics.solve_count == 1
    np.testing.assert_allclose(adjoint.correction_gradient_adjoint_nuclear_spin, adjoint.correction_gradient_adjoint_fixed_grid_spin + adjoint.correction_gradient_adjoint_grid_coordinate_spin + adjoint.correction_gradient_adjoint_grid_weight_spin, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(driver.correction_gradient_response, driver.correction_gradient_metric + driver.correction_gradient_occupied_virtual, rtol=0.0, atol=1.0e-12)
    assert np.max(np.abs(adjoint.correction_gradient_adjoint_grid_coordinate)) > 1.0e-6
    assert np.max(np.abs(adjoint.correction_gradient_adjoint_grid_weight)) > 1.0e-6


def test_zvector_does_not_call_direct_response(monkeypatch, uks_case):
    monkeypatch.setattr(UKSResponseAdapter, "solve", lambda self: (_ for _ in ()).throw(AssertionError("direct response was called")))
    gradient = uks_case.method.gradient(backend="zvector")
    np.testing.assert_allclose(gradient, uks_case.direct_gradient, rtol=0.0, atol=2.0e-10)


def test_adjoint_audit_does_not_resolve(monkeypatch, uks_case):
    sensitivity = uks_case.method.correction_sensitivity()
    objective = uks_case.method._correction_ao_potential(sensitivity)
    adapter = UKSAdjointAdapter(uks_case.reference)
    adjoint = adapter.solve(objective)
    monkeypatch.setattr(adapter._core, "solve", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("adjoint was resolved")))
    adapter.audit_adjoint(adjoint, objective)


def test_resealed_adjoint_is_rejected(uks_case):
    sensitivity = uks_case.method.correction_sensitivity()
    objective = uks_case.method._correction_ao_potential(sensitivity)
    adapter = UKSAdjointAdapter(uks_case.reference)
    adjoint = adapter.solve(objective)
    forged = replace(adjoint, correction_gradient_adjoint_grid_weight=np.zeros_like(adjoint.correction_gradient_adjoint_grid_weight), integrity_fingerprint="")
    from deepks.deephf.pyscf_uks import uks_adjoint_integrity_fingerprint

    forged = replace(forged, integrity_fingerprint=uks_adjoint_integrity_fingerprint(forged))
    with pytest.raises(UKSAdjointError):
        adapter.audit_adjoint(forged, objective)


def test_zvector_failure_clears_driver_without_direct_fallback(monkeypatch, uks_case):
    driver = uks_case.method.nuc_grad_method(backend="zvector")
    monkeypatch.setattr(UKSAdjointAdapter, "solve", lambda self, objective: (_ for _ in ()).throw(UKSAdjointError("injected failure")))
    with pytest.raises(UKSAdjointError, match="injected failure"):
        driver.kernel()
    assert driver.de is None
    assert driver.de_full is None
    assert driver.adjoint_result is None


def test_uks_scanner_is_explicitly_unavailable(uks_case):
    with pytest.raises(UKSAdjointError, match="does not provide"):
        uks_case.method.nuc_grad_method(backend="zvector").as_scanner()


def test_foreign_response_is_rejected(uks_case):
    foreign = UKSDeePHF(
        uks_case.reference,
        uks_case.model,
        projector_basis=uks_case.method._descriptor.projector_basis,
    )
    with pytest.raises(UKSResponseError, match="not produced"):
        foreign.first_order_density(response=uks_case.response)


def test_unconverged_reference_fails_before_gradient(monkeypatch, uks_case):
    monkeypatch.setattr(uks_case.reference, "converged", False)
    with pytest.raises(DeePHFCapabilityError):
        uks_case.method.gradient(backend="direct")


def test_operator_condition_gate_is_fail_closed(uks_case):
    with pytest.raises(DeePHFCapabilityError, match="ill conditioned"):
        UKSResponseAdapter(
            uks_case.reference,
            operator_condition_tolerance=100.0,
        ).validate_response_operator_exact()


def test_uks_production_response_and_adjoint_are_matrix_free(
    uks_case,
    monkeypatch,
):
    def forbidden_dense_audit(*_args, **_kwargs):
        raise AssertionError("the UKS response matrix was materialized")

    monkeypatch.setattr(
        pyscf_uhf._UHFLinearResponseCore,
        "_response_operator_matrix_and_diagnostics",
        forbidden_dense_audit,
    )
    response = uks_case.method.response(operator_dimension_limit=1)
    adjoint = uks_case.method.adjoint(operator_dimension_limit=1)

    assert response.diagnostics.response_dimension > 1
    assert response.diagnostics.operator_diagnostics_are_estimates is True
    assert adjoint.diagnostics.iteration_count > 0


def test_direct_failure_clears_results_without_zvector_fallback(monkeypatch, uks_case):
    driver = uks_case.method.nuc_grad_method(backend="direct")
    monkeypatch.setattr(UKSResponseAdapter, "solve", lambda self: (_ for _ in ()).throw(UKSResponseError("injected direct failure")))
    with pytest.raises(UKSResponseError, match="injected direct failure"):
        driver.kernel()
    assert driver.de is None
    assert driver.de_full is None
    assert driver.response_result is None


@pytest.mark.parametrize(
    ("name", "value"),
    [("residual_tolerance", True), ("objective_symmetry_tolerance", "1e-9"), ("operator_dimension_limit", 0)],
)
def test_adjoint_controls_reject_invalid_scalar_domains(uks_case, name, value):
    with pytest.raises((TypeError, ValueError)):
        UKSAdjointAdapter(uks_case.reference, **{name: value})
