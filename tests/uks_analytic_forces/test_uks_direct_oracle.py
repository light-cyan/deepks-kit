import numpy as np
import pytest
from pyscf import dft

import deepks.deephf.pyscf_uks as pyscf_uks
from deepks.deephf import UHFResponseAdapter, UKSResponse, UKSResponseAdapter
from deepks.deephf.capabilities import DeePHFCapabilityError


def test_cached_uks_rejects_a_changed_numint_implementation(uks_case, monkeypatch):
    pyscf_uks.validate_uks_reference(uks_case.reference)
    original = dft.numint.NumInt.eval_xc_eff

    def replacement(self, *args, **kwargs):
        return original(self, *args, **kwargs)

    monkeypatch.setattr(dft.numint.NumInt, "eval_xc_eff", replacement)
    with pytest.raises(DeePHFCapabilityError, match="DFT implementation changed"):
        pyscf_uks.validate_uks_reference(uks_case.reference)


@pytest.mark.parametrize(
    ("step", "tolerance"),
    [(1.0e-3, 8.0e-7), (3.0e-4, 2.0e-7), (1.0e-4, 9.0e-7)],
)
def test_spin_resolved_density_response_matches_fresh_uks(uks_case, step, tolerance):
    response = uks_case.response
    np.testing.assert_allclose(response.alpha_density_response, uks_case.finite_difference("alpha_density", step), rtol=3.0e-6, atol=tolerance)
    np.testing.assert_allclose(response.beta_density_response, uks_case.finite_difference("beta_density", step), rtol=3.0e-6, atol=tolerance)
    np.testing.assert_allclose(response.total_density_response, uks_case.finite_difference("density", step), rtol=3.0e-6, atol=tolerance)


def test_selected_response_omits_full_translation_diagnostics(uks_case):
    adapter = UKSResponseAdapter(uks_case.reference)
    response = adapter.solve(atom_indices=(1,))

    assert response.diagnostics.alpha_translation_residual is None
    assert response.diagnostics.beta_translation_residual is None
    assert response.diagnostics.translation_residual is None
    assert adapter.audit_response_equations(response) is None


def test_compact_direct_gradient_builds_two_spin_density_responses(uks_case, monkeypatch):
    original = UHFResponseAdapter._density_from_mo_response
    original_solve = UHFResponseAdapter._solve_orbitals
    coordinate_calls = 0
    solved = False

    def counted_solve(instance, *args):
        nonlocal solved
        result = original_solve(instance, *args)
        solved = True
        return result

    def counted(*args):
        nonlocal coordinate_calls
        coordinate_calls += solved
        return original(*args)

    monkeypatch.setattr(UHFResponseAdapter, "_solve_orbitals", counted_solve)
    monkeypatch.setattr(UHFResponseAdapter, "_density_from_mo_response", staticmethod(counted))
    driver = uks_case.method.nuc_grad_method(retain_details=False)
    driver.kernel(atmlst=(1,))

    assert coordinate_calls == 2


@pytest.mark.parametrize(
    ("step", "tolerance"),
    [(1.0e-3, 8.0e-7), (3.0e-4, 2.0e-7), (1.0e-4, 2.0e-7)],
)
def test_relaxed_descriptor_matches_fresh_uks(uks_case, step, tolerance):
    np.testing.assert_allclose(uks_case.method.dq_dR_relaxed(response=uks_case.response), uks_case.finite_difference("descriptor", step), rtol=3.0e-6, atol=tolerance)


@pytest.mark.parametrize(
    ("step", "tolerance"),
    [(1.0e-3, 2.0e-6), (3.0e-4, 3.0e-7), (1.0e-4, 3.0e-7)],
)
def test_direct_total_gradient_matches_fresh_total_energy(uks_case, step, tolerance):
    expected = uks_case.finite_difference("energy", step).reshape(3, 3)
    np.testing.assert_allclose(uks_case.direct_gradient, expected, rtol=3.0e-6, atol=tolerance)


def test_response_partitions_and_public_audit_are_complete(uks_case):
    response = uks_case.response
    assert type(response) is UKSResponse
    np.testing.assert_allclose(response.total_density_response, response.alpha_density_response + response.beta_density_response, rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(response.total_density_response, response.total_density_response_metric + response.total_density_response_occupied_virtual, rtol=0.0, atol=1.0e-14)
    assert np.max(np.abs(response.xc_hamiltonian_derivative_grid_coordinate_spin)) > 1.0e-3
    assert np.max(np.abs(response.xc_hamiltonian_derivative_grid_weight_spin)) > 1.0e-3
    assert response.diagnostics.maximum_residual < response.diagnostics.residual_tolerance
    UKSResponseAdapter(uks_case.reference).audit_response_equations(response)


def test_zero_and_constant_corrections_reduce_to_native_uks(uks_case):
    import copy
    import torch

    from deepks.deephf import UKSDeePHF
    from deepks.deephf.pyscf_uks import native_uks_gradient

    projector_basis = uks_case.method._descriptor.projector_basis
    native = native_uks_gradient(uks_case.reference)
    zero = UKSDeePHF(uks_case.reference, None, projector_basis=projector_basis)
    np.testing.assert_allclose(zero.gradient(), native, rtol=0.0, atol=1.0e-10)
    constant_model = copy.deepcopy(uks_case.model)
    with torch.no_grad():
        for parameter in constant_model.linear.parameters():
            parameter.zero_()
        for parameter in constant_model.densenet.parameters():
            parameter.zero_()
        constant_model.energy_const.fill_(0.37)
    constant = UKSDeePHF(uks_case.reference, constant_model.eval(), projector_basis=projector_basis)
    np.testing.assert_allclose(constant.gradient(backend="zvector"), native, rtol=0.0, atol=1.0e-10)


def test_uks_public_force_calls_have_single_transaction_budgets(
    uks_case,
):
    method = uks_case.method
    for calculation in (method.dq_dR_relaxed, method.gradient):
        assert np.isfinite(calculation()).all()
        counts = method.operation_counts
        assert counts["science_state_fingerprints"] == 2
        assert counts["descriptor_evaluations"] == 1
        assert counts["projected_density_constructions"] == 1
        assert counts["shell_eigenvalue_jacobian_constructions"] == 1
        assert counts.get("derivative_overlap_integral_evaluations", 0) <= 1
        assert counts["direct_response_solves"] == 1
