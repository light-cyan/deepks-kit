import numpy as np
import pytest

from deepks.deephf import RHFResponseAdapter


def test_selected_response_diagnostics_are_reproducible(rhf_oracle_case):
    adapter = RHFResponseAdapter(rhf_oracle_case.reference)
    response = adapter.solve(atom_indices=(1,))

    assert adapter.audit_response_equations(response) is None


@pytest.mark.parametrize(
    ("step", "absolute_tolerance"),
    [
        pytest.param(1.0e-3, 5.0e-7, id="coarse"),
        pytest.param(3.0e-4, 1.0e-7, id="balanced"),
        pytest.param(1.0e-4, 1.0e-7, id="fine"),
    ],
)
def test_complete_ao_density_response_matches_displaced_rhf(
    rhf_oracle_case,
    step,
    absolute_tolerance,
):
    analytic = rhf_oracle_case.response.density_response
    finite_difference = rhf_oracle_case.finite_difference("density", step)

    assert analytic.shape == (3, 3, 7, 7)
    np.testing.assert_allclose(
        analytic,
        finite_difference,
        rtol=2.0e-6,
        atol=absolute_tolerance,
    )


def test_compact_direct_gradient_builds_density_partitions_once(
    rhf_oracle_case,
    monkeypatch,
):
    original = RHFResponseAdapter._density_from_mo_response
    original_solve = RHFResponseAdapter._solve_orbitals
    coordinate_calls = 0
    solved = False

    def counted_solve(instance, *args):
        nonlocal solved
        result = original_solve(instance, *args)
        solved = True
        return result

    def counted(instance, *args):
        nonlocal coordinate_calls
        coordinate_calls += solved
        return original(instance, *args)

    monkeypatch.setattr(RHFResponseAdapter, "_solve_orbitals", counted_solve)
    monkeypatch.setattr(RHFResponseAdapter, "_density_from_mo_response", counted)
    driver = rhf_oracle_case.method.nuc_grad_method(retain_details=False)
    driver.kernel(atmlst=(1,))

    assert coordinate_calls == 2


def test_density_response_satisfies_nonorthogonal_ao_invariants(
    rhf_oracle_case,
):
    response = rhf_oracle_case.response
    density_response = response.density_response
    overlap_response = response.overlap_derivative
    density = rhf_oracle_case.method.ao_density()
    overlap = np.asarray(rhf_oracle_case.reference.get_ovlp())
    overlap_finite_difference = rhf_oracle_case.finite_difference(
        "overlap",
        1.0e-4,
    )

    np.testing.assert_allclose(
        overlap_response,
        overlap_finite_difference,
        rtol=2.0e-8,
        atol=3.0e-9,
    )
    particle_number_residual = (
        np.einsum("...ij,ji->...", density_response, overlap)
        + np.einsum("ij,...ji->...", density, overlap_response)
    )
    idempotency_residual = (
        np.einsum("...ij,jk,kl->...il", density_response, overlap, density)
        + np.einsum("ij,...jk,kl->...il", density, overlap_response, density)
        + np.einsum("ij,jk,...kl->...il", density, overlap, density_response)
        - 2.0 * density_response
    )

    np.testing.assert_allclose(
        particle_number_residual,
        np.zeros_like(particle_number_residual),
        rtol=0.0,
        atol=2.0e-10,
    )
    np.testing.assert_allclose(
        idempotency_residual,
        np.zeros_like(idempotency_residual),
        rtol=0.0,
        atol=2.0e-10,
    )
    assert response.diagnostics.metric_residual < 2.0e-10
