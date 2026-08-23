from dataclasses import fields, replace

import numpy as np
import pytest

import deepks.deephf.pyscf_uhf as pyscf_uhf
from deepks.deephf.capabilities import DeePHFCapabilityError
from deepks.deephf.pyscf_uhf import (
    UHFResponseAdapter,
    UHFResponseError,
    uhf_response_integrity_fingerprint,
)

from .conftest import (
    independent_coupled_uhf_operator,
    independent_metric_density_response,
    run_fresh_uhf,
)


@pytest.fixture(scope="module")
def response_reference():
    return run_fresh_uhf()


@pytest.fixture(scope="module")
def audited_response(response_reference):
    return UHFResponseAdapter(
        response_reference,
        residual_tolerance=1.0e-10,
    ).solve()


def _immutable_copy(value):
    result = np.array(value, copy=True)
    result.flags.writeable = False
    return result


def _independent_orbital_residual(reference, response):
    coefficient = np.asarray(reference.mo_coeff)
    energy = np.asarray(reference.mo_energy)
    occupation = np.asarray(reference.mo_occ)
    occupied = occupation > 0.0
    virtual = occupation == 0.0
    electron_repulsion = np.asarray(reference.mol.intor("int2e", aosym="s1"))
    spin_density_response = (
        response.alpha_density_response,
        response.beta_density_response,
    )
    total_density_response = spin_density_response[0] + spin_density_response[1]
    coulomb_response = np.einsum(
        "mnkl,bxlk->bxmn",
        electron_repulsion,
        total_density_response,
    )
    residuals = []
    for spin in range(2):
        exchange_response = np.einsum(
            "mkln,bxlk->bxmn",
            electron_repulsion,
            spin_density_response[spin],
        )
        induced_potential = coulomb_response - exchange_response
        occupied_coefficient = coefficient[spin][:, occupied[spin]]
        hamiltonian_derivative = (
            response.alpha_hamiltonian_derivative
            if spin == 0
            else response.beta_hamiltonian_derivative
        )
        mo_response = (
            response.alpha_mo_response
            if spin == 0
            else response.beta_mo_response
        )
        hamiltonian_mo = np.einsum(
            "mp,bxmn,ni->bxpi",
            coefficient[spin],
            hamiltonian_derivative,
            occupied_coefficient,
        )
        overlap_mo = np.einsum(
            "mp,bxmn,ni->bxpi",
            coefficient[spin],
            response.overlap_derivative,
            occupied_coefficient,
        )
        induced_mo = np.einsum(
            "mp,bxmn,ni->bxpi",
            coefficient[spin],
            induced_potential,
            occupied_coefficient,
        )
        residual = (
            hamiltonian_mo
            + induced_mo
            - overlap_mo * energy[spin, occupied[spin]]
            + (
                energy[spin, :, None]
                - energy[spin, occupied[spin]]
            )
            * mo_response
        )
        residuals.append(residual[..., virtual[spin], :])
    return tuple(residuals)


def test_coupled_operator_spectrum_matches_independent_ao_integrals(
    response_reference,
    audited_response,
):
    operator = independent_coupled_uhf_operator(response_reference)
    eigenvalues = np.linalg.eigvalsh(0.5 * (operator + operator.T))
    diagnostics = audited_response.diagnostics
    occupations = np.asarray(response_reference.mo_occ)
    alpha_dimension = int(
        np.count_nonzero(occupations[0] > 0.0)
        * np.count_nonzero(occupations[0] == 0.0)
    )
    beta_dimension = int(
        np.count_nonzero(occupations[1] > 0.0)
        * np.count_nonzero(occupations[1] == 0.0)
    )

    np.testing.assert_allclose(operator, operator.T, rtol=0.0, atol=2.0e-14)
    exact = UHFResponseAdapter(response_reference).validate_response_operator_exact()
    np.testing.assert_allclose(
        exact[3],
        eigenvalues[0],
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        exact[4],
        eigenvalues[-1],
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        exact[5],
        eigenvalues[-1] / eigenvalues[0],
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    assert diagnostics.alpha_response_dimension == alpha_dimension
    assert diagnostics.beta_response_dimension == beta_dimension
    assert diagnostics.operator_diagnostics_are_estimates is True
    assert diagnostics.response_dimension == alpha_dimension + beta_dimension
    assert diagnostics.operator_symmetry_residual < 2.0e-14


def test_physical_coupled_residual_matches_independent_ao_reconstruction(
    response_reference,
    audited_response,
):
    alpha_residual, beta_residual = _independent_orbital_residual(
        response_reference,
        audited_response,
    )
    diagnostics = audited_response.diagnostics

    np.testing.assert_allclose(
        audited_response.alpha_orbital_response_residual,
        alpha_residual,
        rtol=0.0,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        audited_response.beta_orbital_response_residual,
        beta_residual,
        rtol=0.0,
        atol=2.0e-12,
    )
    assert diagnostics.alpha_maximum_residual < 1.0e-10
    assert diagnostics.beta_maximum_residual < 1.0e-10
    assert diagnostics.maximum_residual < 1.0e-10
    assert diagnostics.maximum_residual <= diagnostics.residual_tolerance
    assert diagnostics.residual_history[-1] == diagnostics.maximum_residual
    assert diagnostics.refinement_cycles == len(diagnostics.residual_history) - 1


def test_spin_metric_density_and_all_response_invariants_are_complete(
    response_reference,
    audited_response,
):
    independent_metric = independent_metric_density_response(response_reference)
    diagnostics = audited_response.diagnostics

    np.testing.assert_allclose(
        audited_response.alpha_density_response_metric,
        independent_metric[:, :, 0],
        rtol=0.0,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        audited_response.beta_density_response_metric,
        independent_metric[:, :, 1],
        rtol=0.0,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        audited_response.alpha_density_response,
        audited_response.alpha_density_response_occupied_virtual
        + audited_response.alpha_density_response_metric,
        rtol=0.0,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        audited_response.beta_density_response,
        audited_response.beta_density_response_occupied_virtual
        + audited_response.beta_density_response_metric,
        rtol=0.0,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        audited_response.total_density_response,
        audited_response.alpha_density_response
        + audited_response.beta_density_response,
        rtol=0.0,
        atol=2.0e-14,
    )
    for name in (
        "alpha_metric_residual",
        "beta_metric_residual",
        "alpha_idempotency_residual",
        "beta_idempotency_residual",
        "alpha_particle_number_residual",
        "beta_particle_number_residual",
        "density_reconstruction_residual",
        "alpha_translation_residual",
        "beta_translation_residual",
        "translation_residual",
    ):
        assert getattr(diagnostics, name) < 2.0e-12
    np.testing.assert_allclose(
        np.sum(audited_response.alpha_density_response, axis=0),
        0.0,
        rtol=0.0,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        np.sum(audited_response.beta_density_response, axis=0),
        0.0,
        rtol=0.0,
        atol=2.0e-12,
    )


def test_public_audit_rebuilds_the_complete_response(
    response_reference,
    audited_response,
):
    adapter = UHFResponseAdapter(
        response_reference,
        residual_tolerance=1.0e-10,
    )

    assert adapter.audit_response_equations(audited_response) is None


def test_all_response_arrays_are_exact_float64_finite_and_immutable(
    audited_response,
):
    response_arrays = [
        getattr(audited_response, response_field.name)
        for response_field in fields(audited_response)
        if isinstance(getattr(audited_response, response_field.name), np.ndarray)
    ]

    assert response_arrays
    assert all(type(array) is np.ndarray for array in response_arrays)
    assert all(array.dtype == np.dtype(np.float64) for array in response_arrays)
    assert all(np.isfinite(array).all() for array in response_arrays)
    assert all(not array.flags.writeable for array in response_arrays)


def test_resealed_mutable_or_inconsistent_response_is_rejected(
    response_reference,
    audited_response,
):
    adapter = UHFResponseAdapter(
        response_reference,
        residual_tolerance=1.0e-10,
    )
    mutable = replace(
        audited_response,
        total_density_response=np.array(
            audited_response.total_density_response,
            copy=True,
        ),
        integrity_fingerprint="",
    )
    mutable = replace(
        mutable,
        integrity_fingerprint=uhf_response_integrity_fingerprint(mutable),
    )
    with pytest.raises(UHFResponseError, match="must be immutable"):
        adapter.audit_response_equations(mutable)

    inconsistent_density = _immutable_copy(
        audited_response.alpha_density_response
    )
    inconsistent_density_view = np.array(inconsistent_density, copy=True)
    inconsistent_density_view[0, 0, 0, 0] += 1.0e-5
    inconsistent_density_view.flags.writeable = False
    inconsistent = replace(
        audited_response,
        alpha_density_response=inconsistent_density_view,
        integrity_fingerprint="",
    )
    inconsistent = replace(
        inconsistent,
        integrity_fingerprint=uhf_response_integrity_fingerprint(inconsistent),
    )
    with pytest.raises(UHFResponseError, match="alpha density response is inconsistent"):
        adapter.audit_response_equations(inconsistent)


def test_audit_rejects_diagnostics_from_different_controls(
    response_reference,
    audited_response,
):
    adapter = UHFResponseAdapter(
        response_reference,
        residual_tolerance=2.0e-10,
    )

    with pytest.raises(UHFResponseError, match="does not match the adapter"):
        adapter.audit_response_equations(audited_response)


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("cphf_tolerance", True),
        ("residual_tolerance", "1e-9"),
        ("invariant_tolerance", 1.0e-9 + 0.0j),
        ("orbital_gap_tolerance", np.inf),
        ("level_shift", np.nan),
        ("operator_stability_tolerance", False),
        ("operator_condition_tolerance", 1.0),
        ("operator_symmetry_tolerance", 0.0),
        ("max_cycle", True),
        ("max_refinement_cycles", 1.0),
        ("operator_dimension_limit", np.bool_(True)),
    ],
)
def test_response_controls_have_strict_scalar_domains(
    response_reference,
    option,
    value,
):
    with pytest.raises(ValueError):
        UHFResponseAdapter(response_reference, **{option: value})


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"operator_dimension_limit": 21}, "dimension exceeds"),
        ({"operator_stability_tolerance": 0.1}, "unstable or singular"),
        ({"operator_condition_tolerance": 100.0}, "ill conditioned"),
    ],
)
def test_coupled_operator_domain_gates_are_explicit(
    response_reference,
    options,
    message,
):
    with pytest.raises(DeePHFCapabilityError, match=message):
        UHFResponseAdapter(
            response_reference,
            **options,
        ).validate_response_operator_exact()


def test_production_response_does_not_use_dense_debug_audit(
    response_reference,
    monkeypatch,
):
    def forbidden_dense_audit(*_args, **_kwargs):
        raise AssertionError("the UHF response matrix was materialized")

    monkeypatch.setattr(
        pyscf_uhf._UHFLinearResponseCore,
        "_response_operator_matrix_and_diagnostics",
        forbidden_dense_audit,
    )
    response = UHFResponseAdapter(
        response_reference,
        operator_dimension_limit=1,
    ).solve()

    assert response.diagnostics.response_dimension > 1
    assert response.diagnostics.operator_diagnostics_are_estimates is True


def test_nonsymmetric_coupled_operator_is_rejected(
    response_reference,
    monkeypatch,
):
    original_apply = UHFResponseAdapter._apply_occupied_virtual_operator

    def nonsymmetric_apply(self, vectors, *args, **kwargs):
        images = original_apply(self, vectors, *args, **kwargs)
        images = np.array(images, copy=True)
        images[..., 0] += 1.0e-3 * np.asarray(vectors)[..., 1]
        return images

    monkeypatch.setattr(
        UHFResponseAdapter,
        "_apply_occupied_virtual_operator",
        nonsymmetric_apply,
    )
    with pytest.raises(UHFResponseError, match="violates symmetry"):
        UHFResponseAdapter(response_reference).solve()


def test_corrupted_ucphf_solution_fails_without_fallback(
    response_reference,
    monkeypatch,
):
    original_solve = pyscf_uhf.ucphf.solve
    first_alpha_virtual = int(
        np.flatnonzero(np.asarray(response_reference.mo_occ[0]) == 0.0)[0]
    )

    def corrupted_solve(*args, **kwargs):
        response, orbital_energy_response = original_solve(*args, **kwargs)
        alpha = np.array(response[0], copy=True)
        beta = np.array(response[1], copy=True)
        alpha.reshape(len(alpha), alpha.shape[-2], alpha.shape[-1])[
            :, first_alpha_virtual, 0
        ] += 1.0e-4
        return (alpha, beta), orbital_energy_response

    monkeypatch.setattr(pyscf_uhf.ucphf, "solve", corrupted_solve)
    adapter = UHFResponseAdapter(
        response_reference,
        max_refinement_cycles=0,
    )

    with pytest.raises(UHFResponseError, match="coupled response residual exceeds"):
        adapter.solve()


def test_independent_residual_drives_ucphf_refinement(
    response_reference,
    monkeypatch,
):
    original_solve = pyscf_uhf.ucphf.solve
    solve_calls = 0
    first_alpha_virtual = int(
        np.flatnonzero(np.asarray(response_reference.mo_occ[0]) == 0.0)[0]
    )

    def inaccurate_first_solve(*args, **kwargs):
        nonlocal solve_calls
        solve_calls += 1
        response, orbital_energy_response = original_solve(*args, **kwargs)
        if solve_calls == 1:
            alpha = np.array(response[0], copy=True)
            beta = np.array(response[1], copy=True)
            alpha.reshape(len(alpha), alpha.shape[-2], alpha.shape[-1])[
                :, first_alpha_virtual, 0
            ] += 1.0e-5
            response = (alpha, beta)
        return response, orbital_energy_response

    monkeypatch.setattr(pyscf_uhf.ucphf, "solve", inaccurate_first_solve)
    response = UHFResponseAdapter(
        response_reference,
        residual_tolerance=1.0e-10,
    ).solve()

    assert solve_calls >= 2
    assert response.diagnostics.refinement_cycles >= 1
    assert response.diagnostics.residual_history[0] > 1.0e-7
    assert response.diagnostics.maximum_residual < 1.0e-10
