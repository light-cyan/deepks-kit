import numpy as np
import pytest
from pyscf import ao2mo

import deepks.deephf.pyscf_rhf as pyscf_rhf
from deepks.deephf import DeePHF, RHFResponseAdapter, RHFResponseError


PROJECTOR_BASIS = [[0, [0.8, 1.0]], [1, [0.3, 1.0]]]


def test_response_reports_independently_small_residuals(rhf_oracle_case):
    diagnostics = rhf_oracle_case.response.diagnostics

    assert diagnostics.minimum_orbital_gap > 0.8
    assert diagnostics.maximum_residual < 1.0e-10
    assert diagnostics.residual_rms < 1.0e-10
    assert diagnostics.metric_residual < 1.0e-10
    assert diagnostics.idempotency_residual < 1.0e-10
    assert diagnostics.particle_number_residual < 1.0e-10
    assert diagnostics.maximum_residual <= diagnostics.residual_tolerance


def test_response_operator_spectrum_matches_independent_ao2mo_formula(
    rhf_oracle_case,
):
    reference = rhf_oracle_case.reference
    molecule = reference.mol
    coefficient = np.asarray(reference.mo_coeff)
    occupied = reference.mo_occ > 0
    virtual = reference.mo_occ == 0
    occupied_coefficients = coefficient[:, occupied]
    virtual_coefficients = coefficient[:, virtual]
    nocc = int(np.count_nonzero(occupied))
    nvir = int(np.count_nonzero(virtual))
    coulomb = ao2mo.general(
        molecule,
        (
            virtual_coefficients,
            occupied_coefficients,
            virtual_coefficients,
            occupied_coefficients,
        ),
        compact=False,
    ).reshape(nvir, nocc, nvir, nocc)
    exchange_abij = ao2mo.general(
        molecule,
        (
            virtual_coefficients,
            virtual_coefficients,
            occupied_coefficients,
            occupied_coefficients,
        ),
        compact=False,
    ).reshape(nvir, nvir, nocc, nocc)
    exchange_ajib = ao2mo.general(
        molecule,
        (
            virtual_coefficients,
            occupied_coefficients,
            occupied_coefficients,
            virtual_coefficients,
        ),
        compact=False,
    ).reshape(nvir, nocc, nocc, nvir)
    operator = (
        4.0 * coulomb
        - exchange_abij.transpose(0, 2, 1, 3)
        - exchange_ajib.transpose(0, 2, 3, 1)
    )
    orbital_gap = (
        reference.mo_energy[virtual, None]
        - reference.mo_energy[occupied]
    )
    operator = operator.reshape(nvir * nocc, nvir * nocc)
    operator += np.diag(orbital_gap.reshape(-1))
    eigenvalues = np.linalg.eigvalsh(operator)
    diagnostics = rhf_oracle_case.response.diagnostics

    np.testing.assert_allclose(operator, operator.T, rtol=0.0, atol=2.0e-14)
    np.testing.assert_allclose(
        diagnostics.operator_minimum_eigenvalue,
        eigenvalues[0],
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        diagnostics.operator_maximum_eigenvalue,
        eigenvalues[-1],
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        diagnostics.operator_condition_number,
        eigenvalues[-1] / eigenvalues[0],
        rtol=2.0e-13,
        atol=2.0e-13,
    )


def test_solver_level_shift_does_not_change_physical_operator_diagnostics(
    rhf_oracle_case,
):
    shifted = RHFResponseAdapter(
        rhf_oracle_case.reference,
        level_shift=0.2,
    ).solve()
    unshifted = rhf_oracle_case.response.diagnostics
    shifted = shifted.diagnostics

    for name in (
        "operator_minimum_eigenvalue",
        "operator_maximum_eigenvalue",
        "operator_condition_number",
        "operator_symmetry_residual",
    ):
        np.testing.assert_allclose(
            getattr(shifted, name),
            getattr(unshifted, name),
            rtol=0.0,
            atol=2.0e-13,
        )


def test_corrupted_cphf_solution_fails_without_explicit_fallback(
    rhf_oracle_case,
    monkeypatch,
):
    original_solve = pyscf_rhf.cphf.solve
    occupations = np.asarray(rhf_oracle_case.reference.mo_occ)
    nmo = occupations.size
    nocc = int(np.count_nonzero(occupations > 0))
    first_virtual = int(np.flatnonzero(occupations == 0)[0])

    def corrupted_solve(*args, **kwargs):
        response, orbital_energy_response = original_solve(*args, **kwargs)
        response = np.asarray(response).copy()
        response.reshape(-1, nmo, nocc)[:, first_virtual, 0] += 1.0e-4
        return response, orbital_energy_response

    monkeypatch.setattr(pyscf_rhf.cphf, "solve", corrupted_solve)
    method = DeePHF(
        rhf_oracle_case.reference,
        rhf_oracle_case.model,
        projector_basis=PROJECTOR_BASIS,
        response_options={"max_refinement_cycles": 0},
    )
    gradient_driver = method.nuc_grad_method()

    with pytest.raises(
        RHFResponseError,
        match="RHF response residual exceeds tolerance",
    ):
        gradient_driver.kernel()

    assert gradient_driver.response_result is None
    assert gradient_driver.dq_dR_relaxed is None
    assert gradient_driver.de_full is None
    assert gradient_driver.de is None
