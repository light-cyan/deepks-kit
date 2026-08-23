import numpy as np
import pytest

import deepks.deephf.pyscf_rhf as pyscf_rhf
from deepks.deephf import DeePHF, RHFResponseError


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

    assert not hasattr(gradient_driver, "response_result")
    assert not hasattr(gradient_driver, "dq_dR_relaxed")
    assert not hasattr(gradient_driver, "de_full")
    assert gradient_driver.de is None
