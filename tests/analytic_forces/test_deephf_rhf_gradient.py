import numpy as np
import pytest
import torch

from deepks.deephf import DeePHF
from deepks.model.model import CorrNet


PROJECTOR_BASIS = [[0, [0.8, 1.0]], [1, [0.3, 1.0]]]


def _make_constant_model(bias):
    model = CorrNet(
        input_dim=4,
        hidden_sizes=(2,),
        proj_basis=PROJECTOR_BASIS,
    ).double()
    with torch.no_grad():
        model.linear.weight.zero_()
        model.linear.bias.fill_(bias)
        for parameter in model.densenet.parameters():
            parameter.zero_()
    return model.eval()


def test_total_gradient_matches_complete_energy_finite_difference(
    rhf_oracle_case,
):
    gradient_driver = rhf_oracle_case.method.nuc_grad_method()
    analytic = gradient_driver.kernel()

    assert analytic.shape == (3, 3)
    np.testing.assert_allclose(
        gradient_driver.dq_dR_relaxed,
        gradient_driver.dq_dR_explicit + gradient_driver.dq_dR_response,
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        gradient_driver.correction_gradient,
        gradient_driver.correction_gradient_explicit
        + gradient_driver.correction_gradient_response,
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        analytic,
        gradient_driver.reference_gradient + gradient_driver.correction_gradient,
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    assert np.max(np.abs(gradient_driver.correction_gradient_response)) > 1.0e-3

    for step, absolute_tolerance in (
        (1.0e-3, 3.0e-6),
        (3.0e-4, 4.0e-7),
        (1.0e-4, 1.0e-7),
    ):
        finite_difference = rhf_oracle_case.finite_difference(
            "total_energy",
            step,
        )
        np.testing.assert_allclose(
            analytic,
            finite_difference,
            rtol=3.0e-6,
            atol=absolute_tolerance,
            err_msg=f"central-difference step {step:.1e} Bohr",
        )

    fine_difference = rhf_oracle_case.finite_difference(
        "total_energy",
        1.0e-4,
    )
    explicit_only = (
        gradient_driver.reference_gradient
        + gradient_driver.correction_gradient_explicit
    )
    assert np.max(np.abs(fine_difference - explicit_only)) > 1.0e-3


def _assert_constant_correction_has_native_gradient(
    rhf_oracle_case,
    bias,
    expected_correction,
):
    model = _make_constant_model(bias)
    method = DeePHF(
        rhf_oracle_case.reference,
        model,
        projector_basis=PROJECTOR_BASIS,
    )
    method.kernel()
    gradient_driver = method.nuc_grad_method()
    actual = gradient_driver.kernel()
    native = np.asarray(
        rhf_oracle_case.reference.nuc_grad_method().kernel()
    )

    np.testing.assert_allclose(
        method.e_corr,
        expected_correction,
        rtol=0.0,
        atol=2.0e-15,
    )
    np.testing.assert_allclose(
        method.correction_sensitivity(),
        np.zeros((3, 4)),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        gradient_driver.correction_gradient,
        np.zeros((3, 3)),
        rtol=0.0,
        atol=2.0e-15,
    )
    np.testing.assert_allclose(
        actual,
        native,
        rtol=2.0e-12,
        atol=2.0e-12,
    )


def test_zero_correction_reduces_to_native_rhf_gradient(rhf_oracle_case):
    _assert_constant_correction_has_native_gradient(
        rhf_oracle_case,
        bias=0.0,
        expected_correction=0.0,
    )


def test_constant_correction_reduces_to_native_rhf_gradient(rhf_oracle_case):
    _assert_constant_correction_has_native_gradient(
        rhf_oracle_case,
        bias=0.017,
        expected_correction=0.051,
    )


def test_force_sign_and_selected_atom_gradient(rhf_oracle_case):
    method = rhf_oracle_case.method
    complete = method.gradient()
    np.testing.assert_allclose(
        method.forces(),
        -complete,
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    selected_driver = method.nuc_grad_method()
    selected = selected_driver.kernel(atmlst=(np.int64(2), 0))
    np.testing.assert_allclose(
        selected,
        complete[[2, 0]],
        rtol=2.0e-12,
        atol=2.0e-12,
    )


@pytest.mark.parametrize("invalid_index", [0.9, True, "0"])
def test_gradient_rejects_noninteger_atom_indices_before_response(
    rhf_oracle_case,
    invalid_index,
):
    driver = rhf_oracle_case.method.nuc_grad_method()

    with pytest.raises(TypeError, match="atom indices must be integers"):
        driver.kernel(atmlst=[invalid_index])

    assert driver.response_result is None
    assert driver.de_full is None


def test_response_and_gradient_do_not_mutate_native_reference(rhf_oracle_case):
    reference = rhf_oracle_case.reference
    fock_before = np.asarray(reference.get_fock()).copy()
    density_before = np.asarray(reference.make_rdm1()).copy()
    coefficient_before = np.asarray(reference.mo_coeff).copy()
    orbital_energy_before = np.asarray(reference.mo_energy).copy()
    occupation_before = np.asarray(reference.mo_occ).copy()
    total_energy_before = float(reference.e_tot)
    converged_before = bool(reference.converged)

    response = rhf_oracle_case.method.response()
    gradient = rhf_oracle_case.method.gradient()

    assert np.isfinite(response.density_response).all()
    assert np.isfinite(gradient).all()
    assert reference.converged is converged_before
    assert reference.e_tot == total_energy_before
    np.testing.assert_allclose(reference.get_fock(), fock_before, rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(reference.make_rdm1(), density_before)
    np.testing.assert_array_equal(reference.mo_coeff, coefficient_before)
    np.testing.assert_array_equal(reference.mo_energy, orbital_energy_before)
    np.testing.assert_array_equal(reference.mo_occ, occupation_before)
