import numpy as np
import pytest
import torch

from deepks.deephf import UHFDeePHF
from deepks.model.model import CorrNet


PROJECTOR_BASIS = [[0, [0.8, 1.0]], [1, [0.3, 1.0]]]


def _make_constant_model(bias, energy_constant=0.0):
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
        model.energy_const.fill_(energy_constant)
    return model.eval()


@pytest.mark.parametrize(
    ("step", "absolute_tolerance"),
    [
        pytest.param(1.0e-3, 8.0e-7, id="coarse"),
        pytest.param(3.0e-4, 1.0e-7, id="balanced"),
        pytest.param(1.0e-4, 4.0e-8, id="fine"),
    ],
)
def test_total_gradient_matches_complete_energy_finite_difference(
    uhf_oracle_case,
    step,
    absolute_tolerance,
):
    finite_difference = uhf_oracle_case.finite_difference(
        "total_energy",
        step,
    )

    assert uhf_oracle_case.gradient.shape == (3, 3)
    np.testing.assert_allclose(
        uhf_oracle_case.gradient,
        finite_difference,
        rtol=3.0e-6,
        atol=absolute_tolerance,
    )


def test_gradient_detects_spin_metric_and_occupied_virtual_omissions(
    uhf_oracle_case,
):
    driver = uhf_oracle_case.gradient_driver
    finite_difference = uhf_oracle_case.finite_difference(
        "total_energy",
        1.0e-4,
    )
    explicit_only = (
        driver.reference_gradient + driver.correction_gradient_explicit
    )
    metric_omitted = (
        driver.reference_gradient
        + driver.correction_gradient_explicit
        + driver.correction_gradient_occupied_virtual
    )
    occupied_virtual_omitted = (
        driver.reference_gradient
        + driver.correction_gradient_explicit
        + driver.correction_gradient_metric
    )

    assert np.max(np.abs(finite_difference - explicit_only)) > 1.0e-2
    assert np.max(np.abs(finite_difference - metric_omitted)) > 1.0e-2
    assert np.max(
        np.abs(finite_difference - occupied_virtual_omitted)
    ) > 1.0e-3
    assert np.max(
        np.abs(
            driver.correction_gradient
            - driver.correction_gradient_spin[0]
        )
    ) > 1.0e-2
    assert np.max(
        np.abs(
            driver.correction_gradient
            - driver.correction_gradient_spin[1]
        )
    ) > 1.0e-2


def _assert_constant_correction_has_native_gradient(
    uhf_oracle_case,
    bias,
    energy_constant,
    expected_correction,
):
    model = _make_constant_model(bias, energy_constant)
    method = UHFDeePHF(
        uhf_oracle_case.reference,
        model,
        projector_basis=PROJECTOR_BASIS,
    )
    method.kernel()
    gradient_driver = method.nuc_grad_method()
    actual = gradient_driver.kernel()
    native = np.asarray(
        uhf_oracle_case.reference.nuc_grad_method().kernel()
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
        gradient_driver.correction_gradient_spin,
        np.zeros((2, 3, 3)),
        rtol=0.0,
        atol=2.0e-15,
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


def test_zero_correction_reduces_to_native_uhf_gradient(uhf_oracle_case):
    _assert_constant_correction_has_native_gradient(
        uhf_oracle_case,
        bias=0.0,
        energy_constant=0.0,
        expected_correction=0.0,
    )


def test_constant_correction_reduces_to_native_uhf_gradient(
    uhf_oracle_case,
):
    _assert_constant_correction_has_native_gradient(
        uhf_oracle_case,
        bias=0.017,
        energy_constant=0.019,
        expected_correction=0.070,
    )


def test_force_sign_and_selected_atom_gradient(uhf_oracle_case):
    method = uhf_oracle_case.method
    np.testing.assert_allclose(
        method.forces(),
        -uhf_oracle_case.gradient,
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    selected = method.nuc_grad_method().kernel(atmlst=(np.int64(2), 0))
    np.testing.assert_allclose(
        selected,
        uhf_oracle_case.gradient[[2, 0]],
        rtol=2.0e-12,
        atol=2.0e-12,
    )


def test_uhf_public_relaxed_derivative_builds_force_inputs_once(
    uhf_oracle_case,
    monkeypatch,
):
    method = uhf_oracle_case.method
    original_force_inputs = method._force_inputs
    original_explicit_component = method._descriptor.dq_dR_explicit_component
    force_input_calls = 0
    explicit_component_calls = 0

    def counted_force_inputs(**options):
        nonlocal force_input_calls
        force_input_calls += 1
        return original_force_inputs(**options)

    def counted_explicit_component(*args, **options):
        nonlocal explicit_component_calls
        explicit_component_calls += 1
        return original_explicit_component(*args, **options)

    monkeypatch.setattr(method, "_force_inputs", counted_force_inputs)
    monkeypatch.setattr(
        method._descriptor,
        "dq_dR_explicit_component",
        counted_explicit_component,
    )

    assert np.isfinite(method.dq_dR_relaxed()).all()
    assert force_input_calls == 1
    assert explicit_component_calls == 2


def test_complete_and_spin_correction_gradients_are_translationally_invariant(
    uhf_oracle_case,
):
    driver = uhf_oracle_case.gradient_driver

    for gradient in (
        driver.reference_gradient,
        driver.correction_gradient_spin[0],
        driver.correction_gradient_spin[1],
        driver.correction_gradient,
        driver.de_full,
    ):
        np.testing.assert_allclose(
            gradient.sum(axis=0),
            np.zeros(3),
            rtol=0.0,
            atol=2.0e-10,
        )


def test_response_and_gradient_leave_native_uhf_reference_unchanged(
    uhf_oracle_case,
):
    reference = uhf_oracle_case.reference
    fock_before = np.asarray(reference.get_fock()).copy()
    density_before = np.asarray(reference.make_rdm1()).copy()
    coefficient_before = np.asarray(reference.mo_coeff).copy()
    orbital_energy_before = np.asarray(reference.mo_energy).copy()
    occupation_before = np.asarray(reference.mo_occ).copy()
    total_energy_before = float(reference.e_tot)
    converged_before = bool(reference.converged)

    response = uhf_oracle_case.method.response()
    gradient = uhf_oracle_case.method.gradient()

    assert np.isfinite(response.total_density_response).all()
    assert np.isfinite(gradient).all()
    assert reference.converged is converged_before
    assert reference.e_tot == total_energy_before
    np.testing.assert_allclose(
        reference.get_fock(),
        fock_before,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_array_equal(reference.make_rdm1(), density_before)
    np.testing.assert_array_equal(reference.mo_coeff, coefficient_before)
    np.testing.assert_array_equal(reference.mo_energy, orbital_energy_before)
    np.testing.assert_array_equal(reference.mo_occ, occupation_before)
