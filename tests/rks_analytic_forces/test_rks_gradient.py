import numpy as np
import pytest
import torch

from deepks.model.model import CorrNet

from conftest import ORACLE_PROJECTOR_BASIS


def _make_constant_model(bias, energy_constant=0.0):
    model = CorrNet(
        input_dim=4,
        hidden_sizes=(2,),
        proj_basis=ORACLE_PROJECTOR_BASIS,
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
        pytest.param(1.0e-3, 2.0e-6, id="coarse"),
        pytest.param(3.0e-4, 2.0e-7, id="balanced"),
        pytest.param(1.0e-4, 5.0e-8, id="fine"),
    ],
)
def test_total_gradient_matches_complete_energy_finite_difference(
    rks_oracle_case,
    step,
    absolute_tolerance,
):
    finite_difference = rks_oracle_case.finite_difference(
        "total_energy",
        step,
    )

    assert rks_oracle_case.gradient.shape == (3, 3)
    np.testing.assert_allclose(
        rks_oracle_case.gradient,
        finite_difference,
        rtol=3.0e-6,
        atol=absolute_tolerance,
    )


@pytest.mark.parametrize(
    ("step", "absolute_tolerance"),
    [
        pytest.param(1.0e-3, 2.0e-6, id="coarse"),
        pytest.param(3.0e-4, 2.0e-7, id="balanced"),
        pytest.param(1.0e-4, 5.0e-8, id="fine"),
    ],
)
def test_native_grid_response_gradient_matches_base_energy_finite_difference(
    rks_oracle_case,
    step,
    absolute_tolerance,
):
    finite_difference = rks_oracle_case.finite_difference(
        "base_energy",
        step,
    )

    np.testing.assert_allclose(
        rks_oracle_case.gradient_driver.reference_gradient,
        finite_difference,
        rtol=3.0e-6,
        atol=absolute_tolerance,
    )


def test_gradient_driver_preserves_every_response_partition(
    rks_oracle_case,
):
    driver = rks_oracle_case.gradient_driver

    np.testing.assert_allclose(
        driver.dq_dR_relaxed,
        driver.dq_dR_explicit + driver.dq_dR_response,
        rtol=0.0,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        driver.correction_gradient_response,
        driver.correction_gradient_metric
        + driver.correction_gradient_occupied_virtual,
        rtol=0.0,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        driver.correction_gradient,
        driver.correction_gradient_explicit
        + driver.correction_gradient_response,
        rtol=0.0,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        driver.de_full,
        driver.reference_gradient + driver.correction_gradient,
        rtol=0.0,
        atol=2.0e-13,
    )
    assert np.max(np.abs(driver.correction_gradient_explicit)) > 0.05
    assert np.max(np.abs(driver.correction_gradient_metric)) > 0.01
    assert np.max(
        np.abs(driver.correction_gradient_occupied_virtual)
    ) > 0.018
    assert np.max(np.abs(driver.correction_gradient_response)) > 0.02


def test_native_grid_coordinate_and_weight_gradient_parts_are_independent(
    rks_oracle_case,
):
    driver = rks_oracle_case.gradient_driver
    independent = rks_oracle_case.independent

    for actual, expected in (
        (
            driver.reference_gradient,
            independent.native_gradient,
        ),
        (
            driver.reference_gradient_without_grid_response,
            independent.native_gradient_without_grid_response,
        ),
        (
            driver.reference_gradient_xc_grid_coordinate,
            independent.native_gradient_grid_coordinate,
        ),
        (
            driver.reference_gradient_xc_grid_weight,
            independent.native_gradient_grid_weight,
        ),
    ):
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=2.0e-12,
            atol=2.0e-12,
        )
    np.testing.assert_allclose(
        driver.reference_gradient,
        driver.reference_gradient_without_grid_response
        + driver.reference_gradient_xc_grid_coordinate
        + driver.reference_gradient_xc_grid_weight,
        rtol=0.0,
        atol=5.0e-14,
    )
    assert np.max(
        np.abs(driver.reference_gradient_xc_grid_coordinate)
    ) > 0.99
    assert np.max(
        np.abs(driver.reference_gradient_xc_grid_weight)
    ) > 0.98
    assert np.max(
        np.abs(
            driver.reference_gradient
            - driver.reference_gradient_without_grid_response
        )
    ) > 4.0e-3
    finite_difference = rks_oracle_case.finite_difference(
        "base_energy",
        3.0e-4,
    )
    assert np.max(
        np.abs(
            driver.reference_gradient_without_grid_response
            - finite_difference
        )
    ) > 4.0e-3
    assert np.max(
        np.abs(
            driver.reference_gradient
            - driver.reference_gradient_xc_grid_coordinate
            - finite_difference
        )
    ) > 0.98
    assert np.max(
        np.abs(
            driver.reference_gradient
            - driver.reference_gradient_xc_grid_weight
            - finite_difference
        )
    ) > 0.98
    np.testing.assert_allclose(
        driver.reference_gradient.sum(axis=0),
        np.zeros(3),
        rtol=0.0,
        atol=2.0e-10,
    )


@pytest.mark.parametrize(
    ("omission", "minimum_error"),
    [
        pytest.param("without_coulomb", 2.0e-2, id="coulomb"),
        pytest.param("without_fxc", 1.5e-3, id="fxc"),
        pytest.param("without_metric", 5.0e-3, id="metric"),
        pytest.param("without_ao_motion", 1.5e-2, id="ao-motion"),
        pytest.param(
            "without_grid_response",
            1.0e-4,
            id="all-grid-response",
        ),
        pytest.param("without_grid_coordinate", 7.0e-3, id="grid-coordinate"),
        pytest.param("without_grid_weight", 7.0e-3, id="grid-weight"),
    ],
)
def test_total_energy_gradient_detects_each_omitted_response_component(
    rks_oracle_case,
    omission,
    minimum_error,
):
    method = rks_oracle_case.method
    explicit = method.dq_dR_explicit()
    omitted_density = getattr(
        rks_oracle_case.independent,
        omission,
    ).density_response
    omitted_relaxed = explicit + np.einsum(
        "apij,bxij->bxap",
        method.dq_dP(),
        omitted_density,
    )
    omitted_gradient = (
        rks_oracle_case.gradient_driver.reference_gradient
        + np.einsum(
            "bxap,ap->bx",
            omitted_relaxed,
            method.correction_sensitivity(),
        )
    )
    finite_difference = rks_oracle_case.finite_difference(
        "total_energy",
        3.0e-4,
    )

    assert np.max(
        np.abs(omitted_gradient - finite_difference)
    ) > minimum_error


def test_explicit_only_correction_gradient_is_not_relaxed_rks(
    rks_oracle_case,
):
    driver = rks_oracle_case.gradient_driver
    explicit_only = (
        driver.reference_gradient + driver.correction_gradient_explicit
    )
    finite_difference = rks_oracle_case.finite_difference(
        "total_energy",
        3.0e-4,
    )

    assert np.max(np.abs(explicit_only - finite_difference)) > 0.02


@pytest.mark.parametrize(
    ("bias", "energy_constant"),
    [
        pytest.param(0.0, 0.0, id="zero"),
        pytest.param(0.017, 0.023, id="constant"),
    ],
)
def test_zero_and_constant_corrections_reduce_to_native_rks_gradient(
    rks_oracle_case,
    bias,
    energy_constant,
):
    from deepks.deephf import RKSDeePHF

    model = _make_constant_model(bias, energy_constant)
    method = RKSDeePHF(
        rks_oracle_case.reference,
        model,
        projector_basis=ORACLE_PROJECTOR_BASIS,
    )
    method.kernel()
    driver = method.nuc_grad_method()
    actual = driver.kernel()

    np.testing.assert_allclose(
        method.e_corr,
        3.0 * bias + energy_constant,
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
        driver.correction_gradient,
        np.zeros((3, 3)),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        actual,
        rks_oracle_case.independent.native_gradient,
        rtol=2.0e-12,
        atol=2.0e-12,
    )


def test_absent_correction_reduces_to_native_rks_gradient(rks_oracle_case):
    from deepks.deephf import RKSDeePHF

    method = RKSDeePHF(
        rks_oracle_case.reference,
        None,
        projector_basis=ORACLE_PROJECTOR_BASIS,
    )
    assert method.kernel() == rks_oracle_case.reference.e_tot
    driver = method.nuc_grad_method()
    actual = driver.kernel()

    assert method.e_corr == 0.0
    np.testing.assert_allclose(
        method.correction_sensitivity(),
        np.zeros((3, 4)),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        driver.correction_gradient,
        np.zeros((3, 3)),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        actual,
        rks_oracle_case.independent.native_gradient,
        rtol=2.0e-12,
        atol=2.0e-12,
    )
