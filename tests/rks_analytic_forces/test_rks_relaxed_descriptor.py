import numpy as np
import pytest


def test_descriptor_response_is_complete_density_contraction(
    rks_oracle_case,
):
    method = rks_oracle_case.method
    response = method.dq_dR_response(response=rks_oracle_case.response)
    explicit = method.dq_dR_explicit()
    relaxed = method.dq_dR_relaxed(response=rks_oracle_case.response)
    expected_response = np.einsum(
        "apij,bxij->bxap",
        method.dq_dP(),
        rks_oracle_case.response.density_response,
    )

    assert response.shape == (3, 3, 3, 4)
    np.testing.assert_allclose(
        response,
        expected_response,
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        relaxed,
        explicit + response,
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    assert np.max(np.abs(response)) > 0.4
    np.testing.assert_allclose(
        relaxed.sum(axis=0),
        np.zeros_like(relaxed[0]),
        rtol=0.0,
        atol=2.0e-10,
    )
    diagnostics = method.validate_force_compatibility()
    assert diagnostics.structural_zero_blocks == ()
    assert diagnostics.minimum_scaled_gap > 1.0e4


@pytest.mark.parametrize(
    ("step", "absolute_tolerance"),
    [
        pytest.param(1.0e-3, 5.0e-7, id="coarse"),
        pytest.param(3.0e-4, 8.0e-8, id="balanced"),
        pytest.param(1.0e-4, 1.0e-7, id="fine"),
    ],
)
def test_relaxed_descriptor_matches_fresh_rks(
    rks_oracle_case,
    step,
    absolute_tolerance,
):
    relaxed = rks_oracle_case.method.dq_dR_relaxed(
        response=rks_oracle_case.response
    )
    finite_difference = rks_oracle_case.finite_difference(
        "descriptor",
        step,
    )

    np.testing.assert_allclose(
        relaxed,
        finite_difference,
        rtol=2.0e-6,
        atol=absolute_tolerance,
    )


def test_relaxed_descriptor_preserves_metric_and_occupied_virtual_parts(
    rks_oracle_case,
):
    method = rks_oracle_case.method
    response = rks_oracle_case.response
    density_derivative = method.dq_dP()
    explicit = method.dq_dR_explicit()
    metric = np.einsum(
        "apij,bxij->bxap",
        density_derivative,
        response.density_response_metric,
    )
    occupied_virtual = np.einsum(
        "apij,bxij->bxap",
        density_derivative,
        response.density_response_occupied_virtual,
    )
    relaxed = method.dq_dR_relaxed(response=response)
    finite_difference = rks_oracle_case.finite_difference(
        "descriptor",
        3.0e-4,
    )

    np.testing.assert_allclose(
        relaxed,
        explicit + metric + occupied_virtual,
        rtol=0.0,
        atol=2.0e-13,
    )
    assert np.max(np.abs(metric)) > 0.34
    assert np.max(np.abs(occupied_virtual)) > 0.11
    assert np.max(
        np.abs(explicit + occupied_virtual - finite_difference)
    ) > 0.3
    assert np.max(
        np.abs(explicit + metric - finite_difference)
    ) > 0.1


@pytest.mark.parametrize(
    ("omission", "minimum_error"),
    [
        pytest.param("without_coulomb", 0.25, id="coulomb"),
        pytest.param("without_fxc", 0.02, id="fxc"),
        pytest.param("without_metric", 0.1, id="metric"),
        pytest.param("without_ao_motion", 0.3, id="ao-motion"),
        pytest.param(
            "without_grid_response",
            1.5e-3,
            id="all-grid-response",
        ),
        pytest.param("without_grid_coordinate", 0.1, id="grid-coordinate"),
        pytest.param("without_grid_weight", 0.1, id="grid-weight"),
    ],
)
def test_fresh_rks_descriptor_detects_each_omitted_response_component(
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
    finite_difference = rks_oracle_case.finite_difference(
        "descriptor",
        3.0e-4,
    )

    assert np.max(
        np.abs(omitted_relaxed - finite_difference)
    ) > minimum_error


def test_fixed_density_descriptor_derivative_cannot_replace_relaxed_rks(
    rks_oracle_case,
):
    explicit = rks_oracle_case.method.dq_dR_explicit()
    finite_difference = rks_oracle_case.finite_difference(
        "descriptor",
        3.0e-4,
    )

    assert np.max(np.abs(finite_difference - explicit)) > 0.4
