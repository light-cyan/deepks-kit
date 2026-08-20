import numpy as np
import pytest


@pytest.mark.parametrize(
    ("step", "absolute_tolerance"),
    [
        pytest.param(1.0e-3, 5.0e-7, id="coarse"),
        pytest.param(3.0e-4, 7.0e-8, id="balanced"),
        pytest.param(1.0e-4, 5.0e-8, id="fine"),
    ],
)
def test_relaxed_descriptor_matches_fresh_uhf_finite_difference(
    uhf_oracle_case,
    step,
    absolute_tolerance,
):
    analytic = uhf_oracle_case.method.dq_dR_relaxed(
        response=uhf_oracle_case.response
    )
    finite_difference = uhf_oracle_case.finite_difference(
        "descriptor",
        step,
    )

    assert analytic.shape == (3, 3, 3, 4)
    np.testing.assert_allclose(
        analytic,
        finite_difference,
        rtol=2.0e-6,
        atol=absolute_tolerance,
    )


def test_descriptor_response_is_the_complete_spin_density_contraction(
    uhf_oracle_case,
):
    method = uhf_oracle_case.method
    response = uhf_oracle_case.response
    response_spin = method.dq_dR_response_spin(response=response)
    expected_spin = np.einsum(
        "apij,sbxij->sbxap",
        method.dq_dP(),
        np.stack(
            (
                response.alpha_density_response,
                response.beta_density_response,
            )
        ),
    )

    assert response_spin.shape == (2, 3, 3, 3, 4)
    np.testing.assert_allclose(
        response_spin,
        expected_spin,
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        method.dq_dR_response(response=response),
        response_spin.sum(axis=0),
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    assert np.linalg.norm(response_spin[0]) > 0.4
    assert np.linalg.norm(response_spin[1]) > 0.5


def test_explicit_response_and_relaxed_descriptor_partitions_remain_distinct(
    uhf_oracle_case,
):
    method = uhf_oracle_case.method
    response = uhf_oracle_case.response
    explicit_spin = method.dq_dR_explicit_spin()
    response_spin = method.dq_dR_response_spin(response=response)
    relaxed_spin = method.dq_dR_relaxed_spin(response=response)
    explicit = method.dq_dR_explicit()
    response_total = method.dq_dR_response(response=response)
    relaxed = method.dq_dR_relaxed(response=response)

    np.testing.assert_allclose(
        explicit_spin.sum(axis=0),
        explicit,
        rtol=0.0,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        relaxed_spin,
        explicit_spin + response_spin,
        rtol=0.0,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        response_spin.sum(axis=0),
        response_total,
        rtol=0.0,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        relaxed_spin.sum(axis=0),
        relaxed,
        rtol=0.0,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        relaxed,
        explicit + response_total,
        rtol=0.0,
        atol=2.0e-13,
    )
    assert np.max(np.abs(response_total)) > 0.1
    assert np.max(np.abs(relaxed - explicit)) > 0.1


def test_descriptor_response_retains_metric_and_occupied_virtual_parts(
    uhf_oracle_case,
):
    method = uhf_oracle_case.method
    response = uhf_oracle_case.response
    density_metric_spin = np.stack(
        (
            response.alpha_density_response_metric,
            response.beta_density_response_metric,
        )
    )
    density_occupied_virtual_spin = np.stack(
        (
            response.alpha_density_response_occupied_virtual,
            response.beta_density_response_occupied_virtual,
        )
    )
    metric_spin = np.einsum(
        "apij,sbxij->sbxap",
        method.dq_dP(),
        density_metric_spin,
    )
    occupied_virtual_spin = np.einsum(
        "apij,sbxij->sbxap",
        method.dq_dP(),
        density_occupied_virtual_spin,
    )
    response_spin = method.dq_dR_response_spin(response=response)

    np.testing.assert_allclose(
        response_spin,
        metric_spin + occupied_virtual_spin,
        rtol=0.0,
        atol=2.0e-13,
    )
    assert np.linalg.norm(metric_spin[0]) > 0.3
    assert np.linalg.norm(metric_spin[1]) > 0.4
    assert np.linalg.norm(occupied_virtual_spin[0]) > 0.1
    assert np.linalg.norm(occupied_virtual_spin[1]) > 0.1
    relaxed = method.dq_dR_relaxed(response=response)
    metric_omitted = method.dq_dR_explicit() + occupied_virtual_spin.sum(axis=0)
    occupied_virtual_omitted = method.dq_dR_explicit() + metric_spin.sum(axis=0)
    assert np.max(np.abs(relaxed - metric_omitted)) > 0.2
    assert np.max(np.abs(relaxed - occupied_virtual_omitted)) > 0.05


def test_neither_spin_channel_can_replace_the_total_descriptor_response(
    uhf_oracle_case,
):
    method = uhf_oracle_case.method
    response_spin = method.dq_dR_response_spin(
        response=uhf_oracle_case.response
    )
    response_total = response_spin.sum(axis=0)

    assert np.max(np.abs(response_total - response_spin[0])) > 0.1
    assert np.max(np.abs(response_total - response_spin[1])) > 0.1
    assert np.max(np.abs(response_spin[0] - response_spin[1])) > 0.02


def test_relaxed_spin_descriptor_partitions_are_translationally_invariant(
    uhf_oracle_case,
):
    relaxed_spin = uhf_oracle_case.method.dq_dR_relaxed_spin(
        response=uhf_oracle_case.response
    )

    for spin in range(2):
        np.testing.assert_allclose(
            relaxed_spin[spin].sum(axis=0),
            np.zeros_like(relaxed_spin[spin, 0]),
            rtol=0.0,
            atol=2.0e-12,
        )
    np.testing.assert_allclose(
        relaxed_spin.sum(axis=(0, 1)),
        np.zeros_like(relaxed_spin[0, 0]),
        rtol=0.0,
        atol=2.0e-12,
    )


def test_descriptor_fixture_is_strictly_differentiable(uhf_oracle_case):
    assert uhf_oracle_case.minimum_descriptor_gap > 8.0e-2
    diagnostics = uhf_oracle_case.method.validate_force_compatibility()
    assert diagnostics.structural_zero_blocks == ()
    assert diagnostics.minimum_scaled_gap > 1.0e5


def test_explicit_descriptor_derivative_cannot_replace_relaxed_response(
    uhf_oracle_case,
):
    explicit = uhf_oracle_case.method.dq_dR_explicit()
    finite_difference = uhf_oracle_case.finite_difference(
        "descriptor",
        3.0e-4,
    )

    assert np.max(np.abs(finite_difference - explicit)) > 0.3
