import numpy as np
import pytest


@pytest.mark.parametrize(
    ("step", "absolute_tolerance"),
    [
        pytest.param(1.0e-3, 6.0e-7, id="coarse"),
        pytest.param(3.0e-4, 8.0e-8, id="balanced"),
        pytest.param(1.0e-4, 7.0e-8, id="fine"),
    ],
)
@pytest.mark.parametrize(
    ("state_field", "response_field"),
    [
        pytest.param(
            "density_alpha",
            "alpha_density_response",
            id="alpha",
        ),
        pytest.param(
            "density_beta",
            "beta_density_response",
            id="beta",
        ),
        pytest.param(
            "density_total",
            "total_density_response",
            id="total",
        ),
    ],
)
def test_spin_resolved_ao_density_response_matches_finite_difference(
    uhf_oracle_case,
    step,
    absolute_tolerance,
    state_field,
    response_field,
):
    analytic = getattr(uhf_oracle_case.response, response_field)
    finite_difference = uhf_oracle_case.finite_difference(state_field, step)

    assert analytic.shape == (3, 3, 7, 7)
    np.testing.assert_allclose(
        analytic,
        finite_difference,
        rtol=2.0e-6,
        atol=absolute_tolerance,
    )


def test_density_response_preserves_all_spin_and_motion_partitions(
    uhf_oracle_case,
):
    response = uhf_oracle_case.response

    np.testing.assert_allclose(
        response.alpha_density_response,
        response.alpha_density_response_occupied_virtual
        + response.alpha_density_response_metric,
        rtol=0.0,
        atol=2.0e-15,
    )
    np.testing.assert_allclose(
        response.beta_density_response,
        response.beta_density_response_occupied_virtual
        + response.beta_density_response_metric,
        rtol=0.0,
        atol=2.0e-15,
    )
    np.testing.assert_allclose(
        response.total_density_response,
        response.alpha_density_response
        + response.beta_density_response,
        rtol=0.0,
        atol=2.0e-15,
    )
    np.testing.assert_allclose(
        response.total_density_response_occupied_virtual,
        response.alpha_density_response_occupied_virtual
        + response.beta_density_response_occupied_virtual,
        rtol=0.0,
        atol=2.0e-15,
    )
    np.testing.assert_allclose(
        response.total_density_response_metric,
        response.alpha_density_response_metric
        + response.beta_density_response_metric,
        rtol=0.0,
        atol=2.0e-15,
    )
    assert np.linalg.norm(response.alpha_density_response) > 0.5
    assert np.linalg.norm(response.beta_density_response) > 0.5
    assert np.linalg.norm(
        response.total_density_response_occupied_virtual
    ) > 0.1
    assert np.linalg.norm(response.total_density_response_metric) > 0.5
    assert np.max(
        np.abs(
            response.total_density_response
            - response.alpha_density_response
        )
    ) > 0.1
    assert np.max(
        np.abs(
            response.total_density_response
            - response.beta_density_response
        )
    ) > 0.1
    assert np.max(
        np.abs(
            response.total_density_response
            - response.total_density_response_metric
        )
    ) > 0.05
    assert np.max(
        np.abs(
            response.total_density_response
            - response.total_density_response_occupied_virtual
        )
    ) > 0.1


def test_method_density_accessors_retain_spin_resolution(uhf_oracle_case):
    method = uhf_oracle_case.method
    response = uhf_oracle_case.response
    spin_density = method.first_order_spin_density(response=response)
    total_density = method.first_order_density(response=response)

    assert spin_density.shape == (2, 3, 3, 7, 7)
    np.testing.assert_allclose(
        spin_density[0],
        response.alpha_density_response,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        spin_density[1],
        response.beta_density_response,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        total_density,
        spin_density.sum(axis=0),
        rtol=0.0,
        atol=2.0e-15,
    )


def test_overlap_and_spin_metric_responses_match_independent_oracles(
    uhf_oracle_case,
):
    response = uhf_oracle_case.response
    overlap_finite_difference = uhf_oracle_case.finite_difference(
        "overlap",
        1.0e-4,
    )

    np.testing.assert_allclose(
        response.overlap_derivative,
        uhf_oracle_case.overlap_derivative,
        rtol=0.0,
        atol=2.0e-15,
    )
    np.testing.assert_allclose(
        response.overlap_derivative,
        overlap_finite_difference,
        rtol=2.0e-8,
        atol=3.0e-9,
    )
    np.testing.assert_allclose(
        response.alpha_density_response_metric,
        uhf_oracle_case.metric_density_response[:, :, 0],
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        response.beta_density_response_metric,
        uhf_oracle_case.metric_density_response[:, :, 1],
        rtol=2.0e-13,
        atol=2.0e-13,
    )


def test_each_spin_density_satisfies_nonorthogonal_ao_invariants(
    uhf_oracle_case,
):
    response = uhf_oracle_case.response
    overlap = np.asarray(uhf_oracle_case.reference.get_ovlp())
    overlap_response = response.overlap_derivative
    density_ground = np.asarray(uhf_oracle_case.reference.make_rdm1())
    density_responses = (
        response.alpha_density_response,
        response.beta_density_response,
    )

    particle_residuals = []
    for spin, density_response in enumerate(density_responses):
        density = density_ground[spin]
        particle_number_residual = (
            np.einsum("...ij,ji->...", density_response, overlap)
            + np.einsum("ij,...ji->...", density, overlap_response)
        )
        idempotency_residual = (
            np.einsum(
                "...ij,jk,kl->...il",
                density_response,
                overlap,
                density,
            )
            + np.einsum(
                "ij,...jk,kl->...il",
                density,
                overlap_response,
                density,
            )
            + np.einsum(
                "ij,jk,...kl->...il",
                density,
                overlap,
                density_response,
            )
            - density_response
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
        particle_residuals.append(particle_number_residual)
    np.testing.assert_allclose(
        particle_residuals[0] + particle_residuals[1],
        np.zeros_like(particle_residuals[0]),
        rtol=0.0,
        atol=2.0e-10,
    )


def test_spin_density_response_is_translationally_invariant(uhf_oracle_case):
    response = uhf_oracle_case.response

    for density_response in (
        response.alpha_density_response,
        response.beta_density_response,
        response.total_density_response,
    ):
        np.testing.assert_allclose(
            density_response.sum(axis=0),
            np.zeros_like(density_response[0]),
            rtol=0.0,
            atol=6.0e-12,
        )
    diagnostics = response.diagnostics
    assert diagnostics.alpha_translation_residual < 6.0e-12
    assert diagnostics.beta_translation_residual < 6.0e-12
    assert diagnostics.translation_residual < 6.0e-12


def test_coupled_operator_matches_independent_ao_integral_oracle(
    uhf_oracle_case,
):
    operator = uhf_oracle_case.coupled_operator
    eigenvalues = np.linalg.eigvalsh(operator)
    diagnostics = uhf_oracle_case.response.diagnostics

    assert operator.shape == (22, 22)
    np.testing.assert_allclose(
        operator,
        operator.T,
        rtol=0.0,
        atol=2.0e-14,
    )
    assert diagnostics.alpha_response_dimension == 10
    assert diagnostics.beta_response_dimension == 12
    assert diagnostics.response_dimension == 22
    assert eigenvalues[0] > 0.08
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
    assert np.linalg.norm(operator[:10, 10:]) > 0.1
    assert np.linalg.norm(operator[10:, :10]) > 0.1


def test_coupled_response_residual_matches_independent_reconstruction(
    uhf_oracle_case,
):
    alpha_residual, beta_residual = (
        uhf_oracle_case.coupled_response_residual()
    )
    response = uhf_oracle_case.response
    diagnostics = response.diagnostics

    np.testing.assert_allclose(
        alpha_residual,
        response.alpha_orbital_response_residual,
        rtol=0.0,
        atol=3.0e-14,
    )
    np.testing.assert_allclose(
        beta_residual,
        response.beta_orbital_response_residual,
        rtol=0.0,
        atol=3.0e-14,
    )
    alpha_maximum = float(np.max(np.abs(alpha_residual)))
    beta_maximum = float(np.max(np.abs(beta_residual)))
    residual_rms = float(
        np.sqrt(
            (
                np.sum(np.square(alpha_residual))
                + np.sum(np.square(beta_residual))
            )
            / (alpha_residual.size + beta_residual.size)
        )
    )
    np.testing.assert_allclose(
        diagnostics.alpha_maximum_residual,
        alpha_maximum,
        rtol=2.0e-6,
        atol=3.0e-14,
    )
    np.testing.assert_allclose(
        diagnostics.beta_maximum_residual,
        beta_maximum,
        rtol=2.0e-6,
        atol=3.0e-14,
    )
    np.testing.assert_allclose(
        diagnostics.maximum_residual,
        max(alpha_maximum, beta_maximum),
        rtol=2.0e-6,
        atol=3.0e-14,
    )
    np.testing.assert_allclose(
        diagnostics.residual_rms,
        residual_rms,
        rtol=2.0e-6,
        atol=3.0e-14,
    )
    assert diagnostics.maximum_residual < diagnostics.residual_tolerance


def test_coupled_residual_detects_decoupling_and_metric_omission(
    uhf_oracle_case,
):
    operator = uhf_oracle_case.coupled_operator.copy()
    operator[:10, 10:] = 0.0
    operator[10:, :10] = 0.0
    decoupled_residuals = uhf_oracle_case.coupled_response_residual(
        operator_matrix=operator
    )
    metric_omitted_residuals = uhf_oracle_case.coupled_response_residual(
        metric_density_response=np.zeros_like(
            uhf_oracle_case.metric_density_response
        )
    )

    assert max(
        np.max(np.abs(residual)) for residual in decoupled_residuals
    ) > 1.0e-3
    assert max(
        np.max(np.abs(residual)) for residual in metric_omitted_residuals
    ) > 1.0e-3


def test_response_diagnostics_cover_every_spin_invariant(uhf_oracle_case):
    diagnostics = uhf_oracle_case.response.diagnostics

    assert diagnostics.minimum_alpha_orbital_gap > 1.0
    assert diagnostics.minimum_beta_orbital_gap > 0.6
    assert diagnostics.operator_minimum_eigenvalue > 0.08
    assert diagnostics.operator_condition_number < 200.0
    assert diagnostics.operator_symmetry_residual < 1.0e-12
    assert diagnostics.alpha_metric_residual < 1.0e-12
    assert diagnostics.beta_metric_residual < 1.0e-12
    assert diagnostics.alpha_idempotency_residual < 1.0e-12
    assert diagnostics.beta_idempotency_residual < 1.0e-12
    assert diagnostics.alpha_particle_number_residual < 1.0e-12
    assert diagnostics.beta_particle_number_residual < 1.0e-12
    assert diagnostics.density_reconstruction_residual < 1.0e-12
    assert diagnostics.refinement_cycles == len(diagnostics.residual_history) - 1
    assert diagnostics.maximum_residual == diagnostics.residual_history[-1]
