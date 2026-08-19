import numpy as np
import pytest


def test_descriptor_response_is_the_complete_density_contraction(
    rhf_oracle_case,
):
    method = rhf_oracle_case.method
    response = method.dq_dR_response(response=rhf_oracle_case.response)
    explicit = method.dq_dR_explicit()
    relaxed = method.dq_dR_relaxed(response=rhf_oracle_case.response)
    expected_response = np.einsum(
        "apij,bxij->bxap",
        method.dq_dP(),
        rhf_oracle_case.response.density_response,
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
    assert np.max(np.abs(response)) > 0.1
    np.testing.assert_allclose(
        relaxed.sum(axis=0),
        np.zeros_like(relaxed[0]),
        rtol=0.0,
        atol=2.0e-10,
    )
    assert rhf_oracle_case.minimum_descriptor_gap > 4.0e-2
    diagnostics = method.validate_force_compatibility()
    assert diagnostics.structural_zero_blocks == ()
    assert diagnostics.minimum_scaled_gap > 1.0e4


@pytest.mark.parametrize(
    ("step", "absolute_tolerance"),
    [
        pytest.param(1.0e-3, 6.0e-7, id="coarse"),
        pytest.param(3.0e-4, 1.0e-7, id="balanced"),
        pytest.param(1.0e-4, 1.0e-7, id="fine"),
    ],
)
def test_relaxed_descriptor_matches_displaced_rhf(
    rhf_oracle_case,
    step,
    absolute_tolerance,
):
    relaxed = rhf_oracle_case.method.dq_dR_relaxed(
        response=rhf_oracle_case.response
    )
    finite_difference = rhf_oracle_case.finite_difference("descriptor", step)

    np.testing.assert_allclose(
        relaxed,
        finite_difference,
        rtol=2.0e-6,
        atol=absolute_tolerance,
    )


def test_explicit_descriptor_derivative_cannot_replace_relaxed_response(
    rhf_oracle_case,
):
    explicit = rhf_oracle_case.method.dq_dR_explicit()
    finite_difference = rhf_oracle_case.finite_difference(
        "descriptor",
        3.0e-4,
    )

    assert np.max(np.abs(finite_difference - explicit)) > 0.1
