import numpy as np

from deepks.data.fields import select_fields


def test_deephf_exports_distinct_explicit_response_and_relaxed_fields(
    force_generation_case,
):
    case = force_generation_case
    method = case.teacher_method
    gradient = case.teacher_gradient
    fields = select_fields(
        [
            "e_base",
            "e_corr",
            "e_tot",
            "ao_density",
            "descriptor",
            "converged",
            "f_reference_variational",
            "f_corr_explicit",
            "f_corr",
            "f_tot",
            "dq_dR_explicit",
            "dq_dR_response",
            "dq_dR_relaxed",
            "f_corr_target",
        ]
    )
    scf_values = {
        field.name: field.calculate(method)
        for field in fields["scf"]
    }
    gradient_values = {
        field.name: field.calculate(
            gradient,
            **({"force": case.target_force} if "force" in field.required_labels else {}),
        )
        for field in fields["gradient"]
    }

    assert scf_values["converged"] is True
    np.testing.assert_allclose(scf_values["e_base"], case.reference.e_tot)
    np.testing.assert_allclose(scf_values["e_corr"], method.e_corr)
    np.testing.assert_allclose(scf_values["e_tot"], method.e_tot)
    np.testing.assert_allclose(scf_values["ao_density"], method.ao_density())
    np.testing.assert_allclose(scf_values["descriptor"], method.descriptor())
    np.testing.assert_allclose(
        gradient_values["dq_dR_explicit"],
        gradient.dq_dR_explicit,
    )
    np.testing.assert_allclose(
        gradient_values["dq_dR_response"],
        gradient.dq_dR_response,
    )
    np.testing.assert_allclose(
        gradient_values["dq_dR_relaxed"],
        gradient.dq_dR_explicit + gradient.dq_dR_response,
    )
    np.testing.assert_allclose(
        gradient_values["f_reference_variational"],
        -gradient.reference_gradient,
    )
    np.testing.assert_allclose(
        gradient_values["f_corr_explicit"],
        -gradient.correction_gradient_explicit,
    )
    np.testing.assert_allclose(
        gradient_values["f_corr"],
        -gradient.correction_gradient,
    )
    np.testing.assert_allclose(gradient_values["f_tot"], case.target_force)
    np.testing.assert_allclose(
        gradient_values["f_corr_target"],
        -gradient.correction_gradient,
    )
