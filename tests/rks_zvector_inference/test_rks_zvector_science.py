from copy import deepcopy

import numpy as np
import pytest
import torch


def test_rks_adjoint_matches_independent_dense_grid_transpose_oracle(
    rks_oracle_case,
    rks_zvector_oracle,
):
    adjoint = rks_oracle_case.method.adjoint()
    oracle = rks_zvector_oracle

    np.testing.assert_allclose(
        oracle.operator,
        rks_oracle_case.independent.operator,
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        adjoint.objective_ao_potential,
        oracle.objective_ao_potential,
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        adjoint.objective_orbital_gradient,
        oracle.objective_orbital_gradient,
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        adjoint.zvector,
        oracle.zvector,
        rtol=2.0e-11,
        atol=2.0e-11,
    )
    np.testing.assert_allclose(
        adjoint.solver_residual,
        oracle.solver_residual,
        rtol=0.0,
        atol=2.0e-11,
    )
    np.testing.assert_allclose(
        adjoint.transpose_residual,
        oracle.transpose_residual,
        rtol=0.0,
        atol=2.0e-11,
    )
    np.testing.assert_allclose(
        adjoint.physical_residual,
        oracle.transpose_residual,
        rtol=0.0,
        atol=2.0e-11,
    )
    np.testing.assert_allclose(
        adjoint.adjoint_ao_density,
        oracle.adjoint_ao_density,
        rtol=2.0e-11,
        atol=2.0e-11,
    )
    np.testing.assert_allclose(
        adjoint.adjoint_ao_potential,
        oracle.adjoint_ao_potential,
        rtol=2.0e-11,
        atol=2.0e-11,
    )
    assert np.max(
        np.abs(
            oracle.objective_orbital_gradient
            - oracle.one_sided_objective_gradient
        )
    ) > 1.0e-3
    diagnostics = adjoint.diagnostics
    assert diagnostics.solve_count == 1
    assert diagnostics.response_dimension == 10
    assert diagnostics.maximum_solver_residual < 1.0e-9
    assert diagnostics.maximum_transpose_residual < 1.0e-9
    assert diagnostics.maximum_physical_residual < 1.0e-9
    assert diagnostics.operator_is_self_adjoint is True
    assert diagnostics.functional_components == ((1, 1.0), (7, 1.0))
    assert diagnostics.grid_point_count == 3000


def test_independent_rks_adjoint_operator_requires_coulomb_and_lda_fxc(
    rks_oracle_case,
    rks_zvector_oracle,
):
    response_oracle = rks_oracle_case.independent
    objective = rks_zvector_oracle.objective_orbital_gradient.reshape(-1)
    without_coulomb = np.linalg.solve(
        (response_oracle.gap_operator + response_oracle.fxc_operator).T,
        objective,
    )
    without_fxc = np.linalg.solve(
        (response_oracle.gap_operator + response_oracle.coulomb_operator).T,
        objective,
    )

    np.testing.assert_allclose(
        rks_zvector_oracle.operator,
        response_oracle.gap_operator
        + response_oracle.coulomb_operator
        + response_oracle.fxc_operator,
        rtol=0.0,
        atol=2.0e-12,
    )
    assert np.linalg.norm(response_oracle.coulomb_operator) > 0.1
    assert np.linalg.norm(response_oracle.fxc_operator) > 0.1
    assert np.max(
        np.abs(rks_zvector_oracle.zvector.reshape(-1) - without_coulomb)
    ) > 1.0e-3
    assert np.max(
        np.abs(rks_zvector_oracle.zvector.reshape(-1) - without_fxc)
    ) > 1.0e-4


def test_every_rks_adjoint_nuclear_partition_matches_independent_formula(
    rks_oracle_case,
    rks_zvector_oracle,
):
    adjoint = rks_oracle_case.method.adjoint()
    oracle = rks_zvector_oracle

    for field in (
        "correction_gradient_metric",
        "correction_gradient_adjoint_fixed_grid",
        "correction_gradient_adjoint_grid_coordinate",
        "correction_gradient_adjoint_grid_weight",
        "correction_gradient_adjoint_nuclear",
        "correction_gradient_adjoint_metric",
        "correction_gradient_occupied_virtual",
        "correction_gradient_response",
    ):
        np.testing.assert_allclose(
            getattr(adjoint, field),
            getattr(oracle, field),
            rtol=3.0e-11,
            atol=3.0e-11,
            err_msg=field,
        )
    np.testing.assert_allclose(
        oracle.correction_gradient_adjoint_metric,
        oracle.correction_gradient_adjoint_metric_overlap_form,
        rtol=2.0e-11,
        atol=2.0e-11,
    )
    np.testing.assert_allclose(
        adjoint.correction_gradient_adjoint_nuclear,
        adjoint.correction_gradient_adjoint_fixed_grid
        + adjoint.correction_gradient_adjoint_grid_coordinate
        + adjoint.correction_gradient_adjoint_grid_weight,
        rtol=0.0,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        adjoint.correction_gradient_occupied_virtual,
        adjoint.correction_gradient_adjoint_nuclear
        + adjoint.correction_gradient_adjoint_metric,
        rtol=0.0,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        adjoint.correction_gradient_response,
        adjoint.correction_gradient_metric
        + adjoint.correction_gradient_occupied_virtual,
        rtol=0.0,
        atol=2.0e-12,
    )


def test_zvector_metric_and_occupied_virtual_match_direct_density_contractions(
    rks_oracle_case,
    rks_zvector_oracle,
):
    direct = rks_oracle_case.method.nuc_grad_method(backend="direct").run()
    adjoint = rks_oracle_case.method.adjoint()
    objective = rks_zvector_oracle.objective_ao_potential
    direct_metric = np.einsum(
        "ij,...ij->...",
        objective,
        direct.response_result.density_response_metric,
    )
    direct_occupied_virtual = np.einsum(
        "ij,...ij->...",
        objective,
        direct.response_result.density_response_occupied_virtual,
    )

    np.testing.assert_allclose(
        adjoint.correction_gradient_metric,
        direct_metric,
        rtol=3.0e-11,
        atol=3.0e-11,
    )
    np.testing.assert_allclose(
        adjoint.correction_gradient_occupied_virtual,
        direct_occupied_virtual,
        rtol=3.0e-10,
        atol=3.0e-10,
    )
    np.testing.assert_allclose(
        adjoint.correction_gradient_response,
        direct.correction_gradient_response,
        rtol=3.0e-10,
        atol=3.0e-10,
    )


def test_zvector_matches_direct_oracle_by_all_common_gradient_partitions(
    rks_oracle_case,
):
    method = rks_oracle_case.method
    direct = method.nuc_grad_method(backend="direct").run()
    zvector = method.nuc_grad_method(backend="zvector").run()

    assert direct.backend == "direct"
    assert zvector.backend == "zvector"
    for field in (
        "reference_gradient",
        "reference_gradient_without_grid_response",
        "reference_gradient_xc_grid_coordinate",
        "reference_gradient_xc_grid_weight",
        "correction_gradient_explicit",
        "correction_gradient_metric",
        "correction_gradient_occupied_virtual",
        "correction_gradient_response",
        "correction_gradient",
        "de_full",
        "de",
    ):
        np.testing.assert_allclose(
            getattr(zvector, field),
            getattr(direct, field),
            rtol=4.0e-10,
            atol=4.0e-10,
            err_msg=field,
        )
    np.testing.assert_allclose(
        zvector.correction_gradient_adjoint_nuclear,
        zvector.correction_gradient_adjoint_fixed_grid
        + zvector.correction_gradient_adjoint_grid_coordinate
        + zvector.correction_gradient_adjoint_grid_weight,
        rtol=0.0,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        zvector.correction_gradient,
        zvector.correction_gradient_explicit
        + zvector.correction_gradient_response,
        rtol=0.0,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        zvector.de_full,
        zvector.reference_gradient + zvector.correction_gradient,
        rtol=0.0,
        atol=2.0e-12,
    )
    assert np.max(np.abs(zvector.correction_gradient_explicit)) > 1.0e-3


@pytest.mark.parametrize(
    ("step", "absolute_tolerance"),
    [
        pytest.param(1.0e-3, 5.0e-7, id="coarse"),
        pytest.param(3.0e-4, 8.0e-8, id="balanced"),
        pytest.param(1.0e-4, 1.0e-7, id="fine"),
    ],
)
def test_zvector_total_gradient_matches_fresh_rks_total_energy_fd(
    rks_oracle_case,
    step,
    absolute_tolerance,
):
    analytic = rks_oracle_case.method.gradient(backend="zvector")
    finite_difference = rks_oracle_case.finite_difference(
        "total_energy",
        step,
    )

    np.testing.assert_allclose(
        analytic,
        finite_difference,
        rtol=3.0e-6,
        atol=absolute_tolerance,
    )


@pytest.mark.parametrize(
    ("field", "minimum_norm"),
    [
        pytest.param(
            "correction_gradient_adjoint_fixed_grid",
            1.0e-3,
            id="fixed-grid",
        ),
        pytest.param(
            "correction_gradient_adjoint_grid_coordinate",
            1.0e-4,
            id="grid-coordinate",
        ),
        pytest.param(
            "correction_gradient_adjoint_grid_weight",
            1.0e-4,
            id="grid-weight",
        ),
        pytest.param(
            "correction_gradient_metric",
            1.0e-3,
            id="objective-metric",
        ),
        pytest.param(
            "correction_gradient_adjoint_metric",
            1.0e-4,
            id="adjoint-metric",
        ),
        pytest.param(
            "correction_gradient_adjoint_ao_motion",
            1.0e-3,
            id="ao-motion",
        ),
    ],
)
def test_each_independent_response_partition_is_numerically_required(
    rks_zvector_oracle,
    field,
    minimum_norm,
):
    component = getattr(rks_zvector_oracle, field)

    assert np.max(np.abs(component)) > minimum_norm


def _constant_model(template, bias, energy_constant):
    model = deepcopy(template)
    with torch.no_grad():
        model.linear.weight.zero_()
        model.linear.bias.fill_(bias)
        for parameter in model.densenet.parameters():
            parameter.zero_()
        model.energy_const.fill_(energy_constant)
    return model.eval()


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

    model = _constant_model(
        rks_oracle_case.model,
        bias,
        energy_constant,
    )
    method = RKSDeePHF(
        rks_oracle_case.reference,
        model,
        projector_basis=model._pbas,
    )
    method.kernel()
    driver = method.nuc_grad_method(backend="zvector").run()

    np.testing.assert_allclose(
        method.e_corr,
        rks_oracle_case.reference.mol.natm * bias + energy_constant,
        rtol=0.0,
        atol=2.0e-15,
    )
    for field in (
        "correction_gradient_explicit",
        "correction_gradient_metric",
        "correction_gradient_adjoint_fixed_grid",
        "correction_gradient_adjoint_grid_coordinate",
        "correction_gradient_adjoint_grid_weight",
        "correction_gradient_adjoint_nuclear",
        "correction_gradient_adjoint_metric",
        "correction_gradient_occupied_virtual",
        "correction_gradient_response",
        "correction_gradient",
    ):
        np.testing.assert_allclose(
            getattr(driver, field),
            np.zeros((rks_oracle_case.reference.mol.natm, 3)),
            rtol=0.0,
            atol=2.0e-15,
            err_msg=field,
        )
    np.testing.assert_allclose(
        driver.de_full,
        rks_oracle_case.independent.native_gradient,
        rtol=2.0e-12,
        atol=2.0e-12,
    )
