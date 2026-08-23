import numpy as np
import pytest


def test_displaced_references_preserve_root_grid_and_descriptor_domain(
    rks_oracle_case,
):
    assert len(
        rks_oracle_case.occupied_subspace_minimum_singular_values
    ) == 54
    for minimum_singular_value in (
        rks_oracle_case.occupied_subspace_minimum_singular_values.values()
    ):
        assert np.isfinite(minimum_singular_value)
        assert minimum_singular_value > 0.99
    assert rks_oracle_case.minimum_orbital_gap > 0.36
    assert rks_oracle_case.minimum_descriptor_gap > 5.0e-2
    response = rks_oracle_case.response
    assert response.functional_provenance.components == (
        (1, 1.0),
        (7, 1.0),
    )
    assert response.functional_provenance.xc_type == "LDA"
    assert response.functional_provenance.hybrid_coefficient == 0.0
    assert response.functional_provenance.range_separation == (
        0.0,
        0.0,
        0.0,
    )
    assert not response.functional_provenance.has_nlc
    assert response.grid_provenance.prune is None
    assert response.grid_provenance.point_count == 3000
    assert response.grid_provenance.atom_grid == (
        ("H", (20, 50)),
        ("O", (20, 50)),
    )


@pytest.mark.parametrize(
    ("step", "absolute_tolerance"),
    [
        pytest.param(1.0e-3, 5.0e-7, id="coarse"),
        pytest.param(3.0e-4, 8.0e-8, id="balanced"),
        pytest.param(1.0e-4, 1.0e-7, id="fine"),
    ],
)
def test_complete_ao_density_response_matches_fresh_rks(
    rks_oracle_case,
    step,
    absolute_tolerance,
):
    analytic = rks_oracle_case.response.density_response
    finite_difference = rks_oracle_case.finite_difference("density", step)

    assert analytic.shape == (3, 3, 7, 7)
    np.testing.assert_allclose(
        analytic,
        finite_difference,
        rtol=2.0e-6,
        atol=absolute_tolerance,
    )


def test_response_matches_independent_dense_lda_cpks_oracle(
    rks_oracle_case,
):
    response = rks_oracle_case.response
    independent = rks_oracle_case.independent
    solution = independent.solution

    for actual, expected in (
        (response.mo_response, solution.mo_response),
        (response.mo_response_metric, solution.mo_response_metric),
        (
            response.mo_response_occupied_virtual,
            solution.mo_response_occupied_virtual,
        ),
        (response.density_response, solution.density_response),
        (
            response.density_response_metric,
            solution.density_response_metric,
        ),
        (
            response.density_response_occupied_virtual,
            solution.density_response_occupied_virtual,
        ),
    ):
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=2.0e-10,
            atol=2.0e-10,
        )
    np.testing.assert_allclose(
        response.density_response,
        response.density_response_metric
        + response.density_response_occupied_virtual,
        rtol=0.0,
        atol=2.0e-15,
    )
    np.testing.assert_allclose(
        response.orbital_response_residual,
        solution.residual,
        rtol=0.0,
        atol=1.0e-9,
    )
    assert np.max(np.abs(solution.residual)) < 3.0e-15


def test_independent_cpks_operator_contains_coulomb_and_fxc(
    rks_oracle_case,
):
    independent = rks_oracle_case.independent
    operator = independent.operator

    assert operator.shape == (10, 10)
    np.testing.assert_allclose(
        operator,
        independent.gap_operator
        + independent.coulomb_operator
        + independent.fxc_operator,
        rtol=0.0,
        atol=2.0e-15,
    )
    np.testing.assert_allclose(
        operator,
        operator.T,
        rtol=0.0,
        atol=2.0e-14,
    )
    assert np.linalg.norm(independent.coulomb_operator) > 0.1
    assert np.linalg.norm(independent.fxc_operator) > 0.1
    eigenvalues = np.linalg.eigvalsh(operator)
    assert eigenvalues[0] > 0.43
    assert eigenvalues[-1] < 19.0
    assert np.linalg.cond(operator) < 44.0
    diagnostics = rks_oracle_case.response.diagnostics
    np.testing.assert_allclose(
        diagnostics.quadrature_electron_count,
        independent.quadrature_electron_count,
        rtol=0.0,
        atol=2.0e-12,
    )
    assert abs(
        diagnostics.quadrature_electron_count
        - rks_oracle_case.reference.mol.nelectron
    ) < 1.0e-3
def test_xc_nuclear_hamiltonian_contains_every_grid_motion_term(
    rks_oracle_case,
):
    response = rks_oracle_case.response
    independent = rks_oracle_case.independent

    for actual, expected in (
        (
            response.hamiltonian_derivative_fixed_grid,
            independent.hamiltonian_derivative_fixed_grid,
        ),
        (
            response.xc_hamiltonian_derivative_grid_coordinate,
            independent.xc_hamiltonian_derivative_grid_coordinate,
        ),
        (
            response.xc_hamiltonian_derivative_grid_weight,
            independent.xc_hamiltonian_derivative_grid_weight,
        ),
        (
            response.hamiltonian_derivative,
            independent.hamiltonian_derivative,
        ),
    ):
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=2.0e-12,
            atol=2.0e-12,
        )
    np.testing.assert_allclose(
        response.hamiltonian_derivative,
        response.hamiltonian_derivative_fixed_grid
        + response.xc_hamiltonian_derivative_grid_coordinate
        + response.xc_hamiltonian_derivative_grid_weight,
        rtol=0.0,
        atol=2.0e-15,
    )
    assert np.max(
        np.abs(response.xc_hamiltonian_derivative_grid_coordinate)
    ) > 0.29
    assert np.max(
        np.abs(response.xc_hamiltonian_derivative_grid_weight)
    ) > 0.29
    assert np.max(
        np.abs(
            response.xc_hamiltonian_derivative_grid_coordinate
            + response.xc_hamiltonian_derivative_grid_weight
        )
    ) > 3.0e-3


def test_grid_hamiltonian_parts_match_fresh_fixed_density_fock_differences(
    rks_oracle_case,
):
    independent = rks_oracle_case.independent

    np.testing.assert_allclose(
        independent.xc_hamiltonian_derivative_grid_coordinate,
        independent.fresh_grid_fd_coordinate,
        rtol=2.0e-8,
        atol=2.0e-9,
    )
    np.testing.assert_allclose(
        independent.xc_hamiltonian_derivative_grid_weight,
        independent.fresh_grid_fd_weight,
        rtol=2.0e-8,
        atol=2.0e-9,
    )
    np.testing.assert_allclose(
        independent.xc_hamiltonian_derivative_grid_coordinate
        + independent.xc_hamiltonian_derivative_grid_weight,
        independent.fresh_grid_fd_total,
        rtol=2.0e-8,
        atol=2.0e-9,
    )


def test_density_response_satisfies_nonorthogonal_ao_invariants(
    rks_oracle_case,
):
    response = rks_oracle_case.response
    density_response = response.density_response
    density = rks_oracle_case.method.ao_density()
    overlap = np.asarray(rks_oracle_case.reference.get_ovlp())
    overlap_finite_difference = rks_oracle_case.finite_difference(
        "overlap",
        1.0e-4,
    )

    np.testing.assert_allclose(
        response.overlap_derivative,
        rks_oracle_case.independent.overlap_derivative,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        response.overlap_derivative,
        overlap_finite_difference,
        rtol=2.0e-8,
        atol=3.0e-9,
    )
    particle_number_residual = (
        np.einsum("...ij,ji->...", density_response, overlap)
        + np.einsum(
            "ij,...ji->...",
            density,
            response.overlap_derivative,
        )
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
            response.overlap_derivative,
            density,
        )
        + np.einsum(
            "ij,jk,...kl->...il",
            density,
            overlap,
            density_response,
        )
        - 2.0 * density_response
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
    np.testing.assert_allclose(
        density_response.sum(axis=0),
        np.zeros_like(density_response[0]),
        rtol=0.0,
        atol=2.0e-10,
    )
    diagnostics = response.diagnostics
    assert diagnostics.maximum_residual < 1.0e-9
    assert diagnostics.metric_residual < 2.0e-10
    assert diagnostics.idempotency_residual < 2.0e-10
    assert diagnostics.particle_number_residual < 2.0e-10
    assert diagnostics.translation_residual < 2.0e-10


@pytest.mark.parametrize(
    ("omission", "minimum_error"),
    [
        pytest.param("without_coulomb", 0.5, id="coulomb"),
        pytest.param("without_fxc", 0.05, id="fxc"),
        pytest.param("without_metric", 0.2, id="metric"),
        pytest.param("without_ao_motion", 1.0, id="ao-motion"),
        pytest.param("without_grid_response", 4.0e-3, id="all-grid-response"),
        pytest.param("without_grid_coordinate", 0.2, id="grid-coordinate"),
        pytest.param("without_grid_weight", 0.2, id="grid-weight"),
    ],
)
def test_fresh_rks_density_detects_each_omitted_response_component(
    rks_oracle_case,
    omission,
    minimum_error,
):
    finite_difference = rks_oracle_case.finite_difference(
        "density",
        3.0e-4,
    )
    omitted = getattr(rks_oracle_case.independent, omission)

    assert np.max(
        np.abs(omitted.density_response - finite_difference)
    ) > minimum_error
