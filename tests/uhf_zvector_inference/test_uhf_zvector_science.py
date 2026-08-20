import numpy as np
import pytest
import torch
from pyscf import gto, scf

from deepks.deephf import UHFDeePHF
from deepks.model.model import CorrNet

ORACLE_PROJECTOR_BASIS = [[0, [0.8, 1.0]], [1, [0.3, 1.0]]]


def test_coupled_adjoint_matches_independent_ao_oracle(
    uhf_oracle_case,
    independent_uhf_adjoint_oracle,
):
    adjoint = uhf_oracle_case.method.adjoint()
    oracle = independent_uhf_adjoint_oracle

    np.testing.assert_allclose(
        adjoint.objective_ao_potential,
        oracle.objective_ao_potential,
        rtol=0.0,
        atol=2.0e-15,
    )
    for spin_name in ("alpha", "beta"):
        for field_name in (
            "objective_orbital_gradient",
            "zvector",
            "adjoint_ao_density",
            "adjoint_ao_potential",
        ):
            np.testing.assert_allclose(
                getattr(adjoint, f"{spin_name}_{field_name}"),
                getattr(oracle, f"{spin_name}_{field_name}"),
                rtol=2.0e-11,
                atol=2.0e-13,
            )
    for field_name in (
        "correction_gradient_metric_spin",
        "correction_gradient_metric",
        "correction_gradient_adjoint_nuclear_spin",
        "correction_gradient_adjoint_nuclear",
        "correction_gradient_adjoint_metric_spin",
        "correction_gradient_adjoint_metric",
        "correction_gradient_occupied_virtual_spin",
        "correction_gradient_occupied_virtual",
        "correction_gradient_response",
    ):
        np.testing.assert_allclose(
            getattr(adjoint, field_name),
            getattr(oracle, field_name),
            rtol=2.0e-11,
            atol=2.0e-13,
        )
    assert adjoint.diagnostics.solve_count == 1
    assert (
        adjoint.diagnostics.response_dimension
        == adjoint.diagnostics.alpha_response_dimension
        + adjoint.diagnostics.beta_response_dimension
    )
    assert adjoint.diagnostics.maximum_solver_residual < 1.0e-12
    assert adjoint.diagnostics.maximum_transpose_residual < 1.0e-12
    assert adjoint.diagnostics.maximum_physical_residual < 1.0e-12


def test_transpose_rhs_is_bilateral_and_couples_both_spins(
    uhf_oracle_case,
    independent_uhf_adjoint_oracle,
):
    oracle = independent_uhf_adjoint_oracle
    objective = np.concatenate(
        (
            oracle.alpha_objective_orbital_gradient.reshape(-1),
            oracle.beta_objective_orbital_gradient.reshape(-1),
        )
    )
    zvector = np.concatenate(
        (
            oracle.alpha_zvector.reshape(-1),
            oracle.beta_zvector.reshape(-1),
        )
    )

    np.testing.assert_allclose(
        oracle.operator.T @ zvector,
        objective,
        rtol=0.0,
        atol=2.0e-13,
    )
    assert np.linalg.norm(objective - oracle.one_sided_objective_gradient) > 1.0e-3
    alpha_dimension = oracle.alpha_zvector.size
    assert np.linalg.norm(oracle.operator[:alpha_dimension, alpha_dimension:]) > 1.0e-3
    assert np.linalg.norm(oracle.operator[alpha_dimension:, :alpha_dimension]) > 1.0e-3


def test_metric_and_occupied_virtual_terms_match_direct_oracle(
    uhf_oracle_case,
    independent_uhf_adjoint_oracle,
):
    oracle = independent_uhf_adjoint_oracle

    np.testing.assert_allclose(
        oracle.correction_gradient_metric_spin,
        oracle.direct_correction_gradient_metric_spin,
        rtol=0.0,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        oracle.correction_gradient_occupied_virtual,
        oracle.direct_correction_gradient_occupied_virtual,
        rtol=2.0e-9,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        oracle.correction_gradient_response,
        oracle.direct_correction_gradient_response,
        rtol=2.0e-9,
        atol=2.0e-12,
    )
    assert np.linalg.norm(oracle.correction_gradient_metric_spin[0]) > 1.0e-3
    assert np.linalg.norm(oracle.correction_gradient_metric_spin[1]) > 1.0e-3
    assert np.linalg.norm(oracle.correction_gradient_adjoint_nuclear_spin[0]) > 1.0e-4
    assert np.linalg.norm(oracle.correction_gradient_adjoint_nuclear_spin[1]) > 1.0e-4
    assert np.linalg.norm(oracle.correction_gradient_adjoint_metric_spin[0]) > 1.0e-4
    assert np.linalg.norm(oracle.correction_gradient_adjoint_metric_spin[1]) > 1.0e-4


@pytest.mark.parametrize(
    ("step", "absolute_tolerance"),
    [
        pytest.param(1.0e-3, 8.0e-7, id="coarse"),
        pytest.param(3.0e-4, 1.0e-7, id="balanced"),
        pytest.param(1.0e-4, 5.0e-8, id="fine"),
    ],
)
def test_zvector_total_gradient_matches_fresh_uhf_energy_finite_difference(
    uhf_oracle_case,
    step,
    absolute_tolerance,
):
    driver = uhf_oracle_case.method.nuc_grad_method(backend="zvector")
    gradient = driver.kernel()
    finite_difference = uhf_oracle_case.finite_difference(
        "total_energy",
        step,
    )

    np.testing.assert_allclose(
        gradient,
        finite_difference,
        rtol=3.0e-6,
        atol=absolute_tolerance,
    )
    np.testing.assert_allclose(
        gradient,
        uhf_oracle_case.gradient,
        rtol=2.0e-10,
        atol=2.0e-12,
    )


def test_zvector_driver_preserves_all_spin_and_response_partitions(
    uhf_oracle_case,
):
    driver = uhf_oracle_case.method.nuc_grad_method(backend="zvector").run()

    np.testing.assert_allclose(
        driver.correction_gradient_occupied_virtual_spin,
        driver.correction_gradient_adjoint_nuclear_spin
        + driver.correction_gradient_adjoint_metric_spin,
        rtol=0.0,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        driver.correction_gradient_response_spin,
        driver.correction_gradient_metric_spin
        + driver.correction_gradient_occupied_virtual_spin,
        rtol=0.0,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        driver.correction_gradient_spin,
        driver.correction_gradient_explicit_spin
        + driver.correction_gradient_response_spin,
        rtol=0.0,
        atol=2.0e-13,
    )
    for total_name, spin_name in (
        ("correction_gradient_explicit", "correction_gradient_explicit_spin"),
        ("correction_gradient_metric", "correction_gradient_metric_spin"),
        (
            "correction_gradient_adjoint_nuclear",
            "correction_gradient_adjoint_nuclear_spin",
        ),
        (
            "correction_gradient_adjoint_metric",
            "correction_gradient_adjoint_metric_spin",
        ),
        (
            "correction_gradient_occupied_virtual",
            "correction_gradient_occupied_virtual_spin",
        ),
        ("correction_gradient", "correction_gradient_spin"),
    ):
        np.testing.assert_allclose(
            getattr(driver, total_name),
            getattr(driver, spin_name).sum(axis=0),
            rtol=0.0,
            atol=2.0e-13,
        )
    np.testing.assert_allclose(
        driver.de_full,
        driver.reference_gradient + driver.correction_gradient,
        rtol=0.0,
        atol=2.0e-13,
    )


def _constant_model(bias, energy_constant):
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
    ("bias", "energy_constant"),
    [
        pytest.param(0.0, 0.0, id="zero"),
        pytest.param(0.017, 0.019, id="constant"),
    ],
)
def test_zero_and_constant_corrections_reduce_to_native_uhf_gradient(
    uhf_oracle_case,
    bias,
    energy_constant,
):
    method = UHFDeePHF(
        uhf_oracle_case.reference,
        _constant_model(bias, energy_constant),
        projector_basis=ORACLE_PROJECTOR_BASIS,
    )
    method.kernel()
    driver = method.nuc_grad_method(backend="zvector")
    actual = driver.kernel()
    native = np.asarray(
        uhf_oracle_case.reference.nuc_grad_method().kernel()
    )

    np.testing.assert_allclose(
        driver.correction_gradient,
        np.zeros((3, 3)),
        rtol=0.0,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(actual, native, rtol=2.0e-12, atol=2.0e-12)
    assert driver.adjoint_diagnostics.solve_count == 1


def test_distinct_bent_bh2_oracle_matches_direct_and_total_energy_fd(
    uhf_oracle_case,
):
    coordinates = np.array(
        [
            [0.22, -0.17, 0.13],
            [1.62, 0.29, -0.24],
            [-0.51, 1.47, 0.63],
        ],
        dtype=np.float64,
    )
    step = 3.0e-4
    model = uhf_oracle_case.model

    def fresh_reference(current_coordinates):
        molecule = gto.M(
            atom=list(zip(("B", "H", "H"), current_coordinates)),
            basis="sto-3g",
            unit="Bohr",
            charge=0,
            spin=1,
            symmetry=False,
            cart=False,
            verbose=0,
        )
        reference = scf.UHF(molecule)
        reference.conv_tol = 1.0e-13
        reference.conv_tol_grad = 1.0e-10
        reference.conv_tol_cpscf = 1.0e-12
        reference.max_cycle = 100
        reference.kernel()
        assert reference.converged
        return reference

    reference = fresh_reference(coordinates)
    method = UHFDeePHF(
        reference,
        model,
        projector_basis=ORACLE_PROJECTOR_BASIS,
    )
    method.kernel()
    zdriver = method.nuc_grad_method(backend="zvector").run()
    direct = method.gradient(backend="direct")
    finite_difference = np.empty((3, 3), dtype=np.float64)
    for atom_index in range(3):
        for coordinate_index in range(3):
            energies = []
            for direction in (-1, 1):
                displaced = coordinates.copy()
                displaced[atom_index, coordinate_index] += direction * step
                displaced_method = UHFDeePHF(
                    fresh_reference(displaced),
                    model,
                    projector_basis=ORACLE_PROJECTOR_BASIS,
                )
                energies.append(float(displaced_method.kernel()))
            finite_difference[atom_index, coordinate_index] = (
                energies[1] - energies[0]
            ) / (2.0 * step)

    np.testing.assert_allclose(
        zdriver.de_full,
        direct,
        rtol=2.0e-9,
        atol=1.0e-11,
    )
    np.testing.assert_allclose(
        zdriver.de_full,
        finite_difference,
        rtol=3.0e-6,
        atol=1.0e-7,
    )
    np.testing.assert_allclose(
        zdriver.de_full.sum(axis=0),
        np.zeros(3),
        rtol=0.0,
        atol=2.0e-10,
    )
    assert np.min(
        np.linalg.norm(zdriver.correction_gradient_metric_spin, axis=(1, 2))
    ) > 1.0e-3
    assert np.min(
        np.linalg.norm(
            zdriver.correction_gradient_occupied_virtual_spin,
            axis=(1, 2),
        )
    ) > 1.0e-4
