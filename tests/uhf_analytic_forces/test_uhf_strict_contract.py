from dataclasses import fields, replace

import numpy as np
import pytest
import torch
from pyscf import dft, gto, scf

import deepks.deephf.pyscf_uhf as pyscf_uhf
from deepks.deephf import (
    DeePHF,
    DeePHFCapabilityError,
    RHFDeePHFGradients,
    RHFDeePHFZVectorGradients,
    UHFDeePHF,
    UHFDeePHFGradients,
    UHFResponse,
    UHFResponseAdapter,
    UHFResponseDiagnostics,
    UHFResponseError,
    generate_rhf_force_frame,
    validate_uhf_reference,
)
from deepks.descriptor import DescriptorDifferentiabilityError
from deepks.model.model import CorrNet


COORDINATES = np.array(
    [
        [0.13, -0.21, 0.07],
        [1.731385881594, 0.240389779198, -0.280303161599],
        [-0.367019388329, 0.780700019240, 1.613206142906],
    ],
    dtype=np.float64,
)
ATOMS = ("N", "H", "H")
PROJECTOR_BASIS = [[0, [0.8, 1.0]], [1, [0.3, 1.0]]]


def _make_molecule():
    return gto.M(
        atom=list(zip(ATOMS, COORDINATES)),
        basis="sto-3g",
        unit="Bohr",
        charge=0,
        spin=1,
        symmetry=False,
        cart=False,
        verbose=0,
    )


def _run_uhf():
    reference = scf.UHF(_make_molecule())
    reference.conv_tol = 1.0e-13
    reference.conv_tol_grad = 1.0e-10
    reference.conv_tol_cpscf = 1.0e-12
    reference.max_cycle = 100
    reference.kernel()
    assert reference.converged
    return reference


def _make_model(projector_basis=PROJECTOR_BASIS, input_dim=4):
    model = CorrNet(
        input_dim=input_dim,
        hidden_sizes=(2,),
        proj_basis=projector_basis,
    ).double()
    with torch.no_grad():
        weights = torch.linspace(
            0.011,
            0.011 * input_dim,
            input_dim,
            dtype=torch.float64,
        ).reshape(1, input_dim)
        model.linear.weight.copy_(weights)
        model.linear.bias.fill_(0.007)
        for parameter in model.densenet.parameters():
            parameter.zero_()
    return model.eval()


@pytest.fixture(scope="module")
def native_uhf_reference():
    return _run_uhf()


@pytest.fixture(scope="module")
def strict_uhf_method(native_uhf_reference):
    return UHFDeePHF(
        native_uhf_reference,
        _make_model(),
        projector_basis=PROJECTOR_BASIS,
    )


@pytest.fixture(scope="module")
def strict_uhf_response(strict_uhf_method):
    return strict_uhf_method.response()


def _reseal(response):
    return replace(
        response,
        integrity_fingerprint=pyscf_uhf.uhf_response_integrity_fingerprint(
            response
        ),
    )


def test_native_uhf_reference_and_spin_summed_method_contract(
    native_uhf_reference,
    strict_uhf_method,
):
    assert validate_uhf_reference(native_uhf_reference) is native_uhf_reference
    spin_density = strict_uhf_method.spin_ao_density()
    total_density = strict_uhf_method.ao_density()

    assert spin_density.shape == (
        2,
        native_uhf_reference.mol.nao,
        native_uhf_reference.mol.nao,
    )
    np.testing.assert_allclose(
        total_density,
        spin_density.sum(axis=0),
        rtol=0.0,
        atol=0.0,
    )
    assert np.isfinite(strict_uhf_method.kernel())

    with pytest.raises(
        DeePHFCapabilityError,
        match="native pyscf.scf.hf.RHF reference",
    ):
        DeePHF(
            native_uhf_reference,
            None,
            projector_basis=PROJECTOR_BASIS,
        )


@pytest.mark.parametrize(
    "unsupported_factory",
    [
        pytest.param(scf.ROHF, id="rohf"),
        pytest.param(dft.UKS, id="uks"),
        pytest.param(lambda molecule: scf.UHF(molecule).density_fit(), id="df-uhf"),
    ],
)
def test_uhf_method_rejects_nonexact_or_decorated_reference_classes(
    unsupported_factory,
):
    reference = unsupported_factory(_make_molecule())

    with pytest.raises(
        DeePHFCapabilityError,
        match="native pyscf.scf.uhf.UHF reference",
    ):
        UHFDeePHF(reference, None, projector_basis=PROJECTOR_BASIS)


def test_uhf_method_rejects_rhf_and_uhf_subclasses():
    rhf_molecule = gto.M(
        atom="H 0 0 0; H 0 0 1.4",
        basis="sto-3g",
        unit="Bohr",
        spin=0,
        verbose=0,
    )
    rhf_reference = scf.RHF(rhf_molecule).run(conv_tol=1.0e-13)

    class DerivedUHF(scf.uhf.UHF):
        pass

    with pytest.raises(DeePHFCapabilityError, match="native pyscf.scf.uhf.UHF"):
        UHFDeePHF(rhf_reference, None, projector_basis=PROJECTOR_BASIS)
    with pytest.raises(DeePHFCapabilityError, match="native pyscf.scf.uhf.UHF"):
        UHFDeePHF(
            DerivedUHF(_make_molecule()),
            None,
            projector_basis=PROJECTOR_BASIS,
        )


def test_uhf_method_rejects_an_unconverged_reference(native_uhf_reference, monkeypatch):
    monkeypatch.setattr(native_uhf_reference, "converged", False)

    with pytest.raises(DeePHFCapabilityError, match="must be converged"):
        UHFDeePHF(
            native_uhf_reference,
            None,
            projector_basis=PROJECTOR_BASIS,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        pytest.param("fractional", "integer spin-orbital", id="fractional"),
        pytest.param("complex", "orbitals must be real", id="complex"),
        pytest.param("nonfinite", "orbital state must be finite", id="nonfinite"),
        pytest.param("truncated", "complete square", id="truncated"),
        pytest.param("wrong_spin_count", "alpha and beta electron counts", id="spin-count"),
        pytest.param("nonaufbau", "Aufbau ground-state root", id="root-swap"),
    ],
)
def test_uhf_reference_rejects_invalid_or_root_changed_orbital_state(
    native_uhf_reference,
    monkeypatch,
    mutation,
    message,
):
    if mutation == "fractional":
        occupations = np.asarray(native_uhf_reference.mo_occ).copy()
        occupations[0, 0] = 0.5
        monkeypatch.setattr(native_uhf_reference, "mo_occ", occupations)
    elif mutation == "complex":
        coefficients = np.asarray(native_uhf_reference.mo_coeff).astype(np.complex128)
        monkeypatch.setattr(native_uhf_reference, "mo_coeff", coefficients)
    elif mutation == "nonfinite":
        energies = np.asarray(native_uhf_reference.mo_energy).copy()
        energies[0, 0] = np.nan
        monkeypatch.setattr(native_uhf_reference, "mo_energy", energies)
    elif mutation == "truncated":
        coefficients = np.asarray(native_uhf_reference.mo_coeff)[:, :, :-1].copy()
        monkeypatch.setattr(native_uhf_reference, "mo_coeff", coefficients)
    elif mutation == "wrong_spin_count":
        occupations = np.asarray(native_uhf_reference.mo_occ).copy()
        occupations[0, np.flatnonzero(occupations[0] > 0)[-1]] = 0.0
        monkeypatch.setattr(native_uhf_reference, "mo_occ", occupations)
    else:
        occupations = np.asarray(native_uhf_reference.mo_occ).copy()
        last_occupied = np.flatnonzero(occupations[0] > 0)[-1]
        first_virtual = np.flatnonzero(occupations[0] == 0)[0]
        occupations[0, last_occupied] = 0.0
        occupations[0, first_virtual] = 1.0
        monkeypatch.setattr(native_uhf_reference, "mo_occ", occupations)

    with pytest.raises(DeePHFCapabilityError, match=message):
        validate_uhf_reference(native_uhf_reference)


def test_uhf_reference_rejects_tampered_canonical_energies(
    native_uhf_reference,
    monkeypatch,
):
    energies = np.asarray(native_uhf_reference.mo_energy).copy()
    energies[1, 0] += 1.0e-3
    monkeypatch.setattr(native_uhf_reference, "mo_energy", energies)

    with pytest.raises(
        DeePHFCapabilityError,
        match="canonical SCF equations",
    ):
        validate_uhf_reference(native_uhf_reference)


def test_uhf_reference_rejects_external_instance_hooks(
    native_uhf_reference,
    monkeypatch,
):
    monkeypatch.setitem(
        native_uhf_reference.__dict__,
        "get_hcore",
        lambda *args, **kwargs: np.eye(native_uhf_reference.mol.nao),
    )

    with pytest.raises(DeePHFCapabilityError, match="instance hooks: get_hcore"):
        validate_uhf_reference(native_uhf_reference)


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [
        pytest.param("symmetry", "C1", "symmetry-constrained", id="symmetry"),
        pytest.param("cart", True, "spherical AO", id="cartesian"),
        pytest.param("_ecp", {"N": object()}, "all-electron", id="ecp"),
        pytest.param("_pseudo", {"N": object()}, "pseudopotentials", id="pseudo"),
        pytest.param("omega", 0.2, "full Coulomb", id="range-separated"),
        pytest.param("nucmod", {0: 2}, "point nuclei", id="finite-nucleus"),
    ],
)
def test_uhf_reference_rejects_unsupported_molecular_physics(
    native_uhf_reference,
    monkeypatch,
    attribute,
    value,
    message,
):
    monkeypatch.setattr(native_uhf_reference.mol, attribute, value, raising=False)

    with pytest.raises(DeePHFCapabilityError, match=message):
        validate_uhf_reference(native_uhf_reference)


def test_uhf_reference_rejects_ghost_centers_before_orbital_state_checks():
    molecule = gto.M(
        atom="N 0 0 0; ghost-H 0 0 1",
        basis="sto-3g",
        spin=1,
        verbose=0,
    )
    reference = scf.UHF(molecule)
    reference.converged = True

    with pytest.raises(DeePHFCapabilityError, match="real atoms; ghost indices"):
        validate_uhf_reference(reference)


def test_uhf_reference_rejects_non_native_two_electron_state(
    native_uhf_reference,
    monkeypatch,
):
    nao = native_uhf_reference.mol.nao
    monkeypatch.setattr(
        native_uhf_reference,
        "_eri",
        np.zeros((nao, nao, nao, nao), dtype=np.float64),
    )

    with pytest.raises(
        DeePHFCapabilityError,
        match="two-electron interaction does not match",
    ):
        validate_uhf_reference(native_uhf_reference)


def test_uhf_response_is_spin_resolved_immutable_and_fully_reconstructed(
    native_uhf_reference,
    strict_uhf_response,
):
    response = strict_uhf_response
    occupations = np.asarray(native_uhf_reference.mo_occ)
    nalpha = int(np.count_nonzero(occupations[0]))
    nbeta = int(np.count_nonzero(occupations[1]))
    natm = native_uhf_reference.mol.natm
    nao = native_uhf_reference.mol.nao

    assert type(response) is UHFResponse
    assert type(response.diagnostics) is UHFResponseDiagnostics
    assert response.alpha_mo_response.shape == (natm, 3, nao, nalpha)
    assert response.beta_mo_response.shape == (natm, 3, nao, nbeta)
    assert response.total_density_response.shape == (natm, 3, nao, nao)
    for response_field in fields(response):
        value = getattr(response, response_field.name)
        if isinstance(value, np.ndarray):
            assert value.dtype == np.dtype(np.float64)
            assert np.isfinite(value).all()
            assert not value.flags.writeable

    np.testing.assert_allclose(
        response.total_density_response,
        response.alpha_density_response + response.beta_density_response,
        rtol=0.0,
        atol=2.0e-15,
    )
    np.testing.assert_allclose(
        response.total_density_response,
        response.total_density_response_metric
        + response.total_density_response_occupied_virtual,
        rtol=0.0,
        atol=2.0e-15,
    )
    assert np.linalg.norm(response.alpha_density_response) > 0.1
    assert np.linalg.norm(response.beta_density_response) > 0.1
    assert np.linalg.norm(response.total_density_response_metric) > 0.1
    assert np.linalg.norm(response.total_density_response_occupied_virtual) > 0.1


def test_uhf_response_reports_independent_coupled_diagnostics(
    strict_uhf_response,
):
    diagnostics = strict_uhf_response.diagnostics

    assert diagnostics.response_dimension == (
        diagnostics.alpha_response_dimension
        + diagnostics.beta_response_dimension
    )
    assert diagnostics.minimum_alpha_orbital_gap > 0.1
    assert diagnostics.minimum_beta_orbital_gap > 0.1
    assert diagnostics.operator_minimum_eigenvalue > (
        diagnostics.operator_stability_tolerance
    )
    assert diagnostics.operator_condition_number < (
        diagnostics.operator_condition_tolerance
    )
    assert diagnostics.operator_symmetry_residual < 1.0e-12
    assert diagnostics.maximum_residual <= diagnostics.residual_tolerance
    assert diagnostics.alpha_maximum_residual <= diagnostics.residual_tolerance
    assert diagnostics.beta_maximum_residual <= diagnostics.residual_tolerance
    assert diagnostics.density_reconstruction_residual < 1.0e-12
    assert diagnostics.alpha_metric_residual < 1.0e-12
    assert diagnostics.beta_metric_residual < 1.0e-12
    assert diagnostics.alpha_idempotency_residual < 1.0e-12
    assert diagnostics.beta_idempotency_residual < 1.0e-12
    assert diagnostics.alpha_particle_number_residual < 1.0e-12
    assert diagnostics.beta_particle_number_residual < 1.0e-12
    assert diagnostics.alpha_translation_residual < 1.0e-10
    assert diagnostics.beta_translation_residual < 1.0e-10
    assert diagnostics.translation_residual < 1.0e-10


def test_uhf_method_accessors_preserve_every_spin_and_derivative_partition(
    strict_uhf_method,
    strict_uhf_response,
):
    density_spin = strict_uhf_method.first_order_spin_density(
        response=strict_uhf_response
    )
    density_total = strict_uhf_method.first_order_density(
        response=strict_uhf_response
    )
    explicit_spin = strict_uhf_method.dq_dR_explicit_spin()
    response_spin = strict_uhf_method.dq_dR_response_spin(
        response=strict_uhf_response
    )
    relaxed_spin = strict_uhf_method.dq_dR_relaxed_spin(
        response=strict_uhf_response
    )

    np.testing.assert_allclose(
        density_total,
        density_spin.sum(axis=0),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        explicit_spin.sum(axis=0),
        strict_uhf_method.dq_dR_explicit(),
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        relaxed_spin,
        explicit_spin + response_spin,
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        relaxed_spin.sum(axis=0),
        strict_uhf_method.dq_dR_relaxed(response=strict_uhf_response),
        rtol=0.0,
        atol=1.0e-12,
    )


@pytest.mark.parametrize(
    "response_options",
    [
        pytest.param({"cphf_tolerance": np.nan}, id="nan-cphf"),
        pytest.param({"residual_tolerance": np.inf}, id="inf-residual"),
        pytest.param({"invariant_tolerance": True}, id="boolean-invariant"),
        pytest.param({"orbital_gap_tolerance": "1e-7"}, id="string-gap"),
        pytest.param({"level_shift": np.inf}, id="inf-shift"),
        pytest.param({"operator_stability_tolerance": 0.0}, id="zero-stability"),
        pytest.param({"operator_condition_tolerance": 1.0}, id="unit-condition"),
        pytest.param({"operator_symmetry_tolerance": 0.0}, id="zero-symmetry"),
        pytest.param({"max_cycle": 1.5}, id="fractional-cycle"),
        pytest.param({"max_refinement_cycles": True}, id="boolean-refinement"),
        pytest.param({"operator_dimension_limit": 10.5}, id="fractional-limit"),
    ],
)
def test_uhf_response_rejects_invalid_controls(
    native_uhf_reference,
    response_options,
):
    with pytest.raises((TypeError, ValueError)):
        UHFResponseAdapter(native_uhf_reference, **response_options)


def test_uhf_response_operator_stability_condition_and_dimension_gates_are_independent(
    native_uhf_reference,
    strict_uhf_response,
):
    diagnostics = strict_uhf_response.diagnostics

    with pytest.raises(DeePHFCapabilityError, match="unstable or singular"):
        UHFResponseAdapter(
            native_uhf_reference,
            operator_stability_tolerance=(
                diagnostics.operator_minimum_eigenvalue * 1.01
            ),
        ).solve()
    with pytest.raises(DeePHFCapabilityError, match="ill conditioned"):
        UHFResponseAdapter(
            native_uhf_reference,
            operator_condition_tolerance=max(
                1.0001,
                diagnostics.operator_condition_number * 0.99,
            ),
        ).solve()
    with pytest.raises(DeePHFCapabilityError, match="dimension exceeds"):
        UHFResponseAdapter(
            native_uhf_reference,
            operator_dimension_limit=diagnostics.response_dimension - 1,
        ).solve()


def test_uhf_response_rejects_an_asymmetric_coupled_operator(
    native_uhf_reference,
    monkeypatch,
):
    original_apply = UHFResponseAdapter._apply_occupied_virtual_operator

    def asymmetric_apply(self, vectors, *args, **kwargs):
        vectors = np.asarray(vectors)
        result = np.asarray(
            original_apply(self, vectors, *args, **kwargs)
        ).copy()
        result[..., 0] += 1.0e-4 * vectors[..., 1]
        return result

    monkeypatch.setattr(
        UHFResponseAdapter,
        "_apply_occupied_virtual_operator",
        asymmetric_apply,
    )

    with pytest.raises(UHFResponseError, match="violates symmetry"):
        UHFResponseAdapter(native_uhf_reference).solve()


def test_corrupted_ucphf_solution_fails_the_independent_coupled_residual(
    native_uhf_reference,
    monkeypatch,
):
    original_solve = pyscf_uhf.ucphf.solve
    occupations = np.asarray(native_uhf_reference.mo_occ)
    first_alpha_virtual = int(np.flatnonzero(occupations[0] == 0)[0])

    def corrupted_solve(*args, **kwargs):
        response, energy_response = original_solve(*args, **kwargs)
        alpha = np.asarray(response[0]).copy()
        beta = np.asarray(response[1]).copy()
        alpha.reshape(-1, alpha.shape[-2], alpha.shape[-1])[
            :, first_alpha_virtual, 0
        ] += 1.0e-4
        return (alpha, beta), energy_response

    monkeypatch.setattr(pyscf_uhf.ucphf, "solve", corrupted_solve)

    with pytest.raises(UHFResponseError, match="residual exceeds tolerance"):
        UHFResponseAdapter(
            native_uhf_reference,
            max_refinement_cycles=0,
        ).solve()


def test_uhf_gradient_failure_clears_results_without_any_fallback(
    strict_uhf_method,
    monkeypatch,
):
    driver = strict_uhf_method.nuc_grad_method().run()
    assert driver.de_full is not None

    def failed_solve(*args, **kwargs):
        raise RuntimeError("injected UHF coupled solve failure")

    monkeypatch.setattr(pyscf_uhf.ucphf, "solve", failed_solve)

    with pytest.raises(
        UHFResponseError,
        match="UHF coupled CPHF solve failed",
    ):
        driver.kernel()

    for name in (
        "response_result",
        "dq_dR_explicit",
        "dq_dR_response",
        "dq_dR_relaxed",
        "correction_gradient",
        "de_full",
        "de",
    ):
        assert getattr(driver, name) is None


@pytest.mark.parametrize("corruption", ["mutable", "float32", "nonfinite"])
def test_supplied_uhf_response_rejects_resealed_array_forgery(
    strict_uhf_method,
    strict_uhf_response,
    corruption,
):
    value = np.array(strict_uhf_response.alpha_density_response, copy=True)
    if corruption == "float32":
        value = value.astype(np.float32)
    elif corruption == "nonfinite":
        value[0, 0, 0, 0] = np.nan
        value.setflags(write=False)
    forged = replace(strict_uhf_response, alpha_density_response=value)
    forged = _reseal(forged)

    with pytest.raises(UHFResponseError):
        strict_uhf_method.first_order_density(response=forged)


def test_supplied_uhf_response_rejects_coordinated_reconstruction_forgery(
    strict_uhf_method,
    strict_uhf_response,
):
    alpha = np.array(strict_uhf_response.alpha_density_response, copy=True)
    alpha[0, 0, 0, 0] += 1.0e-5
    alpha.setflags(write=False)
    total = np.array(strict_uhf_response.total_density_response, copy=True)
    total[0, 0, 0, 0] += 1.0e-5
    total.setflags(write=False)
    forged = _reseal(
        replace(
            strict_uhf_response,
            alpha_density_response=alpha,
            total_density_response=total,
        )
    )

    with pytest.raises(UHFResponseError, match="alpha density response is inconsistent"):
        strict_uhf_method.first_order_density(response=forged)


def test_supplied_uhf_response_rejects_foreign_identity_and_diagnostics(
    strict_uhf_method,
    strict_uhf_response,
):
    foreign = _reseal(
        replace(
            strict_uhf_response,
            reference_identity=strict_uhf_response.reference_identity + 1,
        )
    )
    forged_diagnostics = replace(
        strict_uhf_response.diagnostics,
        alpha_maximum_residual=(
            strict_uhf_response.diagnostics.alpha_maximum_residual + 1.0e-5
        ),
    )
    forged = _reseal(
        replace(strict_uhf_response, diagnostics=forged_diagnostics)
    )

    with pytest.raises(UHFResponseError, match="belongs to another reference"):
        strict_uhf_method.first_order_density(response=foreign)
    with pytest.raises(UHFResponseError, match="residual diagnostics are inconsistent"):
        strict_uhf_method.first_order_density(response=forged)


def test_response_and_response_options_are_mutually_exclusive(
    strict_uhf_method,
    strict_uhf_response,
):
    for function in (
        strict_uhf_method.first_order_density,
        strict_uhf_method.first_order_spin_density,
        strict_uhf_method.dq_dR_response,
        strict_uhf_method.dq_dR_response_spin,
        strict_uhf_method.dq_dR_relaxed,
        strict_uhf_method.dq_dR_relaxed_spin,
    ):
        with pytest.raises(ValueError, match="mutually exclusive"):
            function(response=strict_uhf_response, cphf_tolerance=1.0e-10)


def test_uhf_direct_backend_is_distinct_and_rejects_adjoint_or_scanner_paths(
    native_uhf_reference,
    strict_uhf_method,
):
    driver = strict_uhf_method.nuc_grad_method(backend="direct")
    assert type(driver) is UHFDeePHFGradients
    assert driver.backend == "direct"

    with pytest.raises(DeePHFCapabilityError, match="backend must be 'direct'"):
        strict_uhf_method.nuc_grad_method(backend="zvector")
    with pytest.raises(DeePHFCapabilityError, match="adjoint backend"):
        strict_uhf_method.adjoint()
    with pytest.raises(DeePHFCapabilityError, match="adjoint backend"):
        UHFDeePHF(
            native_uhf_reference,
            None,
            projector_basis=PROJECTOR_BASIS,
            adjoint_options={"residual_tolerance": 1.0e-9},
        )
    with pytest.raises(UHFResponseError, match="does not provide a gradient scanner"):
        driver.as_scanner()
    with pytest.raises(ValueError, match="unsupported direct backend options"):
        strict_uhf_method.nuc_grad_method(fallback="explicit")


def test_rhf_drivers_and_force_data_reject_uhf_method_or_reference(
    native_uhf_reference,
    strict_uhf_method,
):
    with pytest.raises(TypeError, match="requires an exact DeePHF method"):
        RHFDeePHFGradients(strict_uhf_method)
    with pytest.raises(TypeError, match="requires an exact DeePHF method"):
        RHFDeePHFZVectorGradients(strict_uhf_method)
    with pytest.raises(DeePHFCapabilityError, match="native pyscf.scf.hf.RHF"):
        generate_rhf_force_frame(
            native_uhf_reference,
            projector_basis=PROJECTOR_BASIS,
            e_target=np.float64(native_uhf_reference.e_tot),
            f_target=np.zeros((native_uhf_reference.mol.natm, 3)),
        )


def test_uhf_driver_rejects_an_rhf_method():
    molecule = gto.M(
        atom="H 0 0 0; H 0 0 1.4",
        basis="sto-3g",
        unit="Bohr",
        spin=0,
        verbose=0,
    )
    reference = scf.RHF(molecule).run(conv_tol=1.0e-13)
    method = DeePHF(reference, None, projector_basis=[[0, [0.8, 1.0]]])

    with pytest.raises(TypeError, match="exact UHFDeePHF method"):
        UHFDeePHFGradients(method)


def test_uhf_force_path_rejects_an_incompatible_projector_model(
    native_uhf_reference,
):
    model = _make_model(projector_basis=[[0, [0.9, 1.0]]])

    with pytest.raises(DeePHFCapabilityError, match="projector metadata"):
        UHFDeePHF(
            native_uhf_reference,
            model,
            projector_basis=PROJECTOR_BASIS,
        )


def test_uhf_force_path_rejects_a_nondifferentiable_descriptor(
    native_uhf_reference,
):
    projector_basis = [[4, [0.2, 1.0]]]
    model = _make_model(projector_basis=projector_basis, input_dim=9)
    method = UHFDeePHF(
        native_uhf_reference,
        model,
        projector_basis=projector_basis,
    )

    with pytest.raises(
        DescriptorDifferentiabilityError,
        match="structural zero block sensitivity spread",
    ):
        method.response()


def test_uhf_gradient_driver_rejects_corrupted_binding_and_invalid_atoms(
    strict_uhf_method,
):
    driver = strict_uhf_method.nuc_grad_method()

    with pytest.raises(TypeError, match="atom indices must be integers"):
        driver.kernel(atmlst=[True])
    assert driver.response_result is None
    assert driver.de_full is None

    driver._backend = "zvector"
    with pytest.raises(UHFResponseError, match="binding is invalid"):
        driver.kernel()
    assert driver.response_result is None
    assert driver.de_full is None
