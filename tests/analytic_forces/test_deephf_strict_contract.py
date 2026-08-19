from dataclasses import fields, replace

import numpy as np
import pytest
import torch
from pyscf import ao2mo, gto, scf

import deepks.deephf.pyscf_rhf as pyscf_rhf
from deepks.deephf import (
    DeePHF,
    DeePHFCapabilityError,
    RHFResponseAdapter,
    RHFResponseError,
)
from deepks.model.model import CorrNet


SMALL_PROJECTOR_BASIS = [[0, [1.0, 1.0]]]


def _small_reference():
    molecule = gto.M(
        atom="H 0 0 0; H 0 0 1.4",
        basis="sto-3g",
        unit="Bohr",
        verbose=0,
    )
    reference = scf.RHF(molecule)
    reference.conv_tol = 1.0e-13
    reference.kernel()
    assert reference.converged
    return reference


def _small_model(projector_basis=SMALL_PROJECTOR_BASIS):
    model = CorrNet(
        input_dim=1,
        hidden_sizes=(2,),
        proj_basis=projector_basis,
    ).double()
    with torch.no_grad():
        model.linear.weight.fill_(0.01)
        model.linear.bias.zero_()
        for parameter in model.densenet.parameters():
            parameter.zero_()
    return model.eval()


@pytest.mark.parametrize(
    "response_options",
    [
        pytest.param({"cphf_tolerance": np.nan}, id="nan-cphf-tolerance"),
        pytest.param({"residual_tolerance": np.inf}, id="inf-residual-tolerance"),
        pytest.param({"invariant_tolerance": np.nan}, id="nan-invariant-tolerance"),
        pytest.param({"orbital_gap_tolerance": np.inf}, id="inf-gap-tolerance"),
        pytest.param(
            {"operator_stability_tolerance": np.nan},
            id="nan-operator-stability-tolerance",
        ),
        pytest.param(
            {"operator_condition_tolerance": np.inf},
            id="inf-operator-condition-tolerance",
        ),
        pytest.param(
            {"operator_symmetry_tolerance": np.nan},
            id="nan-operator-symmetry-tolerance",
        ),
    ],
)
def test_response_rejects_nonfinite_tolerances(
    rhf_oracle_case,
    response_options,
):
    with pytest.raises(ValueError, match="response tolerances must be finite"):
        rhf_oracle_case.method.response(**response_options)


@pytest.mark.parametrize("level_shift", [np.nan, np.inf])
def test_response_rejects_nonfinite_level_shift(
    rhf_oracle_case,
    level_shift,
):
    with pytest.raises(ValueError, match="response level_shift must be finite"):
        rhf_oracle_case.method.response(level_shift=level_shift)


def test_reference_rejects_tampered_canonical_orbital_energies():
    reference = _small_reference()
    reference.mo_energy = np.asarray(reference.mo_energy).copy()
    reference.mo_energy[0] += 1.0e-3

    with pytest.raises(
        DeePHFCapabilityError,
        match="do not satisfy the canonical SCF equations",
    ):
        DeePHF(reference, None, projector_basis=SMALL_PROJECTOR_BASIS)


def test_reference_rejects_truncated_virtual_orbitals():
    reference = _small_reference()
    reference.mo_coeff = np.asarray(reference.mo_coeff)[:, :-1].copy()
    reference.mo_energy = np.asarray(reference.mo_energy)[:-1].copy()
    reference.mo_occ = np.asarray(reference.mo_occ)[:-1].copy()

    with pytest.raises(
        DeePHFCapabilityError,
        match="requires a complete square MO coefficient matrix",
    ):
        DeePHF(reference, None, projector_basis=SMALL_PROJECTOR_BASIS)


def test_reference_rejects_non_float64_orbital_state():
    reference = _small_reference()
    reference.mo_coeff = np.asarray(reference.mo_coeff, dtype=np.float32)

    with pytest.raises(
        DeePHFCapabilityError,
        match="orbital state must use numpy.float64",
    ):
        DeePHF(reference, None, projector_basis=SMALL_PROJECTOR_BASIS)


def test_reference_rejects_object_orbital_state_with_capability_error():
    reference = _small_reference()
    reference.mo_coeff = np.asarray(reference.mo_coeff, dtype=object)

    with pytest.raises(
        DeePHFCapabilityError,
        match="orbital state must use numpy.float64",
    ):
        DeePHF(reference, None, projector_basis=SMALL_PROJECTOR_BASIS)


def test_reference_rejects_a_nonphysical_two_electron_integral_cache():
    molecule = gto.M(
        atom="H 0 0 0; H 0 0 1.4",
        basis="sto-3g",
        unit="Bohr",
        verbose=0,
    )
    reference = scf.RHF(molecule)
    reference._eri = ao2mo.restore(
        8,
        0.5 * molecule.intor("int2e"),
        molecule.nao,
    )
    reference.conv_tol = 1.0e-13
    reference.kernel()
    assert reference.converged

    with pytest.raises(
        DeePHFCapabilityError,
        match="two-electron interaction does not match the native molecular integrals",
    ):
        DeePHF(reference, None, projector_basis=SMALL_PROJECTOR_BASIS)


@pytest.mark.parametrize(
    "hook_name",
    ["get_occ", "get_veff", "make_rdm1", "nuc_grad_method"],
)
def test_reference_rejects_instance_response_hooks(hook_name):
    reference = _small_reference()
    setattr(reference, hook_name, getattr(reference, hook_name))

    with pytest.raises(
        DeePHFCapabilityError,
        match=rf"unsupported instance hooks: {hook_name}",
    ):
        DeePHF(reference, None, projector_basis=SMALL_PROJECTOR_BASIS)


@pytest.mark.parametrize(
    "hook_name",
    ["intor", "aoslice_by_atom", "atom_coords", "atom_charges", "energy_nuc"],
)
def test_reference_rejects_molecule_instance_hooks(hook_name):
    reference = _small_reference()
    setattr(reference.mol, hook_name, getattr(reference.mol, hook_name))

    with pytest.raises(
        DeePHFCapabilityError,
        match=rf"unsupported instance hooks: {hook_name}",
    ):
        DeePHF(reference, None, projector_basis=SMALL_PROJECTOR_BASIS)


def test_response_rejects_an_unsupported_pyscf_series(
    rhf_oracle_case,
    monkeypatch,
):
    monkeypatch.setattr(pyscf_rhf.pyscf, "__version__", "2.15.0")

    with pytest.raises(
        DeePHFCapabilityError,
        match="adapter supports PySCF 2.14",
    ):
        RHFResponseAdapter(rhf_oracle_case.reference)


def test_response_rejects_an_insufficient_occupied_virtual_gap(
    rhf_oracle_case,
):
    with pytest.raises(
        DeePHFCapabilityError,
        match="occupied-virtual gap is outside the strict response domain",
    ):
        rhf_oracle_case.method.response(orbital_gap_tolerance=2.0)


def test_response_reports_a_stable_well_conditioned_coupled_operator(
    rhf_oracle_case,
):
    diagnostics = rhf_oracle_case.response.diagnostics

    assert diagnostics.response_dimension == 10
    assert diagnostics.operator_minimum_eigenvalue > 0.4
    assert diagnostics.operator_maximum_eigenvalue > 10.0
    assert diagnostics.operator_condition_number < 50.0
    assert diagnostics.operator_symmetry_residual < 1.0e-12


def test_response_rejects_an_excessive_coupled_operator_condition_number(
    rhf_oracle_case,
):
    with pytest.raises(
        DeePHFCapabilityError,
        match="response operator is ill conditioned",
    ):
        rhf_oracle_case.method.response(operator_condition_tolerance=40.0)


def test_response_rejects_a_singular_coupled_operator_without_fallback():
    molecule = gto.M(
        atom="C 0 0 0; C 0 0 3.0",
        basis="sto-3g",
        unit="Bohr",
        symmetry=False,
        cart=False,
        verbose=0,
    )
    reference = scf.RHF(molecule)
    reference.conv_tol = 1.0e-13
    reference.conv_tol_grad = 1.0e-10
    reference.kernel()
    assert reference.converged
    occupied = reference.mo_occ > 0
    virtual = reference.mo_occ == 0
    assert np.min(
        reference.mo_energy[virtual, None] - reference.mo_energy[occupied]
    ) > 0.3
    method = DeePHF(
        reference,
        None,
        projector_basis=SMALL_PROJECTOR_BASIS,
    )
    gradient_driver = method.nuc_grad_method()

    with pytest.raises(
        DeePHFCapabilityError,
        match="response operator is unstable or singular",
    ):
        gradient_driver.kernel()

    assert gradient_driver.response_result is None
    assert gradient_driver.dq_dR_relaxed is None
    assert gradient_driver.de_full is None
    assert gradient_driver.de is None


@pytest.mark.parametrize(
    ("option_name", "option_value"),
    [
        pytest.param("max_cycle", 1.5, id="noninteger-max-cycle"),
        pytest.param(
            "max_refinement_cycles",
            1.5,
            id="noninteger-refinement-cycles",
        ),
        pytest.param(
            "operator_dimension_limit",
            10.5,
            id="noninteger-operator-dimension-limit",
        ),
    ],
)
def test_response_rejects_noninteger_limits(
    rhf_oracle_case,
    option_name,
    option_value,
):
    with pytest.raises(ValueError, match="must be an integer"):
        rhf_oracle_case.method.response(**{option_name: option_value})


def test_response_rejects_a_condition_audit_outside_its_dimension_limit(
    rhf_oracle_case,
):
    with pytest.raises(
        DeePHFCapabilityError,
        match="response dimension exceeds the explicit condition-audit limit",
    ):
        rhf_oracle_case.method.response(operator_dimension_limit=9)


@pytest.mark.parametrize(
    "response_options",
    [
        pytest.param(
            {"operator_stability_tolerance": 0.0},
            id="zero-operator-stability-tolerance",
        ),
        pytest.param(
            {"operator_condition_tolerance": 1.0},
            id="unit-operator-condition-tolerance",
        ),
        pytest.param(
            {"operator_symmetry_tolerance": 0.0},
            id="zero-operator-symmetry-tolerance",
        ),
        pytest.param(
            {"operator_dimension_limit": True},
            id="boolean-operator-dimension-limit",
        ),
        pytest.param(
            {"operator_dimension_limit": 0},
            id="zero-operator-dimension-limit",
        ),
    ],
)
def test_response_rejects_invalid_operator_audit_controls(
    rhf_oracle_case,
    response_options,
):
    with pytest.raises(ValueError):
        rhf_oracle_case.method.response(**response_options)


def test_response_arrays_are_immutable(rhf_oracle_case):
    response = rhf_oracle_case.response
    response_arrays = [
        getattr(response, field.name)
        for field in fields(response)
        if isinstance(getattr(response, field.name), np.ndarray)
    ]

    assert response_arrays
    assert all(not array.flags.writeable for array in response_arrays)
    with pytest.raises(ValueError):
        response.density_response[0, 0, 0, 0] = 0.0


def test_mutable_response_array_is_rejected(rhf_oracle_case):
    mutable_response = replace(
        rhf_oracle_case.response,
        density_response=rhf_oracle_case.response.density_response.copy(),
    )

    with pytest.raises(
        RHFResponseError,
        match="density_response must be immutable",
    ):
        rhf_oracle_case.method.first_order_density(response=mutable_response)


def test_non_float64_response_array_raises_typed_response_error(
    rhf_oracle_case,
):
    invalid_density = np.full(
        rhf_oracle_case.response.density_response.shape,
        "0",
        dtype="U1",
    )
    invalid_density.flags.writeable = False
    forged = replace(
        rhf_oracle_case.response,
        density_response=invalid_density,
        integrity_fingerprint="",
    )
    forged = replace(
        forged,
        integrity_fingerprint=pyscf_rhf.response_integrity_fingerprint(forged),
    )

    with pytest.raises(
        RHFResponseError,
        match="density_response must use numpy.float64",
    ):
        rhf_oracle_case.method.first_order_density(response=forged)


@pytest.mark.parametrize(
    "consumer_name",
    ["first_order_density", "dq_dR_response", "dq_dR_relaxed"],
)
def test_supplied_response_and_response_options_are_mutually_exclusive(
    rhf_oracle_case,
    consumer_name,
):
    consumer = getattr(rhf_oracle_case.method, consumer_name)

    with pytest.raises(
        ValueError,
        match="response and response_options are mutually exclusive",
    ):
        consumer(
            response=rhf_oracle_case.response,
            residual_tolerance=1.0e-8,
        )


def test_forged_zero_density_response_fails_integrity_and_cross_level_checks(
    rhf_oracle_case,
):
    response = rhf_oracle_case.response

    def immutable_zeros(value):
        result = np.zeros_like(value)
        result.flags.writeable = False
        return result

    forged = replace(
        response,
        density_response=immutable_zeros(response.density_response),
        density_response_occupied_virtual=immutable_zeros(
            response.density_response_occupied_virtual
        ),
        density_response_metric=immutable_zeros(
            response.density_response_metric
        ),
    )
    with pytest.raises(
        RHFResponseError,
        match="failed its integrity check",
    ):
        rhf_oracle_case.method.first_order_density(response=forged)

    resealed_forgery = replace(
        forged,
        integrity_fingerprint=pyscf_rhf.response_integrity_fingerprint(forged),
    )
    with pytest.raises(
        RHFResponseError,
        match="complete density response is inconsistent",
    ):
        rhf_oracle_case.method.first_order_density(response=resealed_forgery)


def test_coordinated_zero_response_fails_independent_equation_audit(
    rhf_oracle_case,
):
    response = rhf_oracle_case.response

    def immutable_zeros(value):
        result = np.zeros_like(value)
        result.flags.writeable = False
        return result

    zero_fields = {
        name: immutable_zeros(getattr(response, name))
        for name in (
            "mo_response",
            "mo_response_occupied_virtual",
            "mo_response_metric",
            "coefficient_response",
            "coefficient_response_occupied_virtual",
            "coefficient_response_metric",
            "density_response",
            "density_response_occupied_virtual",
            "density_response_metric",
            "orbital_response_residual",
        )
    }
    diagnostics = replace(
        response.diagnostics,
        maximum_residual=0.0,
        residual_rms=0.0,
        metric_residual=0.0,
        idempotency_residual=0.0,
        particle_number_residual=0.0,
        refinement_cycles=0,
        residual_history=(0.0,),
    )
    forged = replace(
        response,
        **zero_fields,
        diagnostics=diagnostics,
        integrity_fingerprint="",
    )
    forged = replace(
        forged,
        integrity_fingerprint=pyscf_rhf.response_integrity_fingerprint(forged),
    )

    with pytest.raises(
        RHFResponseError,
        match="orbital residual is not independently reproducible",
    ):
        rhf_oracle_case.method.first_order_density(response=forged)


def test_same_trusted_response_cannot_be_resealed_after_coordinated_mutation():
    method = DeePHF(
        _small_reference(),
        None,
        projector_basis=SMALL_PROJECTOR_BASIS,
    )
    response = method.response()

    def immutable_zeros(value):
        result = np.zeros_like(value)
        result.flags.writeable = False
        return result

    for name in (
        "mo_response",
        "mo_response_occupied_virtual",
        "mo_response_metric",
        "coefficient_response",
        "coefficient_response_occupied_virtual",
        "coefficient_response_metric",
        "density_response",
        "density_response_occupied_virtual",
        "density_response_metric",
        "orbital_response_residual",
    ):
        object.__setattr__(response, name, immutable_zeros(getattr(response, name)))
    object.__setattr__(
        response,
        "diagnostics",
        replace(
            response.diagnostics,
            maximum_residual=0.0,
            residual_rms=0.0,
            metric_residual=0.0,
            idempotency_residual=0.0,
            particle_number_residual=0.0,
            refinement_cycles=0,
            residual_history=(0.0,),
        ),
    )
    object.__setattr__(
        response,
        "integrity_fingerprint",
        pyscf_rhf.response_integrity_fingerprint(response),
    )

    with pytest.raises(
        RHFResponseError,
        match="metric_residual diagnostic is inconsistent",
    ):
        method.first_order_density(response=response)


def test_relabelled_response_partitions_fail_subspace_audit(rhf_oracle_case):
    response = rhf_oracle_case.response

    def immutable_copy(value):
        result = np.array(value, copy=True)
        result.flags.writeable = False
        return result

    def immutable_zeros(value):
        result = np.zeros_like(value)
        result.flags.writeable = False
        return result

    forged = replace(
        response,
        mo_response_occupied_virtual=immutable_zeros(
            response.mo_response_occupied_virtual
        ),
        coefficient_response_occupied_virtual=immutable_zeros(
            response.coefficient_response_occupied_virtual
        ),
        density_response_occupied_virtual=immutable_zeros(
            response.density_response_occupied_virtual
        ),
        mo_response_metric=immutable_copy(response.mo_response),
        coefficient_response_metric=immutable_copy(
            response.coefficient_response
        ),
        density_response_metric=immutable_copy(response.density_response),
        integrity_fingerprint="",
    )
    forged = replace(
        forged,
        integrity_fingerprint=pyscf_rhf.response_integrity_fingerprint(forged),
    )

    with pytest.raises(
        RHFResponseError,
        match="metric response has virtual-space support",
    ):
        rhf_oracle_case.method.first_order_density(response=forged)


def test_foreign_response_is_rejected(rhf_oracle_case):
    foreign_method = DeePHF(
        _small_reference(),
        None,
        projector_basis=SMALL_PROJECTOR_BASIS,
    )

    with pytest.raises(
        RHFResponseError,
        match="belongs to another reference",
    ):
        foreign_method.first_order_density(response=rhf_oracle_case.response)


def test_stale_response_is_rejected_after_reference_state_change():
    reference = _small_reference()
    method = DeePHF(
        reference,
        None,
        projector_basis=SMALL_PROJECTOR_BASIS,
    )
    response = method.response()
    reference.mo_coeff = np.asarray(reference.mo_coeff).copy()
    reference.mo_coeff[:, 0] *= -1.0

    with pytest.raises(
        RHFResponseError,
        match="does not match the current RHF state",
    ):
        method.first_order_density(response=response)


def test_float32_model_is_rejected():
    model = _small_model().float()

    with pytest.raises(
        DeePHFCapabilityError,
        match="correction model must use torch.float64",
    ):
        DeePHF(
            _small_reference(),
            model,
            projector_basis=SMALL_PROJECTOR_BASIS,
        )


def test_multi_output_model_is_rejected():
    class MultiOutputModel(torch.nn.Module):
        input_dim = 1

        def forward(self, values):
            total = values.sum()
            return torch.stack((total, total))

    with pytest.raises(
        DeePHFCapabilityError,
        match="must produce exactly one scalar energy",
    ):
        DeePHF(
            _small_reference(),
            MultiOutputModel(),
            projector_basis=SMALL_PROJECTOR_BASIS,
        )


def test_model_projector_metadata_mismatch_is_rejected():
    model = _small_model(projector_basis=[[0, [2.0, 1.0]]])

    with pytest.raises(
        DeePHFCapabilityError,
        match="projector metadata does not match projector_basis",
    ):
        DeePHF(
            _small_reference(),
            model,
            projector_basis=SMALL_PROJECTOR_BASIS,
        )


@pytest.mark.parametrize("nonfinite", [np.nan, np.inf])
def test_nonfinite_model_parameter_is_rejected(nonfinite):
    model = _small_model()
    with torch.no_grad():
        model.linear.weight.fill_(nonfinite)

    with pytest.raises(
        DeePHFCapabilityError,
        match="parameters and buffers must be finite",
    ):
        DeePHF(
            _small_reference(),
            model,
            projector_basis=SMALL_PROJECTOR_BASIS,
        )


def test_nonfinite_model_output_is_rejected():
    class NonfiniteOutputModel(torch.nn.Module):
        input_dim = 1

        def forward(self, values):
            return values.sum() * torch.tensor(float("nan"))

    with pytest.raises(
        DeePHFCapabilityError,
        match="correction model output must be finite",
    ):
        DeePHF(
            _small_reference(),
            NonfiniteOutputModel(),
            projector_basis=SMALL_PROJECTOR_BASIS,
        )
