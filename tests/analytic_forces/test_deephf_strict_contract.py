from dataclasses import fields

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


def test_response_rejects_an_excessive_coupled_operator_condition_number(
    rhf_oracle_case,
):
    with pytest.raises(
        DeePHFCapabilityError,
        match="response operator is ill conditioned",
    ):
        RHFResponseAdapter(
            rhf_oracle_case.reference,
            operator_condition_tolerance=40.0,
        ).validate_response_operator_exact()


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
    with pytest.raises(
        DeePHFCapabilityError,
        match="response operator is unstable or singular",
    ):
        RHFResponseAdapter(reference).validate_response_operator_exact()

@pytest.mark.parametrize(
    ("option_name", "option_value"),
    [
        pytest.param("max_cycle", 1.5, id="noninteger-max-cycle"),
        pytest.param(
            "max_refinement_cycles",
            1.5,
            id="noninteger-refinement-cycles",
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
    stored_names = {field.name for field in fields(response)}
    response_arrays = [
        getattr(response, field.name)
        for field in fields(response)
        if isinstance(getattr(response, field.name), np.ndarray)
    ]

    assert response_arrays
    assert all(not array.flags.writeable for array in response_arrays)
    retained_bytes = sum(array.nbytes for array in response_arrays)
    derived_arrays = (
        response.mo_response_metric,
        response.mo_response_occupied_virtual,
        response.coefficient_response,
        response.coefficient_response_metric,
        response.coefficient_response_occupied_virtual,
        response.density_response,
        response.density_response_metric,
        response.density_response_occupied_virtual,
    )
    legacy_bytes = retained_bytes + sum(array.nbytes for array in derived_arrays)
    assert retained_bytes < legacy_bytes
    assert not {
        "mo_response_metric",
        "mo_response_occupied_virtual",
        "coefficient_response",
        "density_response",
    }.intersection(stored_names)
    assert all(
        not np.shares_memory(response.mo_response, array)
        for array in derived_arrays
    )
    with pytest.raises(ValueError):
        response.density_response[0, 0, 0, 0] = 0.0






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










def test_foreign_response_is_rejected(rhf_oracle_case):
    foreign_method = DeePHF(
        _small_reference(),
        None,
        projector_basis=SMALL_PROJECTOR_BASIS,
    )

    with pytest.raises(
        RHFResponseError,
        match="was not produced by this method",
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
        DeePHFCapabilityError,
        match="scientific state changed",
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


def test_multi_output_model_is_rejected_during_first_calculation():
    class MultiOutputModel(torch.nn.Module):
        input_dim = 1

        def forward(self, values):
            total = values.sum()
            return torch.stack((total, total))

    method = DeePHF(
        _small_reference(),
        MultiOutputModel(),
        projector_basis=SMALL_PROJECTOR_BASIS,
    )

    with pytest.raises(
        DeePHFCapabilityError,
        match="must produce exactly one scalar energy",
    ):
        method.kernel()
    assert (method.e_base, method.e_corr, method.e_tot) == (None, None, None)


@pytest.mark.parametrize(
    ("output_kind", "message"),
    (
        ("rank", "output must have rank zero or one"),
        ("dtype", "output must use torch.float64"),
        ("complex", "output must be real"),
    ),
)
def test_invalid_model_output_contract_fails_during_first_calculation(
    output_kind,
    message,
):
    class InvalidOutputModel(torch.nn.Module):
        input_dim = 1

        def forward(self, values):
            output = values.sum()
            if output_kind == "rank":
                return output.reshape(1, 1)
            if output_kind == "dtype":
                return output.float()
            return output.to(torch.complex128)

    method = DeePHF(
        _small_reference(),
        InvalidOutputModel(),
        projector_basis=SMALL_PROJECTOR_BASIS,
    )
    assert (method.e_base, method.e_corr, method.e_tot) == (None, None, None)

    with pytest.raises(DeePHFCapabilityError, match=message):
        method.kernel()
    assert (method.e_base, method.e_corr, method.e_tot) == (None, None, None)


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


def test_nonfinite_model_output_is_rejected_during_first_calculation():
    class NonfiniteOutputModel(torch.nn.Module):
        input_dim = 1

        def forward(self, values):
            return values.sum() * torch.tensor(float("nan"))

    method = DeePHF(
        _small_reference(),
        NonfiniteOutputModel(),
        projector_basis=SMALL_PROJECTOR_BASIS,
    )

    with pytest.raises(
        DeePHFCapabilityError,
        match="correction model output must be finite",
    ):
        method.kernel()
    assert (method.e_base, method.e_corr, method.e_tot) == (None, None, None)


def test_failed_energy_recalculation_clears_all_published_energy_fields():
    method = DeePHF(
        _small_reference(),
        _small_model(),
        projector_basis=SMALL_PROJECTOR_BASIS,
    )
    assert np.isfinite(method.kernel())
    assert all(value is not None for value in (method.e_base, method.e_corr, method.e_tot))

    with torch.no_grad():
        method.model.linear.weight.fill_(float("nan"))

    with pytest.raises(
        DeePHFCapabilityError,
        match="parameters and buffers must be finite",
    ):
        method.kernel()
    assert (method.e_base, method.e_corr, method.e_tot) == (None, None, None)


def test_force_gradient_rejects_detached_descriptor_dependence():
    class DetachedModel(torch.nn.Module):
        input_dim = 1

        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float64))

        def forward(self, values):
            return self.scale * values.detach().sum() + 0.0 * values.sum()

    method = DeePHF(
        _small_reference(),
        DetachedModel(),
        projector_basis=SMALL_PROJECTOR_BASIS,
    )
    driver = method.nuc_grad_method(retain_details=False)

    with pytest.raises(
        DeePHFCapabilityError,
        match="force derivatives require an exact .*CorrNet",
    ):
        driver.kernel()
    assert (method.e_base, method.e_corr, method.e_tot) == (None, None, None)
    assert driver.de is None
    assert driver.descriptor_diagnostics is None
