from dataclasses import fields, replace

import numpy as np
import pytest
import torch
from pyscf import dft, gto, scf
from pyscf.dft import gen_grid, libxc, radi

import deepks.deephf.pyscf_rks as pyscf_rks
import deepks.deephf.rks_gradient as rks_gradient
from deepks.deephf import (
    DeePHFCapabilityError,
    RHFDeePHFGradients,
    RHFDeePHFZVectorGradients,
    RKSDeePHF,
    RKSDeePHFGradients,
    RKSDeePHFZVectorGradients,
    RKSNativeGradient,
    RKSResponse,
    RKSResponseAdapter,
    RKSResponseDiagnostics,
    RKSResponseError,
    ScalarAdjointProblem,
    generate_rhf_force_frame,
    validate_rks_reference,
)

from deepks.descriptor import DescriptorDifferentiabilityError
from deepks.model.model import CorrNet

from conftest import ORACLE_PROJECTOR_BASIS


def _reseal(response):
    return replace(
        response,
        integrity_fingerprint=pyscf_rks.rks_response_integrity_fingerprint(
            response
        ),
    )


def _run_cross_molecule_rks(atom):
    molecule = gto.M(
        atom=atom,
        basis="sto-3g",
        unit="Bohr",
        charge=0,
        spin=0,
        symmetry=False,
        cart=False,
        verbose=0,
    )
    reference = dft.RKS(molecule)
    reference.xc = "LDA_X + LDA_C_VWN"
    reference.conv_tol = 1.0e-13
    reference.conv_tol_grad = 1.0e-10
    reference.conv_tol_cpscf = 1.0e-12
    reference.max_cycle = 100
    reference.small_rho_cutoff = 0.0
    symbols = {
        molecule.atom_symbol(atom_index)
        for atom_index in range(molecule.natm)
    }
    reference.grids.atom_grid = {
        symbol: (20, 50) for symbol in symbols
    }
    reference.grids.prune = None
    reference.grids.alignment = 1
    reference.grids.symmetry = False
    reference.grids.build(with_non0tab=True, sort_grids=False)
    reference.kernel()
    assert reference.converged
    return reference


def test_rks_public_method_is_spin_summed_rank_two_and_energy_only(
    rks_oracle_case,
    monkeypatch,
):
    reference = rks_oracle_case.reference
    method = rks_oracle_case.method

    assert validate_rks_reference(reference) is reference
    assert type(method) is RKSDeePHF
    assert method.ao_density().shape == (reference.mol.nao, reference.mol.nao)

    def forbidden_response(*args, **kwargs):
        raise AssertionError("energy-only evaluation entered the response backend")

    monkeypatch.setattr(RKSResponseAdapter, "solve", forbidden_response)
    energy = method.kernel()

    assert np.isfinite(energy)
    assert energy == method.e_base + method.e_corr


def test_rks_reference_requires_the_exact_native_class(rks_oracle_case):
    molecule = rks_oracle_case.reference.mol

    class DerivedRKS(dft.rks.RKS):
        pass

    unsupported = (
        scf.RHF(molecule),
        scf.UHF(molecule),
        scf.ROHF(molecule),
        dft.UKS(molecule),
        DerivedRKS(molecule),
    )
    for reference in unsupported:
        with pytest.raises(
            DeePHFCapabilityError,
            match="native pyscf.dft.rks.RKS reference",
        ):
            RKSDeePHF(
                reference,
                None,
                projector_basis=ORACLE_PROJECTOR_BASIS,
            )


def test_rks_reference_rejects_an_unconverged_state(
    rks_oracle_case,
    monkeypatch,
):
    reference = rks_oracle_case.reference
    monkeypatch.setattr(reference, "converged", False)

    with pytest.raises(DeePHFCapabilityError, match="must be converged"):
        validate_rks_reference(reference)


@pytest.mark.parametrize(
    "xc_code",
    [
        pytest.param("LDA_X + LDA_C_VWN", id="canonical"),
        pytest.param("LDA,VWN", id="lda-vwn"),
        pytest.param("SVWN", id="svwn"),
        pytest.param("1,7", id="libxc-identifiers"),
    ],
)
def test_rks_functional_aliases_are_accepted_by_normalized_semantics(
    rks_oracle_case,
    monkeypatch,
    xc_code,
):
    reference = rks_oracle_case.reference
    monkeypatch.setattr(reference, "xc", xc_code)

    assert validate_rks_reference(reference) is reference


@pytest.mark.parametrize(
    "xc_code",
    [
        pytest.param("0.5*LDA_X + LDA_C_VWN", id="coefficient"),
        pytest.param("PBE", id="gga"),
        pytest.param("TPSS", id="meta-gga"),
        pytest.param("B3LYP", id="hybrid"),
    ],
)
def test_rks_functional_domain_rejects_noncanonical_components(
    rks_oracle_case,
    monkeypatch,
    xc_code,
):
    reference = rks_oracle_case.reference
    monkeypatch.setattr(reference, "xc", xc_code)

    with pytest.raises(DeePHFCapabilityError, match="normalized LibXC components"):
        validate_rks_reference(reference)


def test_rks_rejects_registered_custom_libxc_even_with_canonical_components(
    rks_oracle_case,
    monkeypatch,
):
    reference = rks_oracle_case.reference
    monkeypatch.setitem(
        libxc._CUSTOM_FUNC_R,
        reference.xc,
        object(),
    )

    with pytest.raises(
        DeePHFCapabilityError,
        match="registered custom LibXC",
    ):
        validate_rks_reference(reference)


def test_rks_rejects_a_noncanonical_libxc_parameter_signature(
    rks_oracle_case,
    monkeypatch,
):
    original_evaluate = dft.numint.NumInt.eval_xc_eff

    def altered_evaluate(self, *args, **kwargs):
        values = list(original_evaluate(self, *args, **kwargs))
        values[0] = np.asarray(values[0]).copy() + 1.0e-7
        return tuple(values)

    monkeypatch.setattr(
        dft.numint.NumInt,
        "eval_xc_eff",
        altered_evaluate,
    )

    with pytest.raises(DeePHFCapabilityError, match="parameters do not match"):
        validate_rks_reference(rks_oracle_case.reference)


def test_rks_rejects_an_uncharacterized_libxc_version(
    rks_oracle_case,
    monkeypatch,
):
    monkeypatch.setattr(libxc, "__version__", "7.0.1-test")

    with pytest.raises(DeePHFCapabilityError, match="LibXC 7.0.0"):
        validate_rks_reference(rks_oracle_case.reference)


@pytest.mark.parametrize(
    "cutoff",
    [
        pytest.param(0.0, id="zero"),
        pytest.param(True, id="boolean"),
        pytest.param(np.nan, id="nan"),
    ],
)
def test_rks_rejects_noncanonical_numint_cutoff(
    rks_oracle_case,
    monkeypatch,
    cutoff,
):
    integration = rks_oracle_case.reference._numint
    had_instance_value = "cutoff" in integration.__dict__
    with monkeypatch.context() as patch:
        patch.setattr(integration, "cutoff", cutoff)
        with pytest.raises(DeePHFCapabilityError, match="native NumInt cutoff"):
            validate_rks_reference(rks_oracle_case.reference)
    if not had_instance_value:
        integration.__dict__.pop("cutoff", None)


def test_rks_rejects_range_separation_and_nonlocal_correlation_controls(
    rks_oracle_case,
    monkeypatch,
):
    reference = rks_oracle_case.reference
    cases = (
        (reference._numint, "omega", 0.2, "range-separation parameter"),
        (reference, "nlc", "VV10", "does not support NLC"),
    )
    for target, attribute, value, message in cases:
        with monkeypatch.context() as patch:
            patch.setattr(target, attribute, value)
            with pytest.raises(DeePHFCapabilityError, match=message):
                validate_rks_reference(reference)


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [
        pytest.param("alignment", True, "alignment", id="boolean-alignment"),
        pytest.param("cutoff", True, "cutoff", id="boolean-cutoff"),
    ],
)
def test_rks_grid_rejects_boolean_numeric_controls(
    rks_oracle_case,
    monkeypatch,
    attribute,
    value,
    message,
):
    reference = rks_oracle_case.reference
    had_instance_value = attribute in reference.grids.__dict__
    with monkeypatch.context() as patch:
        patch.setattr(reference.grids, attribute, value)
        with pytest.raises(DeePHFCapabilityError, match=message):
            validate_rks_reference(reference)
    if not had_instance_value:
        reference.grids.__dict__.pop(attribute, None)
    reference.grids.build(with_non0tab=True, sort_grids=False)


def test_rks_reference_rejects_boolean_small_density_cutoff(
    rks_oracle_case,
    monkeypatch,
):
    reference = rks_oracle_case.reference
    monkeypatch.setattr(reference, "small_rho_cutoff", False)

    with pytest.raises(DeePHFCapabilityError, match="small_rho_cutoff"):
        validate_rks_reference(reference)


def test_rks_requires_native_numint_and_exact_grid_controls(
    rks_oracle_case,
    monkeypatch,
):
    reference = rks_oracle_case.reference

    class DerivedNumInt(dft.numint.NumInt):
        pass

    cases = (
        (reference, "_numint", DerivedNumInt(), "exact native"),
        (reference, "small_rho_cutoff", 1.0e-7, "small_rho_cutoff=0"),
        (reference.grids, "prune", dft.gen_grid.nwchem_prune, "unpruned"),
        (reference.grids, "alignment", 8, "alignment=1"),
        (reference.grids, "atom_grid", {"O": (30, 50), "H": (20, 50)}, "20, 50"),
    )
    for target, attribute, value, message in cases:
        with monkeypatch.context() as patch:
            patch.setattr(target, attribute, value)
            with pytest.raises(DeePHFCapabilityError, match=message):
                validate_rks_reference(reference)
        if reference.grids.coords is None:
            reference.grids.build(with_non0tab=True, sort_grids=False)


def test_rks_requires_exact_native_grid_generators_and_tables(
    rks_oracle_case,
    monkeypatch,
):
    reference = rks_oracle_case.reference
    cases = (
        ("radi_method", radi.delley, "radial method"),
        (
            "radii_adjust",
            radi.becke_atomic_radii_adjust,
            "atomic-radii adjustment",
        ),
        ("becke_scheme", gen_grid.stratmann, "Becke partition"),
        ("atomic_radii", radi.COVALENT_RADII, "BRAGG_RADII"),
        ("cutoff", float(gen_grid.CUTOFF) * 2.0, "cutoff"),
    )
    for attribute, value, message in cases:
        had_instance_value = attribute in reference.grids.__dict__
        with monkeypatch.context() as patch:
            patch.setattr(reference.grids, attribute, value)
            with pytest.raises(DeePHFCapabilityError, match=message):
                validate_rks_reference(reference)
        if not had_instance_value:
            reference.grids.__dict__.pop(attribute, None)
        reference.grids.build(with_non0tab=True, sort_grids=False)


def test_rks_rejects_in_place_bragg_radii_table_mutation(rks_oracle_case):
    original = np.array(radi.BRAGG_RADII, copy=True)
    try:
        radi.BRAGG_RADII[1] += 1.0e-8
        with pytest.raises(
            DeePHFCapabilityError,
            match="BRAGG_RADII table was modified",
        ):
            validate_rks_reference(rks_oracle_case.reference)
    finally:
        radi.BRAGG_RADII[...] = original


def test_rks_rejects_replaced_module_and_grid_radial_default(
    rks_oracle_case,
    monkeypatch,
):
    original = radi.treutler_ahlrichs

    def replaced_radial_default(*args, **kwargs):
        return original(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(radi, "treutler_ahlrichs", replaced_radial_default)
        patch.setattr(
            gen_grid.Grids,
            "radi_method",
            replaced_radial_default,
        )
        with pytest.raises(DeePHFCapabilityError, match="radial method"):
            validate_rks_reference(rks_oracle_case.reference)


def test_rks_rejects_grid_instance_hooks_and_response_order_faults(
    rks_oracle_case,
    monkeypatch,
):
    reference = rks_oracle_case.reference
    with monkeypatch.context() as patch:
        patch.setitem(reference.grids.__dict__, "custom_hook", lambda: None)
        with pytest.raises(DeePHFCapabilityError, match="grid has unsupported instance hooks"):
            validate_rks_reference(reference)

    original_generator = pyscf_rks.rks_grad.grids_response_cc

    def reversed_blocks(grid):
        return tuple(reversed(tuple(original_generator(grid))))

    with monkeypatch.context() as patch:
        patch.setattr(
            pyscf_rks.rks_grad,
            "grids_response_cc",
            reversed_blocks,
        )
        with pytest.raises(
            DeePHFCapabilityError,
            match="grid-response generator was modified",
        ):
            validate_rks_reference(reference)


@pytest.mark.parametrize(
    ("field_name", "corruption", "message"),
    [
        pytest.param("coords", "nonfinite", "nonfinite", id="coordinates-nan"),
        pytest.param("weights", "shape", "shape", id="weights-shape"),
        pytest.param(
            "quadrature_weights",
            "float32",
            "quadrature weights must be float64",
            id="quadrature-float32",
        ),
    ],
)
def test_rks_rejects_invalid_grid_cache_arrays(
    rks_oracle_case,
    monkeypatch,
    field_name,
    corruption,
    message,
):
    reference = rks_oracle_case.reference
    value = np.asarray(getattr(reference.grids, field_name)).copy()
    if corruption == "nonfinite":
        value.reshape(-1)[0] = np.nan
    elif corruption == "shape":
        value = value[:-1]
    else:
        value = value.astype(np.float32)
    monkeypatch.setattr(reference.grids, field_name, value)

    with pytest.raises(DeePHFCapabilityError, match=message):
        validate_rks_reference(reference)


def test_rks_rejects_float_atom_index_grid_cache(
    rks_oracle_case,
    monkeypatch,
):
    reference = rks_oracle_case.reference
    atom_indices = np.asarray(reference.grids.atm_idx).astype(np.float64)
    monkeypatch.setattr(reference.grids, "atm_idx", atom_indices)

    with pytest.raises(DeePHFCapabilityError, match="fresh deterministic build"):
        validate_rks_reference(reference)


@pytest.mark.parametrize(
    "field_name",
    ["coords", "weights", "atm_idx", "quadrature_weights", "non0tab"],
)
def test_rks_rejects_mutated_prebuilt_grid_provenance(
    rks_oracle_case,
    monkeypatch,
    field_name,
):
    reference = rks_oracle_case.reference
    value = np.asarray(getattr(reference.grids, field_name)).copy()
    value.reshape(-1)[0] += 1
    monkeypatch.setattr(reference.grids, field_name, value)

    with pytest.raises(DeePHFCapabilityError, match="fresh deterministic build"):
        validate_rks_reference(reference)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        pytest.param("fractional", "integer closed-shell", id="fractional"),
        pytest.param("complex", "orbitals must be real", id="complex"),
        pytest.param("float32", "numpy.float64", id="float32"),
        pytest.param("nonfinite", "orbital state must be finite", id="nonfinite"),
        pytest.param("truncated", "complete square", id="truncated"),
        pytest.param("nonaufbau", "Aufbau ground-state root", id="root-swap"),
    ],
)
def test_rks_rejects_invalid_or_root_changed_orbital_state(
    rks_oracle_case,
    monkeypatch,
    mutation,
    message,
):
    reference = rks_oracle_case.reference
    if mutation == "fractional":
        occupations = np.asarray(reference.mo_occ).copy()
        occupations[0] = 1.5
        monkeypatch.setattr(reference, "mo_occ", occupations)
    elif mutation == "complex":
        monkeypatch.setattr(
            reference,
            "mo_coeff",
            np.asarray(reference.mo_coeff).astype(np.complex128),
        )
    elif mutation == "float32":
        monkeypatch.setattr(
            reference,
            "mo_energy",
            np.asarray(reference.mo_energy).astype(np.float32),
        )
    elif mutation == "nonfinite":
        energies = np.asarray(reference.mo_energy).copy()
        energies[0] = np.nan
        monkeypatch.setattr(reference, "mo_energy", energies)
    elif mutation == "truncated":
        monkeypatch.setattr(
            reference,
            "mo_coeff",
            np.asarray(reference.mo_coeff)[:, :-1].copy(),
        )
    else:
        occupations = np.asarray(reference.mo_occ).copy()
        last_occupied = np.flatnonzero(occupations > 0.0)[-1]
        first_virtual = np.flatnonzero(occupations == 0.0)[0]
        occupations[last_occupied] = 0.0
        occupations[first_virtual] = 2.0
        monkeypatch.setattr(reference, "mo_occ", occupations)

    with pytest.raises(DeePHFCapabilityError, match=message):
        validate_rks_reference(reference)


def test_rks_rejects_reference_molecule_and_numint_instance_hooks(
    rks_oracle_case,
    monkeypatch,
):
    reference = rks_oracle_case.reference
    cases = (
        (reference, "get_hcore", lambda *args, **kwargs: None, "reference has unsupported instance hooks"),
        (reference.mol, "atom_coords", lambda *args, **kwargs: None, "molecule has unsupported instance hooks"),
        (reference._numint, "eval_ao", lambda *args, **kwargs: None, "NumInt object has unsupported instance hooks"),
    )
    for target, attribute, value, message in cases:
        with monkeypatch.context() as patch:
            patch.setitem(target.__dict__, attribute, value)
            with pytest.raises(DeePHFCapabilityError, match=message):
                validate_rks_reference(reference)


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [
        pytest.param("with_df", object(), "density fitting", id="density-fitting"),
        pytest.param("with_solvent", object(), "solvent", id="solvent"),
        pytest.param("with_x2c", object(), "X2C", id="x2c"),
        pytest.param("mm_mol", object(), "QM/MM", id="qmmm"),
    ],
)
def test_rks_rejects_reference_decorations(
    rks_oracle_case,
    monkeypatch,
    attribute,
    value,
    message,
):
    reference = rks_oracle_case.reference
    monkeypatch.setattr(reference, attribute, value, raising=False)

    with pytest.raises(DeePHFCapabilityError, match=message):
        validate_rks_reference(reference)


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [
        pytest.param("symmetry", "C1", "symmetry-constrained", id="symmetry"),
        pytest.param("cart", True, "spherical AO", id="cartesian"),
        pytest.param("_ecp", {"O": object()}, "all-electron", id="ecp"),
        pytest.param("_pseudo", {"O": object()}, "pseudopotentials", id="pseudo"),
        pytest.param("omega", 0.2, "full Coulomb", id="range-separated"),
        pytest.param("nucmod", {0: 2}, "point nuclei", id="finite-nucleus"),
    ],
)
def test_rks_rejects_unsupported_molecular_physics(
    rks_oracle_case,
    monkeypatch,
    attribute,
    value,
    message,
):
    reference = rks_oracle_case.reference
    monkeypatch.setattr(reference.mol, attribute, value, raising=False)

    with pytest.raises(DeePHFCapabilityError, match=message):
        validate_rks_reference(reference)


def test_rks_rejects_ghost_centers_before_orbital_checks():
    molecule = gto.M(
        atom="O 0 0 0; ghost-H 0 0 1; H 0 1 0; H 1 0 0",
        basis="sto-3g",
        spin=0,
        verbose=0,
    )
    reference = dft.RKS(molecule)
    reference.converged = True

    with pytest.raises(DeePHFCapabilityError, match="real atoms; ghost indices"):
        validate_rks_reference(reference)


def test_rks_rejects_tampered_canonical_energy_and_total_energy(
    rks_oracle_case,
    monkeypatch,
):
    reference = rks_oracle_case.reference
    with monkeypatch.context() as patch:
        energies = np.asarray(reference.mo_energy).copy()
        energies[0] += 1.0e-3
        patch.setattr(reference, "mo_energy", energies)
        with pytest.raises(DeePHFCapabilityError, match="canonical SCF equations"):
            validate_rks_reference(reference)
    with monkeypatch.context() as patch:
        patch.setattr(reference, "e_tot", float(reference.e_tot) + 1.0e-4)
        with pytest.raises(DeePHFCapabilityError, match="total energy is inconsistent"):
            validate_rks_reference(reference)


def test_rks_rejects_a_polluted_ground_state_effective_potential(
    rks_oracle_case,
    monkeypatch,
):
    reference = rks_oracle_case.reference
    original_get_veff = dft.rks.RKS.get_veff

    def polluted_get_veff(self, *args, **kwargs):
        value = np.asarray(original_get_veff(self, *args, **kwargs)).copy()
        value[0, 0] += 1.0e-4
        return value

    monkeypatch.setattr(dft.rks.RKS, "get_veff", polluted_get_veff)

    with pytest.raises(DeePHFCapabilityError, match="does not match direct Coulomb"):
        validate_rks_reference(reference)


def test_rks_response_is_immutable_complete_and_audited(rks_oracle_case):
    response = rks_oracle_case.response
    diagnostics = response.diagnostics
    reference = rks_oracle_case.reference
    nocc = int(np.count_nonzero(np.asarray(reference.mo_occ) > 0.0))

    assert type(response) is RKSResponse
    assert type(diagnostics) is RKSResponseDiagnostics
    assert response.mo_response.shape == (reference.mol.natm, 3, reference.mol.nao, nocc)
    assert response.density_response.shape == (
        reference.mol.natm,
        3,
        reference.mol.nao,
        reference.mol.nao,
    )
    for response_field in fields(response):
        value = getattr(response, response_field.name)
        if isinstance(value, np.ndarray):
            assert value.dtype == np.dtype(np.float64)
            assert np.isfinite(value).all()
            assert not value.flags.writeable

    np.testing.assert_allclose(
        response.density_response,
        response.density_response_metric
        + response.density_response_occupied_virtual,
        rtol=0.0,
        atol=2.0e-15,
    )
    np.testing.assert_allclose(
        response.hamiltonian_derivative,
        response.hamiltonian_derivative_fixed_grid
        + response.xc_hamiltonian_derivative_grid_coordinate
        + response.xc_hamiltonian_derivative_grid_weight,
        rtol=0.0,
        atol=2.0e-15,
    )
    assert response.functional_provenance.components == ((1, 1.0), (7, 1.0))
    assert response.functional_provenance.numint_cutoff == 1.0e-13
    assert response.grid_provenance.sort_grids is False
    assert diagnostics.maximum_residual <= diagnostics.residual_tolerance
    assert diagnostics.operator_minimum_eigenvalue > diagnostics.operator_stability_tolerance
    assert diagnostics.operator_condition_number < diagnostics.operator_condition_tolerance
    assert diagnostics.metric_residual < diagnostics.invariant_tolerance
    assert diagnostics.particle_number_residual < diagnostics.invariant_tolerance


def test_rks_response_operator_satisfies_the_reference_neutral_protocol(
    rks_oracle_case,
):
    problem = RKSResponseAdapter(
        rks_oracle_case.reference
    ).linear_response_problem()
    vector = np.linspace(-0.37, 0.41, problem.dimension, dtype=np.float64)
    identity = np.eye(problem.dimension, dtype=np.float64)
    matrix = np.column_stack([problem.apply(root) for root in identity])

    assert isinstance(problem, ScalarAdjointProblem)
    assert not hasattr(problem, "dense_operator")
    assert matrix.shape == (problem.dimension, problem.dimension)
    assert matrix.dtype == np.dtype(np.float64)
    np.testing.assert_allclose(
        problem.apply(vector),
        matrix @ vector,
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        problem.apply_transpose(vector),
        matrix.T @ vector,
        rtol=0.0,
        atol=2.0e-15,
    )


@pytest.mark.parametrize(
    "response_options",
    [
        pytest.param({"cphf_tolerance": np.nan}, id="nan-cphf"),
        pytest.param({"residual_tolerance": True}, id="boolean-residual"),
        pytest.param({"invariant_tolerance": "1e-9"}, id="string-invariant"),
        pytest.param({"max_cycle": 1.5}, id="fractional-cycle"),
        pytest.param({"max_refinement_cycles": True}, id="boolean-refinement"),
        pytest.param({"operator_condition_tolerance": 1.0}, id="unit-condition"),
        pytest.param({"operator_dimension_limit": 0}, id="zero-dimension"),
    ],
)
def test_rks_response_rejects_invalid_controls(
    rks_oracle_case,
    response_options,
):
    with pytest.raises((TypeError, ValueError)):
        RKSResponseAdapter(rks_oracle_case.reference, **response_options)


def test_rks_response_operator_gates_dimension_stability_and_condition(
    rks_oracle_case,
):
    reference = rks_oracle_case.reference
    exact = RKSResponseAdapter(reference).validate_response_operator_exact()
    dimension, minimum, _maximum, condition = exact[:4]

    with pytest.raises(DeePHFCapabilityError, match="dimension exceeds"):
        RKSResponseAdapter(
            reference,
            operator_dimension_limit=dimension - 1,
        ).validate_response_operator_exact()
    with pytest.raises(DeePHFCapabilityError, match="unstable or singular"):
        RKSResponseAdapter(
            reference,
            operator_stability_tolerance=minimum * 1.01,
        ).validate_response_operator_exact()
    with pytest.raises(DeePHFCapabilityError, match="ill conditioned"):
        RKSResponseAdapter(
            reference,
            operator_condition_tolerance=max(
                1.0001,
                condition * 0.99,
            ),
        ).validate_response_operator_exact()


def test_rks_production_response_does_not_use_dense_debug_audit(
    rks_oracle_case,
    monkeypatch,
):
    def forbidden_dense_audit(*_args, **_kwargs):
        raise AssertionError("the RKS response matrix was materialized")

    monkeypatch.setattr(
        pyscf_rks._RKSLinearResponseCore,
        "_response_operator_matrix_and_diagnostics",
        forbidden_dense_audit,
    )
    response = RKSResponseAdapter(
        rks_oracle_case.reference,
        operator_dimension_limit=1,
    ).solve()

    assert response.diagnostics.response_dimension > 1
    assert response.diagnostics.operator_diagnostics_are_estimates is True


def test_rks_response_rejects_an_asymmetric_operator(
    rks_oracle_case,
    monkeypatch,
):
    original_apply = RKSResponseAdapter._apply_occupied_virtual_operator

    def asymmetric_apply(self, response, *args, **kwargs):
        response = np.asarray(response)
        result = np.asarray(original_apply(self, response, *args, **kwargs)).copy()
        flat_response = response.reshape(*response.shape[:-2], -1)
        flat_result = result.reshape(*result.shape[:-2], -1)
        flat_result[..., 0] += 1.0e-4 * flat_response[..., 1]
        return result

    monkeypatch.setattr(
        RKSResponseAdapter,
        "_apply_occupied_virtual_operator",
        asymmetric_apply,
    )

    with pytest.raises(RKSResponseError, match="violates symmetry"):
        RKSResponseAdapter(rks_oracle_case.reference).linear_response_problem()


def test_corrupted_cpks_solution_fails_the_independent_residual(
    rks_oracle_case,
    monkeypatch,
):
    reference = rks_oracle_case.reference
    original_solve = pyscf_rks.cphf.solve
    first_virtual = int(np.flatnonzero(np.asarray(reference.mo_occ) == 0.0)[0])

    def corrupted_solve(*args, **kwargs):
        response, energy_response = original_solve(*args, **kwargs)
        response = np.asarray(response).copy()
        response.reshape(-1, response.shape[-2], response.shape[-1])[
            :, first_virtual, 0
        ] += 1.0e-4
        return response, energy_response

    monkeypatch.setattr(pyscf_rks.cphf, "solve", corrupted_solve)

    with pytest.raises(RKSResponseError, match="residual exceeds tolerance"):
        RKSResponseAdapter(
            reference,
            max_refinement_cycles=0,
        ).solve()


def test_supplied_rks_response_rejects_foreign_or_resealed_forgery(
    rks_oracle_case,
):
    response = rks_oracle_case.response
    adapter = RKSResponseAdapter(rks_oracle_case.reference)
    foreign = _reseal(
        replace(
            response,
            reference_identity=response.reference_identity + 1,
        )
    )
    forged_density = np.array(response.density_response, copy=True)
    forged_density[0, 0, 0, 0] += 1.0e-5
    forged_density.setflags(write=False)
    forged = _reseal(replace(response, density_response=forged_density))

    with pytest.raises(RKSResponseError, match="belongs to another reference"):
        adapter.audit_response_equations(foreign)
    with pytest.raises(RKSResponseError):
        adapter.audit_response_equations(forged)


@pytest.mark.parametrize(
    "corruption",
    ["mutable", "float32", "object", "nonfinite"],
)
def test_supplied_rks_response_rejects_invalid_resealed_arrays(
    rks_oracle_case,
    corruption,
):
    response = rks_oracle_case.response
    value = np.array(response.density_response, copy=True)
    if corruption == "float32":
        value = value.astype(np.float32)
        value.setflags(write=False)
    elif corruption == "object":
        value = value.astype(object)
        value.setflags(write=False)
    elif corruption == "nonfinite":
        value[0, 0, 0, 0] = np.nan
        value.setflags(write=False)
    forged = _reseal(replace(response, density_response=value))

    with pytest.raises(RKSResponseError):
        RKSResponseAdapter(
            rks_oracle_case.reference
        ).audit_response_equations(forged)


def test_supplied_rks_response_rejects_stale_state_fingerprint(
    rks_oracle_case,
):
    response = _reseal(
        replace(rks_oracle_case.response, state_fingerprint="stale-geometry")
    )

    with pytest.raises(RKSResponseError, match="state is stale"):
        RKSResponseAdapter(
            rks_oracle_case.reference
        ).audit_response_equations(response)


@pytest.mark.parametrize(
    ("field_name", "message"),
    [
        pytest.param("mo_response_metric", "MO response partition", id="occupied-occupied-gauge"),
        pytest.param(
            "xc_hamiltonian_derivative_grid_coordinate",
            "grid-coordinate XC Hamiltonian derivative",
            id="grid-coordinate-h1",
        ),
        pytest.param(
            "xc_hamiltonian_derivative_grid_weight",
            "grid-weight XC Hamiltonian derivative",
            id="grid-weight-h1",
        ),
        pytest.param(
            "orbital_response_residual",
            "orbital residual",
            id="physical-residual",
        ),
    ],
)
def test_supplied_rks_response_rejects_resealed_equation_forgery(
    rks_oracle_case,
    field_name,
    message,
):
    response = rks_oracle_case.response
    value = np.array(getattr(response, field_name), copy=True)
    value.reshape(-1)[0] += 1.0e-5
    value.setflags(write=False)
    forged = _reseal(replace(response, **{field_name: value}))

    with pytest.raises(RKSResponseError, match=message):
        RKSResponseAdapter(
            rks_oracle_case.reference
        ).audit_response_equations(forged)


def test_supplied_rks_response_rejects_resealed_operator_diagnostics(
    rks_oracle_case,
):
    response = rks_oracle_case.response
    diagnostics = replace(
        response.diagnostics,
        operator_condition_number=(
            response.diagnostics.operator_condition_number + 1.0e-4
        ),
    )
    forged = _reseal(replace(response, diagnostics=diagnostics))

    with pytest.raises(
        RKSResponseError,
        match="operator_condition_number is not reproducible",
    ):
        RKSResponseAdapter(
            rks_oracle_case.reference
        ).audit_response_equations(forged)


def test_rks_method_rejects_same_state_response_from_an_independent_adapter(
    rks_oracle_case,
):
    method = rks_oracle_case.method
    trusted = method.response()
    foreign = RKSResponseAdapter(rks_oracle_case.reference).solve()

    assert foreign is not trusted
    assert foreign.state_fingerprint == trusted.state_fingerprint
    with pytest.raises(
        RKSResponseError,
        match="was not produced by this RKS DeePHF method",
    ):
        method.first_order_density(response=foreign)


def test_rks_method_rejects_coordinated_resealed_response_and_tolerance_forgery(
    rks_oracle_case,
):
    method = rks_oracle_case.method
    response = method.response()
    replacements = {}
    for field_name in (
        "mo_response",
        "mo_response_occupied_virtual",
        "coefficient_response",
        "coefficient_response_occupied_virtual",
        "density_response",
        "density_response_occupied_virtual",
        "orbital_response_residual",
    ):
        value = np.array(getattr(response, field_name), copy=True)
        value.reshape(-1)[0] += 1.0e-5
        value.setflags(write=False)
        replacements[field_name] = value
    replacements["diagnostics"] = replace(
        response.diagnostics,
        residual_tolerance=1.0,
    )
    forged = _reseal(replace(response, **replacements))

    with pytest.raises(
        RKSResponseError,
        match="was not produced by this RKS DeePHF method",
    ):
        method.first_order_density(response=forged)


def test_rks_method_rejects_resealed_residual_history_and_cycle_forgery(
    rks_oracle_case,
):
    method = rks_oracle_case.method
    response = method.response()
    diagnostics = replace(
        response.diagnostics,
        residual_history=response.diagnostics.residual_history + (0.0,),
        refinement_cycles=response.diagnostics.refinement_cycles + 1,
    )
    forged = _reseal(replace(response, diagnostics=diagnostics))

    with pytest.raises(
        RKSResponseError,
        match="was not produced by this RKS DeePHF method",
    ):
        method.first_order_density(response=forged)


def test_rks_method_rejects_same_object_mutation_even_after_resealing(
    rks_oracle_case,
):
    method = rks_oracle_case.method
    response = method.response()
    original_density = response.density_response
    original_integrity = response.integrity_fingerprint
    changed_density = np.array(original_density, copy=True)
    changed_density[0, 0, 0, 0] += 1.0e-5
    changed_density.setflags(write=False)
    try:
        object.__setattr__(response, "density_response", changed_density)
        object.__setattr__(
            response,
            "integrity_fingerprint",
            pyscf_rks.rks_response_integrity_fingerprint(response),
        )
        with pytest.raises(
            RKSResponseError,
            match="changed after it was produced",
        ):
            method.first_order_density(response=response)
    finally:
        object.__setattr__(response, "density_response", original_density)
        object.__setattr__(response, "integrity_fingerprint", original_integrity)


def test_trusted_rks_response_can_be_reused_without_another_solve(
    rks_oracle_case,
    monkeypatch,
):
    method = rks_oracle_case.method
    first_response = method.response()
    second_response = method.response()

    def forbidden_solve(*args, **kwargs):
        raise AssertionError("trusted response consumption called solve")

    monkeypatch.setattr(RKSResponseAdapter, "solve", forbidden_solve)
    first = method.first_order_density(response=first_response)
    second = method.first_order_density(response=second_response)
    repeated = method.first_order_density(response=first_response)

    assert first is first_response.density_response
    assert second is second_response.density_response
    assert repeated is first_response.density_response


def test_rks_response_and_options_are_mutually_exclusive(rks_oracle_case):
    method = rks_oracle_case.method
    response = rks_oracle_case.response

    for function in (
        method.first_order_density,
        method.dq_dR_response,
        method.dq_dR_relaxed,
    ):
        with pytest.raises(ValueError, match="mutually exclusive"):
            function(response=response, cphf_tolerance=1.0e-10)


def test_rks_direct_backend_remains_default_and_rejects_scanner_and_force_data_paths(
    rks_oracle_case,
):
    reference = rks_oracle_case.reference
    method = rks_oracle_case.method
    driver = method.nuc_grad_method(backend="direct")

    assert type(driver) is RKSDeePHFGradients
    assert driver.backend == "direct"
    zvector_driver = method.nuc_grad_method(backend="zvector")
    assert type(zvector_driver) is RKSDeePHFZVectorGradients
    assert zvector_driver.backend == "zvector"
    with pytest.raises(RKSResponseError, match="gradient scanner"):
        driver.as_scanner()
    with pytest.raises(ValueError, match="unsupported direct backend options"):
        method.nuc_grad_method(fallback="explicit")
    with pytest.raises(TypeError, match="requires an exact DeePHF method"):
        RHFDeePHFGradients(method)
    with pytest.raises(TypeError, match="requires an exact DeePHF method"):
        RHFDeePHFZVectorGradients(method)
    with pytest.raises(DeePHFCapabilityError, match="native pyscf.scf.hf.RHF"):
        generate_rhf_force_frame(
            reference,
            projector_basis=ORACLE_PROJECTOR_BASIS,
            e_target=np.float64(reference.e_tot),
            f_target=np.zeros((reference.mol.natm, 3)),
        )


def test_rks_native_gradient_result_preserves_observable_grid_partitions(
    rks_oracle_case,
):
    driver = rks_oracle_case.gradient_driver
    native = driver.native_gradient_result

    assert type(native) is RKSNativeGradient
    np.testing.assert_allclose(
        native.gradient,
        native.gradient_without_grid_response
        + native.xc_grid_coordinate
        + native.xc_grid_weight,
        rtol=0.0,
        atol=5.0e-14,
    )
    assert driver.reference_gradient is native.gradient
    assert driver.reference_gradient_without_grid_response is native.gradient_without_grid_response
    assert driver.reference_gradient_xc_grid_coordinate is native.xc_grid_coordinate
    assert driver.reference_gradient_xc_grid_weight is native.xc_grid_weight
    assert driver.reference_gradient_reconstruction_residual == native.reconstruction_residual


def test_rks_gradient_failure_clears_results_and_trusted_response(
    rks_oracle_case,
    monkeypatch,
):
    method = rks_oracle_case.method
    driver = rks_oracle_case.gradient_driver
    assert driver.de_full is not None
    assert method._trusted_response is not None

    def failed_solve(*args, **kwargs):
        raise RKSResponseError("injected RKS response failure")

    monkeypatch.setattr(RKSResponseAdapter, "solve", failed_solve)

    with pytest.raises(RKSResponseError, match="injected RKS response failure"):
        driver.kernel()

    assert method._trusted_response is None
    assert method._trusted_response_integrity is None
    for name in (
        "response_result",
        "native_gradient_result",
        "dq_dR_explicit",
        "dq_dR_response",
        "dq_dR_relaxed",
        "correction_gradient",
        "de_full",
        "de",
    ):
        assert getattr(driver, name) is None


def test_rks_grid_quadrature_failure_clears_trusted_response(
    rks_oracle_case,
    monkeypatch,
):
    method = rks_oracle_case.method

    def failed_eval_ao(*args, **kwargs):
        raise RuntimeError("injected RKS grid quadrature failure")

    monkeypatch.setattr(dft.numint.NumInt, "eval_ao", failed_eval_ao)

    with pytest.raises(
        (DeePHFCapabilityError, RKSResponseError),
        match="injected RKS grid quadrature failure",
    ):
        method.response()
    assert method._trusted_response is None
    assert method._trusted_response_integrity is None


def test_rks_native_gradient_failure_clears_driver_results(
    rks_oracle_case,
    monkeypatch,
):
    driver = rks_oracle_case.method.nuc_grad_method()

    def failed_native_gradient(*args, **kwargs):
        raise RKSResponseError("injected native RKS gradient failure")

    monkeypatch.setattr(
        rks_gradient,
        "native_rks_gradient",
        failed_native_gradient,
    )

    with pytest.raises(RKSResponseError, match="injected native RKS gradient failure"):
        driver.kernel()
    assert driver.response_result is None
    assert driver.native_gradient_result is None
    assert driver.correction_gradient is None
    assert driver.de_full is None
    assert driver.de is None


def test_rks_model_and_projector_failures_do_not_enter_response_fallback(
    rks_oracle_case,
):
    method = rks_oracle_case.method
    original_model = method.model
    try:
        method.model = object()
        with pytest.raises(DeePHFCapabilityError):
            method.response()
        assert method._trusted_response is None
    finally:
        method.model = original_model

    original_descriptor = method._descriptor
    try:
        method._descriptor = object()
        with pytest.raises(DeePHFCapabilityError, match="descriptor identity changed"):
            method.response()
        assert method._trusted_response is None
    finally:
        method._descriptor = original_descriptor


def test_rks_nondifferentiable_descriptor_fails_before_response(
    rks_oracle_case,
):
    projector_basis = [[4, [0.2, 1.0]]]
    model = CorrNet(
        input_dim=9,
        hidden_sizes=(2,),
        proj_basis=projector_basis,
    ).double()
    with torch.no_grad():
        model.linear.weight.zero_()
        model.linear.bias.fill_(0.007)
        for parameter in model.densenet.parameters():
            parameter.zero_()
    method = RKSDeePHF(
        rks_oracle_case.reference,
        model.eval(),
        projector_basis=projector_basis,
    )

    with pytest.raises(
        DescriptorDifferentiabilityError,
        match="eigenvalue gap|structural zero block",
    ):
        method.response()
    assert method._trusted_response is None


def test_rks_gradient_driver_rejects_corrupted_binding_and_invalid_atoms(
    rks_oracle_case,
):
    driver = rks_oracle_case.method.nuc_grad_method()

    with pytest.raises(TypeError, match="atom indices must be integers"):
        driver.kernel(atmlst=[True])
    assert driver.response_result is None
    assert driver.de_full is None

    driver._backend = "zvector"
    with pytest.raises(RKSResponseError, match="binding is invalid"):
        driver.kernel()
    assert driver.response_result is None
    assert driver.de_full is None


def test_grid_weight_derivative_fault_is_rejected_before_any_response_solve(
    rks_oracle_case,
    monkeypatch,
):
    reference = rks_oracle_case.reference
    method = rks_oracle_case.method
    driver = rks_oracle_case.gradient_driver
    original_generator = pyscf_rks.rks_grad.grids_response_cc
    solve_calls = 0

    def perturbed_grid_response(grid):
        blocks = []
        for coordinates, weights, weight_derivative in original_generator(grid):
            changed = np.array(weight_derivative, copy=True)
            changed[0, 0, 0] += 1.0e-3
            changed[1, 0, 0] -= 1.0e-3
            perturbation = changed - np.asarray(weight_derivative)
            assert np.max(np.abs(perturbation.sum(axis=0))) < 1.0e-15
            blocks.append((coordinates, weights, changed))
        return tuple(blocks)

    def forbidden_cpks_solve(*args, **kwargs):
        nonlocal solve_calls
        solve_calls += 1
        raise AssertionError("grid-weight fault reached the CPKS solver")

    monkeypatch.setattr(
        pyscf_rks.rks_grad,
        "grids_response_cc",
        perturbed_grid_response,
    )
    monkeypatch.setattr(
        pyscf_rks,
        "_SUPPORTED_GRIDS_RESPONSE",
        perturbed_grid_response,
    )
    monkeypatch.setattr(pyscf_rks.cphf, "solve", forbidden_cpks_solve)

    for operation in (
        lambda: validate_rks_reference(reference),
        lambda: RKSResponseAdapter(reference).solve(),
        method.response,
        lambda: pyscf_rks.native_rks_gradient(reference),
        driver.kernel,
    ):
        with pytest.raises(
            (DeePHFCapabilityError, RKSResponseError),
            match="does not match independent finite differences",
        ):
            operation()

    assert solve_calls == 0
    assert method._trusted_response is None
    assert method._trusted_response_integrity is None
    for name in (
        "response_result",
        "native_gradient_result",
        "dq_dR_explicit",
        "dq_dR_response",
        "dq_dR_relaxed",
        "correction_gradient",
        "de_full",
        "de",
    ):
        assert getattr(driver, name) is None


def test_grid_host_block_repartition_is_rejected_before_response_solve(
    rks_oracle_case,
    monkeypatch,
):
    reference = rks_oracle_case.reference
    original_generator = pyscf_rks.rks_grad.grids_response_cc
    solve_calls = 0

    def repartitioned_grid_response(grid):
        original_blocks = tuple(original_generator(grid))
        assert tuple(len(block[1]) for block in original_blocks) == (
            1000,
            1000,
            1000,
        )
        coordinates = np.vstack([block[0] for block in original_blocks])
        weights = np.hstack([block[1] for block in original_blocks])
        weight_derivative = np.concatenate(
            [block[2] for block in original_blocks],
            axis=-1,
        )
        boundaries = (0, 500, 2000, 3000)
        return tuple(
            (
                coordinates[start:stop],
                weights[start:stop],
                weight_derivative[..., start:stop],
            )
            for start, stop in zip(boundaries, boundaries[1:])
        )

    def forbidden_cpks_solve(*args, **kwargs):
        nonlocal solve_calls
        solve_calls += 1
        raise AssertionError("grid host-block fault reached the CPKS solver")

    monkeypatch.setattr(
        pyscf_rks.rks_grad,
        "grids_response_cc",
        repartitioned_grid_response,
    )
    monkeypatch.setattr(
        pyscf_rks,
        "_SUPPORTED_GRIDS_RESPONSE",
        repartitioned_grid_response,
    )
    monkeypatch.setattr(pyscf_rks.cphf, "solve", forbidden_cpks_solve)

    for operation in (
        lambda: validate_rks_reference(reference),
        lambda: RKSResponseAdapter(reference).solve(),
        lambda: pyscf_rks.native_rks_gradient(reference),
    ):
        with pytest.raises(
            (DeePHFCapabilityError, RKSResponseError),
            match="host-atom block shape",
        ):
            operation()

    assert solve_calls == 0


def test_cross_molecule_strict_rks_smoke_and_block_weight_fault(
    monkeypatch,
):
    hydrogen = _run_cross_molecule_rks(
        "H 0.11 -0.07 0.60; H -0.16 0.09 1.40"
    )
    lithium_hydride = _run_cross_molecule_rks(
        "Li 0.13 -0.19 0.31; H -0.17 0.23 3.11"
    )

    assert validate_rks_reference(hydrogen) is hydrogen
    assert validate_rks_reference(lithium_hydride) is lithium_hydride
    adapter = RKSResponseAdapter(hydrogen)
    response = adapter.solve()
    adapter.audit_response_equations(response)
    assert response.density_response.shape == (
        hydrogen.mol.natm,
        3,
        hydrogen.mol.nao,
        hydrogen.mol.nao,
    )

    original_generator = pyscf_rks.rks_grad.grids_response_cc

    def changed_block_weight(grid):
        blocks = []
        for coordinates, weights, weight_derivative in original_generator(grid):
            changed_weights = np.array(weights, copy=True)
            changed_weights[0] += 1.0e-8
            blocks.append(
                (coordinates, changed_weights, weight_derivative)
            )
        return tuple(blocks)

    assert 0.0 < pyscf_rks._GRID_RESPONSE_WEIGHT_ATOL < 1.0e-170
    monkeypatch.setattr(
        pyscf_rks.rks_grad,
        "grids_response_cc",
        changed_block_weight,
    )
    monkeypatch.setattr(
        pyscf_rks,
        "_SUPPORTED_GRIDS_RESPONSE",
        changed_block_weight,
    )

    with pytest.raises(
        DeePHFCapabilityError,
        match="host block does not match the energy grid",
    ):
        validate_rks_reference(hydrogen)
