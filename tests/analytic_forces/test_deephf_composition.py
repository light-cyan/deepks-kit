import numpy as np
import pytest
import torch
from pyscf import gto, scf

from deepks.deephf import DeePHF, DeePHFCapabilityError, validate_reference
from deepks.descriptor import DescriptorDifferentiabilityError
from deepks.deepks import RDeePKS
from deepks.model.model import CorrNet


PROJECTOR_BASIS = [[0, [1.0, 1.0]]]


def _make_hydrogen_molecule():
    return gto.M(
        atom="H 0 0 0; H 0 0 1.4",
        basis="sto-3g",
        unit="Bohr",
        verbose=0,
    )


def _run_rhf(molecule):
    reference = scf.RHF(molecule)
    reference.conv_tol = 1.0e-12
    reference.kernel()
    assert reference.converged
    return reference


def _make_model():
    model = CorrNet(
        input_dim=1,
        hidden_sizes=(2,),
        proj_basis=PROJECTOR_BASIS,
    ).double()
    with torch.no_grad():
        model.linear.weight.fill_(0.01)
        model.linear.bias.fill_(0.002)
        for parameter in model.densenet.parameters():
            parameter.zero_()
    return model.eval()


def test_deephf_composes_energy_without_mutating_the_native_reference():
    reference = _run_rhf(_make_hydrogen_molecule())
    fock_before = np.asarray(reference.get_fock()).copy()
    density_before = np.asarray(reference.make_rdm1()).copy()
    mo_coeff_before = np.asarray(reference.mo_coeff).copy()
    mo_energy_before = np.asarray(reference.mo_energy).copy()
    mo_occ_before = np.asarray(reference.mo_occ).copy()
    energy_before = float(reference.e_tot)
    converged_before = bool(reference.converged)

    method = DeePHF(
        reference,
        _make_model(),
        projector_basis=PROJECTOR_BASIS,
    )
    total_energy = method.kernel()
    diagnostics = method.validate_force_compatibility()

    assert method.reference is reference
    assert method.e_base == energy_before
    assert abs(method.e_corr) > 1.0e-6
    assert total_energy == method.e_base + method.e_corr
    assert method.e_tot == total_energy
    assert diagnostics.structural_zero_blocks == ()
    assert np.isinf(diagnostics.minimum_scaled_gap)
    assert validate_reference(reference) is reference

    assert reference.converged is converged_before
    assert reference.e_tot == energy_before
    np.testing.assert_allclose(reference.get_fock(), fock_before, rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(reference.make_rdm1(), density_before)
    np.testing.assert_array_equal(reference.mo_coeff, mo_coeff_before)
    np.testing.assert_array_equal(reference.mo_energy, mo_energy_before)
    np.testing.assert_array_equal(reference.mo_occ, mo_occ_before)

    for analytic_gradient_name in ("nuc_grad_method", "gradient", "forces"):
        assert not hasattr(method, analytic_gradient_name)


def test_deephf_rejects_an_unconverged_reference():
    reference = scf.RHF(_make_hydrogen_molecule())
    assert not reference.converged

    with pytest.raises(DeePHFCapabilityError, match="must be converged"):
        DeePHF(reference, None, projector_basis=PROJECTOR_BASIS)


def test_deephf_rejects_a_ghost_reference():
    molecule = gto.M(
        atom="He 0 0 0; X-H 0 0 1.4",
        basis="sto-3g",
        unit="Bohr",
        verbose=0,
    )
    reference = _run_rhf(molecule)

    with pytest.raises(DeePHFCapabilityError, match=r"ghost indices: \[1\]"):
        DeePHF(reference, None, projector_basis=PROJECTOR_BASIS)


def test_deephf_rejects_decorated_and_deepks_references():
    molecule = _make_hydrogen_molecule()
    decorated_reference = scf.RHF(molecule).density_fit()
    decorated_reference.conv_tol = 1.0e-12
    decorated_reference.kernel()
    assert decorated_reference.converged

    deepks_reference = RDeePKS(
        molecule,
        None,
        projector_basis=PROJECTOR_BASIS,
    )
    deepks_reference.conv_tol = 1.0e-12
    deepks_reference.kernel()
    assert deepks_reference.converged

    for unsupported_reference in (decorated_reference, deepks_reference):
        with pytest.raises(
            DeePHFCapabilityError,
            match="undecorated native pyscf.scf.hf.RHF reference",
        ):
            DeePHF(
                unsupported_reference,
                None,
                projector_basis=PROJECTOR_BASIS,
            )


def test_deephf_rejects_fractional_occupations_and_complex_orbitals():
    fractional_reference = _run_rhf(_make_hydrogen_molecule())
    fractional_reference.mo_occ = fractional_reference.mo_occ.copy()
    fractional_reference.mo_occ[0] = 1.5
    with pytest.raises(DeePHFCapabilityError, match="integer closed-shell"):
        DeePHF(
            fractional_reference,
            None,
            projector_basis=PROJECTOR_BASIS,
        )

    complex_reference = _run_rhf(_make_hydrogen_molecule())
    complex_reference.mo_coeff = complex_reference.mo_coeff.astype(complex)
    with pytest.raises(DeePHFCapabilityError, match="orbitals must be real"):
        DeePHF(
            complex_reference,
            None,
            projector_basis=PROJECTOR_BASIS,
        )


def test_deephf_validates_model_sensitivity_in_a_structural_zero_space():
    reference = _run_rhf(_make_hydrogen_molecule())
    projector_basis = [[1, [1.0, 1.0]]]

    def make_model(zero_space_weights):
        model = CorrNet(
            input_dim=3,
            hidden_sizes=(2,),
            proj_basis=projector_basis,
        ).double()
        with torch.no_grad():
            model.linear.weight[0] = torch.tensor(
                [*zero_space_weights, -0.3],
                dtype=torch.float64,
            )
            model.linear.bias.zero_()
            for parameter in model.densenet.parameters():
                parameter.zero_()
        return model.eval()

    compatible = DeePHF(
        reference,
        make_model((0.2, 0.2)),
        projector_basis=projector_basis,
    )
    diagnostics = compatible.validate_force_compatibility()
    assert diagnostics.structural_zero_blocks == (
        (0, 0, 0, 2),
        (1, 0, 0, 2),
    )

    incompatible = DeePHF(
        reference,
        make_model((0.2, 0.25)),
        projector_basis=projector_basis,
    )
    with pytest.raises(
        DescriptorDifferentiabilityError,
        match="structural zero block sensitivity spread",
    ):
        incompatible.validate_force_compatibility()
