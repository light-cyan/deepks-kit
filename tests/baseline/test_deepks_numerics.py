import numpy as np
import torch
from pyscf import gto

from deepks.model.model import CorrNet
from deepks.scf.penalty import CoulombPenalty
from deepks.scf.scf import DSCF, UDSCF
from deepks.utils import get_shell_sec, load_basis


MOLECULE_COORDINATES = np.array([[0.10, -0.20, 0.05], [0.70, 0.30, 0.40]])
PROJECTOR_BASIS = "sto-3g@H"
EXPECTED_TOTAL_ENERGY = -2.492903870493804
EXPECTED_CORRECTION_ENERGY = 0.036782841678991
EXPECTED_DESCRIPTOR_VALUES = np.array(
    [[1.902421760735855], [1.375862439946038]]
)
EXPECTED_TOTAL_GRADIENT = np.array(
    [
        [0.953525479316305, 0.794604566096920, 0.556223196267844],
        [-0.953525479316304, -0.794604566096921, -0.556223196267844],
    ]
)
EXPECTED_EXPLICIT_CORRECTION_GRADIENT = np.array(
    [
        [0.008912442645454, 0.007427035537879, 0.005198924876515],
        [-0.008912442645454, -0.007427035537879, -0.005198924876515],
    ]
)
EXPECTED_UNRESTRICTED_ENERGY = -2.295254330377794
EXPECTED_UNRESTRICTED_DESCRIPTOR_VALUES = np.array(
    [[1.907798783099344], [1.702201584722110]]
)
EXPECTED_UNRESTRICTED_GRADIENT = np.array(
    [
        [1.516723476340869, 1.263936230284059, 0.884755361198841],
        [-1.516723476340870, -1.263936230284058, -0.884755361198839],
    ]
)
EXPECTED_PENALIZED_ENERGY = -2.5284420352365866
EXPECTED_PENALIZED_DENSITY = np.array(
    [
        [1.431363818353504, 0.322496330787210],
        [0.322496330787210, 0.072660690481089],
    ]
)
DFT_XC = "lda,vwn"
EXPECTED_RKS_ENERGY = -2.463517186762549
EXPECTED_RKS_DESCRIPTOR_VALUES = np.array(
    [[1.904937456638993], [1.397245489178295]]
)
EXPECTED_RKS_GRADIENT = np.array(
    [
        [0.983915741969157, 0.821517276671565, 0.577257624212020],
        [-0.983915741969157, -0.821517276671566, -0.577257624212020],
    ]
)
EXPECTED_UKS_ENERGY = -2.272540190512929
EXPECTED_UKS_DESCRIPTOR_VALUES = np.array(
    [[1.909219790166272], [1.720331607111384]]
)
EXPECTED_UKS_GRADIENT = np.array(
    [
        [1.522443569811000, 1.272876184897640, 0.882241244360377],
        [-1.522443569811001, -1.272876184897639, -0.882241244360375],
    ]
)


def make_molecule(coordinates):
    return gto.M(
        atom=[("He", coordinates[0]), ("H", coordinates[1])],
        basis="sto-3g",
        charge=1,
        spin=0,
        unit="Bohr",
        verbose=0,
    )


def make_deterministic_model():
    input_dimension = sum(get_shell_sec(load_basis(PROJECTOR_BASIS)))
    model = CorrNet(
        input_dim=input_dimension,
        hidden_sizes=(3,),
        proj_basis=PROJECTOR_BASIS,
    ).double()
    with torch.no_grad():
        model.linear.weight.fill_(0.01)
        model.linear.bias.fill_(0.002)
        for parameter in model.densenet.parameters():
            parameter.zero_()
    return model.eval()


def make_unrestricted_molecule(coordinates):
    return gto.M(
        atom=[("He", coordinates[0]), ("H", coordinates[1])],
        basis="sto-3g",
        charge=0,
        spin=1,
        unit="Bohr",
        verbose=0,
    )


def make_unrestricted_model():
    input_dimension = sum(get_shell_sec(load_basis(PROJECTOR_BASIS)))
    model = CorrNet(
        input_dim=input_dimension,
        hidden_sizes=(3,),
        proj_basis=PROJECTOR_BASIS,
    ).double()
    with torch.no_grad():
        model.linear.weight.fill_(0.015)
        model.linear.bias.fill_(-0.002)
        for parameter in model.densenet.parameters():
            parameter.zero_()
    return model.eval()


def run_deepks(coordinates, model):
    method = DSCF(make_molecule(coordinates), model)
    method.conv_tol = 1e-12
    method.conv_tol_grad = 1e-11
    method.max_cycle = 100
    energy = method.kernel()
    assert method.converged
    return method, energy


def run_unrestricted_deepks(coordinates, model):
    method = UDSCF(make_unrestricted_molecule(coordinates), model)
    method.conv_tol = 1e-12
    method.conv_tol_grad = 1e-11
    method.max_cycle = 100
    energy = method.kernel()
    assert method.converged
    return method, energy


def configure_dft_grid(method):
    method.grids.level = 0
    method.grids.prune = None
    method.small_rho_cutoff = 0


def run_rks_deepks(coordinates, model):
    method = DSCF(make_molecule(coordinates), model, xc=DFT_XC)
    configure_dft_grid(method)
    method.conv_tol = 1e-12
    method.conv_tol_grad = 1e-11
    method.max_cycle = 100
    energy = method.kernel()
    assert method.converged
    return method, energy


def run_uks_deepks(coordinates, model):
    method = UDSCF(make_unrestricted_molecule(coordinates), model, xc=DFT_XC)
    configure_dft_grid(method)
    method.conv_tol = 1e-12
    method.conv_tol_grad = 1e-11
    method.max_cycle = 100
    energy = method.kernel()
    assert method.converged
    return method, energy


def finite_difference_first_atom(run_method, coordinates, model, step=1e-4):
    finite_difference = np.empty(3)
    for coordinate_index in range(3):
        forward_coordinates = coordinates.copy()
        backward_coordinates = coordinates.copy()
        forward_coordinates[0, coordinate_index] += step
        backward_coordinates[0, coordinate_index] -= step
        _, forward_energy = run_method(forward_coordinates, model)
        _, backward_energy = run_method(backward_coordinates, model)
        finite_difference[coordinate_index] = (
            forward_energy - backward_energy
        ) / (2 * step)
    return finite_difference


def test_nonzero_self_consistent_deepks_energy_and_gradient_baseline():
    model = make_deterministic_model()
    method, total_energy = run_deepks(MOLECULE_COORDINATES, model)
    density = method.make_rdm1()
    correction_energy, correction_potential = method.get_corr(density)

    np.testing.assert_allclose(
        total_energy,
        EXPECTED_TOTAL_ENERGY,
        rtol=0,
        atol=2e-11,
    )
    np.testing.assert_allclose(
        correction_energy,
        EXPECTED_CORRECTION_ENERGY,
        rtol=0,
        atol=2e-11,
    )
    np.testing.assert_allclose(
        method.make_eig(density),
        EXPECTED_DESCRIPTOR_VALUES,
        rtol=2e-11,
        atol=2e-11,
    )
    assert np.linalg.norm(correction_potential) > 1e-3

    gradient_method = method.nuc_grad_method()
    analytic_gradient = gradient_method.kernel()
    np.testing.assert_allclose(
        analytic_gradient,
        EXPECTED_TOTAL_GRADIENT,
        rtol=2e-11,
        atol=2e-11,
    )
    np.testing.assert_allclose(
        gradient_method.dec,
        EXPECTED_EXPLICIT_CORRECTION_GRADIENT,
        rtol=2e-11,
        atol=2e-11,
    )
    np.testing.assert_allclose(
        analytic_gradient - gradient_method.get_base(),
        gradient_method.dec,
        rtol=2e-12,
        atol=2e-12,
    )
    assert np.linalg.norm(gradient_method.dec) > 1e-3

    step = 1e-4
    finite_difference = np.empty(3)
    for coordinate_index in range(3):
        forward_coordinates = MOLECULE_COORDINATES.copy()
        backward_coordinates = MOLECULE_COORDINATES.copy()
        forward_coordinates[0, coordinate_index] += step
        backward_coordinates[0, coordinate_index] -= step
        _, forward_energy = run_deepks(forward_coordinates, model)
        _, backward_energy = run_deepks(backward_coordinates, model)
        finite_difference[coordinate_index] = (
            forward_energy - backward_energy
        ) / (2 * step)

    np.testing.assert_allclose(
        analytic_gradient[0],
        finite_difference,
        rtol=2e-7,
        atol=5e-8,
    )
    np.testing.assert_allclose(
        analytic_gradient.sum(axis=0),
        np.zeros(3),
        rtol=0,
        atol=2e-12,
    )


def test_unrestricted_deepks_spin_summed_descriptor_and_gradient_baseline():
    model = make_unrestricted_model()
    method, total_energy = run_unrestricted_deepks(MOLECULE_COORDINATES, model)
    spin_density = method.make_rdm1()
    analytic_gradient = method.nuc_grad_method().kernel()

    assert spin_density.shape == (2, 2, 2)
    np.testing.assert_allclose(
        total_energy,
        EXPECTED_UNRESTRICTED_ENERGY,
        rtol=0.0,
        atol=3e-10,
    )
    np.testing.assert_allclose(
        method.make_eig(spin_density),
        EXPECTED_UNRESTRICTED_DESCRIPTOR_VALUES,
        rtol=0.0,
        atol=3e-10,
    )
    np.testing.assert_allclose(
        analytic_gradient,
        EXPECTED_UNRESTRICTED_GRADIENT,
        rtol=0.0,
        atol=3e-9,
    )

    step = 1e-4
    finite_difference = np.empty(3)
    for coordinate_index in range(3):
        forward_coordinates = MOLECULE_COORDINATES.copy()
        backward_coordinates = MOLECULE_COORDINATES.copy()
        forward_coordinates[0, coordinate_index] += step
        backward_coordinates[0, coordinate_index] -= step
        _, forward_energy = run_unrestricted_deepks(forward_coordinates, model)
        _, backward_energy = run_unrestricted_deepks(backward_coordinates, model)
        finite_difference[coordinate_index] = (
            forward_energy - backward_energy
        ) / (2 * step)

    np.testing.assert_allclose(
        analytic_gradient[0],
        finite_difference,
        rtol=3e-7,
        atol=5e-8,
    )
    np.testing.assert_allclose(
        analytic_gradient.sum(axis=0),
        np.zeros(3),
        rtol=0.0,
        atol=3e-12,
    )


def test_rks_deepks_energy_descriptor_and_gradient_baseline():
    model = make_deterministic_model()
    method, total_energy = run_rks_deepks(MOLECULE_COORDINATES, model)
    density = method.make_rdm1()
    descriptor_values = method.make_eig(density)
    gradient_method = method.nuc_grad_method()
    gradient_method.grid_response = True
    analytic_gradient = gradient_method.kernel()

    assert density.shape == (2, 2)
    assert descriptor_values.shape == (2, 1)
    assert analytic_gradient.shape == (2, 3)
    assert np.isfinite(total_energy)
    assert np.isfinite(descriptor_values).all()
    assert np.isfinite(analytic_gradient).all()
    np.testing.assert_allclose(
        total_energy,
        EXPECTED_RKS_ENERGY,
        rtol=0.0,
        atol=5e-10,
    )
    np.testing.assert_allclose(
        descriptor_values,
        EXPECTED_RKS_DESCRIPTOR_VALUES,
        rtol=0.0,
        atol=5e-10,
    )
    np.testing.assert_allclose(
        analytic_gradient,
        EXPECTED_RKS_GRADIENT,
        rtol=0.0,
        atol=5e-9,
    )

    finite_difference = finite_difference_first_atom(
        run_rks_deepks,
        MOLECULE_COORDINATES,
        model,
    )
    np.testing.assert_allclose(
        analytic_gradient[0],
        finite_difference,
        rtol=2e-7,
        atol=5e-8,
    )
    np.testing.assert_allclose(
        analytic_gradient.sum(axis=0),
        np.zeros(3),
        rtol=0.0,
        atol=3e-12,
    )


def test_uks_deepks_energy_descriptor_and_gradient_baseline():
    model = make_unrestricted_model()
    method, total_energy = run_uks_deepks(MOLECULE_COORDINATES, model)
    spin_density = method.make_rdm1()
    descriptor_values = method.make_eig(spin_density)
    gradient_method = method.nuc_grad_method()
    gradient_method.grid_response = True
    analytic_gradient = gradient_method.kernel()

    assert spin_density.shape == (2, 2, 2)
    assert descriptor_values.shape == (2, 1)
    assert analytic_gradient.shape == (2, 3)
    assert np.isfinite(total_energy)
    assert np.isfinite(descriptor_values).all()
    assert np.isfinite(analytic_gradient).all()
    np.testing.assert_allclose(
        total_energy,
        EXPECTED_UKS_ENERGY,
        rtol=0.0,
        atol=5e-10,
    )
    np.testing.assert_allclose(
        descriptor_values,
        EXPECTED_UKS_DESCRIPTOR_VALUES,
        rtol=0.0,
        atol=5e-10,
    )
    np.testing.assert_allclose(
        analytic_gradient,
        EXPECTED_UKS_GRADIENT,
        rtol=0.0,
        atol=5e-9,
    )

    finite_difference = finite_difference_first_atom(
        run_uks_deepks,
        MOLECULE_COORDINATES,
        model,
    )
    np.testing.assert_allclose(
        analytic_gradient[0],
        finite_difference,
        rtol=2e-7,
        atol=5e-8,
    )
    np.testing.assert_allclose(
        analytic_gradient.sum(axis=0),
        np.zeros(3),
        rtol=0.0,
        atol=3e-12,
    )


def test_restricted_deepks_gradient_scanner_matches_fresh_methods():
    model = make_deterministic_model()
    initial_method, _ = run_deepks(MOLECULE_COORDINATES, model)
    scanner = initial_method.nuc_grad_method().as_scanner()

    for displacement in (-0.06, 0.08):
        coordinates = MOLECULE_COORDINATES.copy()
        coordinates[1, 2] += displacement
        scanned_energy, scanned_gradient = scanner(make_molecule(coordinates))
        fresh_method, fresh_energy = run_deepks(coordinates, model)
        fresh_gradient = fresh_method.nuc_grad_method().kernel()

        np.testing.assert_allclose(scanned_energy, fresh_energy, rtol=0.0, atol=3e-10)
        np.testing.assert_allclose(
            scanned_gradient,
            fresh_gradient,
            rtol=0.0,
            atol=5e-8,
        )


def test_coulomb_penalty_scf_state_baseline():
    molecule = make_molecule(MOLECULE_COORDINATES)
    target_density = np.zeros((molecule.nao, molecule.nao))
    method = DSCF(
        molecule,
        None,
        proj_basis=PROJECTOR_BASIS,
        penalties=[CoulombPenalty(target_density, strength=0.1)],
    )
    method.conv_tol = 1e-12
    method.max_cycle = 100

    total_energy = method.kernel()

    assert method.converged
    np.testing.assert_allclose(
        total_energy,
        EXPECTED_PENALIZED_ENERGY,
        rtol=0.0,
        atol=3e-10,
    )
    np.testing.assert_allclose(
        method.make_rdm1(),
        EXPECTED_PENALIZED_DENSITY,
        rtol=0.0,
        atol=3e-9,
    )
