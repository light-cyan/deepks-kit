from pathlib import Path

import numpy as np
import pytest
import torch
from ase import Atoms, units
from ase.md.velocitydistribution import Stationary, thermalize_momenta
from ase.md.verlet import VelocityVerlet
from pyscf import gto

from deepks.deephf import (
    DeePHFCalculator,
    GPUDeePHF,
    GPUUHFDeePHF,
    build_reference,
    make_deephf,
)
from deepks.deephf.gpu_method import gpu_reference_family
from deepks.gpu import require_cuda_device
from deepks.model.model import CorrNet


PROJECTOR_BASIS = [[0, [0.8, 1.0]]]


def _model():
    model = CorrNet(
        input_dim=1,
        hidden_sizes=(2,),
        actv_fn="tanh",
        use_resnet=False,
        proj_basis=PROJECTOR_BASIS,
    ).double().eval()
    generator = torch.Generator().manual_seed(17)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.uniform_(-0.05, 0.05, generator=generator)
    return model


def _h2(distance):
    return gto.M(
        atom=f"H 0 0 0; H 0 0 {distance}",
        basis="sto-3g",
        unit="Bohr",
        spin=0,
        symmetry=False,
        cart=False,
        verbose=0,
    )


def _oh(distance):
    return gto.M(
        atom=f"O 0 0 0; H 0 0 {distance}",
        basis="sto-3g",
        unit="Bohr",
        spin=1,
        symmetry=False,
        cart=False,
        verbose=0,
    )


def _energy(molecule, family, model):
    reference = build_reference(
        molecule,
        family,
        scf_args={"conv_tol_grad": 1.0e-7}
        if family in {"rks", "uks"}
        else None,
    )
    method = make_deephf(
        reference,
        model,
        projector_basis=PROJECTOR_BASIS,
    )
    return method.kernel()


def _correction_energy(molecule, family, model, *, dm0=None):
    reference = build_reference(
        molecule,
        family,
        scf_args={"conv_tol_grad": 1.0e-7},
        dm0=dm0,
    )
    method = make_deephf(
        reference,
        model,
        projector_basis=PROJECTOR_BASIS,
    )
    return method.kernel() - float(reference.e_tot)


def _finite_difference(molecule_factory, family, model, coordinate, step=1.0e-4):
    plus = _energy(molecule_factory(coordinate + step), family, model)
    minus = _energy(molecule_factory(coordinate - step), family, model)
    return (plus - minus) / (2.0 * step)


def test_workflow_keeps_exact_gpu_references_without_cpu_conversion():
    require_cuda_device()
    restricted = build_reference(_h2(1.4), "rhf")
    unrestricted = build_reference(_oh(1.8), "uhf")

    assert gpu_reference_family(restricted) == "rhf"
    assert gpu_reference_family(unrestricted) == "uhf"
    assert type(make_deephf(restricted, None, projector_basis=PROJECTOR_BASIS)) is GPUDeePHF
    assert type(make_deephf(unrestricted, None, projector_basis=PROJECTOR_BASIS)) is GPUUHFDeePHF
    source = (Path(__file__).parents[2] / "deepks/deephf/workflow.py").read_text()
    assert ".to_cpu()" not in source


def test_rhf_gpu_analytic_gradient_matches_total_energy_finite_difference():
    require_cuda_device()
    model = _model()
    method = make_deephf(
        build_reference(_h2(1.4), "rhf"),
        model,
        projector_basis=PROJECTOR_BASIS,
    )
    assert next(method.model.parameters()).device.type == "cuda"
    analytic = method.gradient()[1, 2]
    numerical = _finite_difference(_h2, "rhf", model, 1.4)

    np.testing.assert_allclose(analytic, numerical, rtol=0.0, atol=2.0e-6)
    assert method.operation_counts["gpu_direct_response_solves"] == 1


def test_uhf_gpu_analytic_gradient_matches_total_energy_finite_difference():
    require_cuda_device()
    model = _model()
    method = make_deephf(
        build_reference(_oh(1.8), "uhf"),
        model,
        projector_basis=PROJECTOR_BASIS,
    )
    analytic = method.gradient()[1, 2]
    numerical = _finite_difference(_oh, "uhf", model, 1.8)

    np.testing.assert_allclose(analytic, numerical, rtol=0.0, atol=3.0e-6)
    assert method.operation_counts["gpu_direct_response_solves"] == 1


@pytest.mark.parametrize(
    ("family", "molecule_factory", "distance"),
    (("rks", _h2, 1.4), ("uks", _oh, 1.8)),
)
def test_dft_gpu_correction_gradient_matches_correction_energy_finite_difference(
    family,
    molecule_factory,
    distance,
):
    require_cuda_device()
    model = _model()
    reference = build_reference(
        molecule_factory(distance),
        family,
        scf_args={"conv_tol_grad": 1.0e-7},
    )
    method = make_deephf(
        reference,
        model,
        projector_basis=PROJECTOR_BASIS,
    )
    analytic = (
        method.gradient()[1, 2]
        - reference.nuc_grad_method().kernel()[1, 2]
    )
    step = 1.0e-4
    density = reference.make_rdm1().copy()
    numerical = (
        _correction_energy(
            molecule_factory(distance + step), family, model, dm0=density
        )
        - _correction_energy(
            molecule_factory(distance - step), family, model, dm0=density
        )
    ) / (2.0 * step)

    np.testing.assert_allclose(analytic, numerical, rtol=0.0, atol=3.0e-6)
    assert method.operation_counts["gpu_direct_response_solves"] == 1


def test_b3lyp_gpu_analytic_gradient_matches_total_energy_finite_difference():
    require_cuda_device()
    model = _model()
    dft_args = {
        "xc": "B3LYP5",
        "grid_mode": "default",
        "grid_level": 3,
        "small_rho_cutoff": 0.0,
    }

    def evaluate(distance, *, gradient=False):
        method = make_deephf(
            build_reference(
                _h2(distance),
                "rks",
                scf_args={"conv_tol_grad": 1.0e-7},
                dft_args=dft_args,
            ),
            model,
            projector_basis=PROJECTOR_BASIS,
        )
        return method.gradient()[1, 2] if gradient else method.kernel()

    analytic = evaluate(1.4, gradient=True)
    step = 1.0e-4
    numerical = (evaluate(1.4 + step) - evaluate(1.4 - step)) / (2.0 * step)

    np.testing.assert_allclose(analytic, numerical, rtol=0.0, atol=5.0e-6)


def test_uhf_gradient_scanner_reuses_density_and_tracks_both_spin_roots():
    require_cuda_device()
    method = make_deephf(
        build_reference(_oh(1.8), "uhf"),
        None,
        projector_basis=PROJECTOR_BASIS,
    )
    scanner = method.nuc_grad_method().as_scanner()
    first_energy, first_gradient = scanner(_oh(1.8))
    second_energy, second_gradient = scanner(_oh(1.801))

    assert np.isfinite((first_energy, second_energy)).all()
    assert np.isfinite(first_gradient).all()
    assert np.isfinite(second_gradient).all()
    assert scanner.records[0]["initial_guess_source"] == "existing_reference"
    assert scanner.records[1]["initial_guess_source"] == "previous_density"
    assert set(scanner.records[1]["occupied_subspace_overlaps"]) == {"alpha", "beta"}
    assert scanner.records[1]["minimum_occupied_overlap"] > 0.99


def test_ase_calculator_converts_energy_and_force_units():
    require_cuda_device()
    molecule = _h2(1.4)
    method = make_deephf(
        build_reference(molecule, "rhf"),
        None,
        projector_basis=PROJECTOR_BASIS,
    )
    expected_energy = method.kernel()
    expected_force = -method.gradient()
    atoms = Atoms("H2", positions=molecule.atom_coords(unit="Angstrom"))
    atoms.calc = DeePHFCalculator(method)

    np.testing.assert_allclose(
        atoms.get_potential_energy(), expected_energy * units.Hartree, atol=1.0e-10
    )
    np.testing.assert_allclose(
        atoms.get_forces(), expected_force * units.Hartree / units.Bohr, atol=1.0e-8
    )
    assert len(atoms.calc.scanner.records) == 1


def test_short_ase_nve_trajectory_has_bounded_total_energy_error():
    require_cuda_device()
    molecule = _h2(1.4)
    method = make_deephf(
        build_reference(molecule, "rhf"),
        None,
        projector_basis=PROJECTOR_BASIS,
    )
    atoms = Atoms("H2", positions=molecule.atom_coords(unit="Angstrom"))
    atoms.calc = DeePHFCalculator(method)
    thermalize_momenta(
        atoms,
        temperature_K=50.0,
        rng=np.random.default_rng(9),
    )
    Stationary(atoms)
    dynamics = VelocityVerlet(atoms, timestep=0.05 * units.fs)
    totals = []

    def record():
        totals.append(atoms.get_potential_energy() + atoms.get_kinetic_energy())

    dynamics.attach(record, interval=1)
    dynamics.run(5)
    drift = np.asarray(totals) - totals[0]

    assert len(totals) == 6
    assert np.max(np.abs(drift)) < 1.0e-4
    assert len(atoms.calc.scanner.records) == 6
