import numpy as np
import pytest
from pyscf import gto

from deepks.descriptor import (
    AtomicDensityDescriptor,
    descriptor_atom_indices,
    is_ghost_atom,
)


COORDINATES = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
AO_DENSITY = np.array([[1.0, 0.25], [0.25, 0.4]])
PROJECTOR_BASIS = [[0, [1.0, 1.0]]]


def _make_ghost_molecule(ghost_label, coordinates=COORDINATES):
    return gto.M(
        atom=[("H", coordinates[0]), (ghost_label, coordinates[1])],
        basis="sto-3g",
        spin=1,
        unit="Bohr",
        verbose=0,
    )


def _finite_difference(ghost_label, evaluator, step=1.0e-5):
    molecule = _make_ghost_molecule(ghost_label)
    baseline = evaluator(AtomicDensityDescriptor(molecule, PROJECTOR_BASIS))
    result = np.empty((molecule.natm, 3, *baseline.shape))
    for atom_index in range(molecule.natm):
        for coordinate_index in range(3):
            forward_coordinates = COORDINATES.copy()
            backward_coordinates = COORDINATES.copy()
            forward_coordinates[atom_index, coordinate_index] += step
            backward_coordinates[atom_index, coordinate_index] -= step
            forward = evaluator(
                AtomicDensityDescriptor(
                    _make_ghost_molecule(ghost_label, forward_coordinates),
                    PROJECTOR_BASIS,
                )
            )
            backward = evaluator(
                AtomicDensityDescriptor(
                    _make_ghost_molecule(ghost_label, backward_coordinates),
                    PROJECTOR_BASIS,
                )
            )
            result[atom_index, coordinate_index] = (
                forward - backward
            ) / (2.0 * step)
    return result


def test_ghost_aliases_use_the_same_charge_based_descriptor_mapping():
    descriptors = []
    for ghost_label in ("X-H", "ghost-H"):
        molecule = _make_ghost_molecule(ghost_label)
        shared_descriptor = AtomicDensityDescriptor(molecule, PROJECTOR_BASIS)

        assert molecule.atom_charge(0) == 1
        assert molecule.atom_charge(1) == 0
        assert not is_ghost_atom(molecule, 0)
        assert is_ghost_atom(molecule, 1)
        assert descriptor_atom_indices(molecule) == (0,)
        assert shared_descriptor.descriptor_atom_indices == (0,)
        assert shared_descriptor.n_descriptor_atoms == 1
        assert shared_descriptor.projector_mol.natm == 1
        descriptors.append(shared_descriptor.descriptor(AO_DENSITY))

    np.testing.assert_allclose(descriptors[0], descriptors[1], rtol=0.0, atol=0.0)


@pytest.mark.parametrize("ghost_label", ["X-H", "ghost-H"])
def test_ghost_ao_motion_is_included_in_explicit_projected_density_derivative(
    ghost_label,
):
    molecule = _make_ghost_molecule(ghost_label)
    shared_descriptor = AtomicDensityDescriptor(molecule, PROJECTOR_BASIS)
    analytic = shared_descriptor.dD_dR_explicit(AO_DENSITY, flatten=True)
    finite_difference = _finite_difference(
        ghost_label,
        lambda descriptor: descriptor.projected_density(AO_DENSITY, flatten=True),
    )

    assert analytic.shape == (2, 3, 1, 1)
    assert np.linalg.norm(analytic[1]) > 1.0e-2
    np.testing.assert_allclose(analytic, finite_difference, rtol=1.0e-8, atol=3.0e-9)
    np.testing.assert_allclose(
        analytic.sum(axis=0),
        np.zeros_like(analytic[0]),
        rtol=0.0,
        atol=2.0e-12,
    )


@pytest.mark.parametrize("ghost_label", ["X-H", "ghost-H"])
def test_ghost_ao_motion_is_included_in_explicit_descriptor_derivative(
    ghost_label,
):
    molecule = _make_ghost_molecule(ghost_label)
    shared_descriptor = AtomicDensityDescriptor(molecule, PROJECTOR_BASIS)
    analytic = shared_descriptor.dq_dR_explicit(AO_DENSITY)
    finite_difference = _finite_difference(
        ghost_label,
        lambda descriptor: descriptor.descriptor(AO_DENSITY),
    )

    assert analytic.shape == (2, 3, 1, 1)
    assert np.linalg.norm(analytic[1]) > 1.0e-2
    np.testing.assert_allclose(analytic, finite_difference, rtol=1.0e-8, atol=3.0e-9)
    np.testing.assert_allclose(
        analytic.sum(axis=0),
        np.zeros_like(analytic[0]),
        rtol=0.0,
        atol=2.0e-12,
    )
