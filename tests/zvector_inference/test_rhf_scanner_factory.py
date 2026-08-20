from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
from pyscf import gto
from pyscf.scf import hf as scf_hf

import deepks.deephf.pyscf_rhf as pyscf_rhf
from deepks.deephf.pyscf_rhf import (
    RHFScannerReferenceError,
    RHFScannerReferenceFactory,
)


def _sealed_root(root, occupied_coefficients):
    forged = replace(
        root,
        integrity_fingerprint="",
        occupied_coefficients=pyscf_rhf._immutable_array(
            np.asarray(occupied_coefficients, dtype=np.float64)
        ),
    )
    return replace(
        forged,
        integrity_fingerprint=pyscf_rhf._root_integrity_fingerprint(forged),
    )


def test_factory_builds_fresh_native_rhf_without_a_warm_start(
    zvector_algebra_case,
    monkeypatch,
):
    reference = zvector_algebra_case.reference
    factory = RHFScannerReferenceFactory(reference)
    initial_root = factory.initial_root
    calls = []
    original_kernel = scf_hf.RHF.kernel

    def observed_kernel(self, dm0=None, **kwargs):
        calls.append((id(self), dm0))
        return original_kernel(self, dm0=dm0, **kwargs)

    monkeypatch.setattr(scf_hf.RHF, "kernel", observed_kernel)
    coordinates = zvector_algebra_case.coordinates.copy()
    coordinates[1, 0] += 2.0e-3
    fresh, candidate_root = factory.build(coordinates, initial_root)

    assert type(fresh) is scf_hf.RHF
    assert fresh is not reference
    assert fresh.mol is not reference.mol
    assert calls == [(id(fresh), None)]
    assert fresh.chkfile is None
    assert fresh.callback is None
    assert fresh.diis_file is None
    assert fresh.converged
    for name, value in factory.scf_controls.items():
        assert getattr(fresh, name) == value
    assert candidate_root.parent_state_fingerprint == initial_root.state_fingerprint
    assert candidate_root.minimum_occupied_overlap > 0.99
    assert not candidate_root.occupied_coefficients.flags.writeable
    assert not candidate_root.occupations.flags.writeable
    assert initial_root.integrity_fingerprint == pyscf_rhf._root_integrity_fingerprint(
        initial_root
    )


def test_factory_accepts_only_coordinates_or_a_static_compatible_molecule(
    zvector_algebra_case,
):
    reference = zvector_algebra_case.reference
    factory = RHFScannerReferenceFactory(reference)
    initial_root = factory.initial_root
    compatible = deepcopy(reference.mol)
    coordinates = zvector_algebra_case.coordinates.copy()
    coordinates[2, 1] -= 1.0e-3
    compatible.set_geom_(coordinates, unit="Bohr", inplace=True)

    from_float32, _ = factory.build(
        coordinates.astype(np.float32),
        initial_root,
    )
    from_molecule, _ = factory.build(compatible, initial_root)

    np.testing.assert_allclose(
        from_float32.mol.atom_coords(unit="Bohr"),
        coordinates.astype(np.float32).astype(np.float64),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        from_molecule.mol.atom_coords(unit="Bohr"),
        coordinates,
        rtol=0.0,
        atol=0.0,
    )
    with pytest.raises(TypeError, match="real integer or floating"):
        factory.build(coordinates.astype(np.complex128), initial_root)
    with pytest.raises(TypeError, match="real integer or floating"):
        factory.build(coordinates.astype(object), initial_root)
    with pytest.raises(ValueError, match="coordinates have shape"):
        factory.build(coordinates[:-1], initial_root)

    incompatible = gto.M(
        atom=list(zip(("O", "H", "H"), coordinates)),
        basis="6-31g",
        unit="Bohr",
        symmetry=False,
        cart=False,
        verbose=0,
    )
    with pytest.raises(RHFScannerReferenceError, match="static metadata"):
        factory.build(incompatible, initial_root)


def test_root_overlap_is_invariant_to_signs_and_occupied_rotations(
    zvector_algebra_case,
):
    reference = zvector_algebra_case.reference
    factory = RHFScannerReferenceFactory(reference)
    root = factory.initial_root
    generator = np.random.default_rng(20260820)
    rotation, _ = np.linalg.qr(
        generator.normal(
            size=(
                root.occupied_coefficients.shape[1],
                root.occupied_coefficients.shape[1],
            )
        )
    )
    rotation[:, 0] *= -1.0
    rotated_root = _sealed_root(
        root,
        root.occupied_coefficients @ rotation,
    )

    _, candidate = factory.build(
        zvector_algebra_case.coordinates,
        rotated_root,
    )

    assert candidate.minimum_occupied_overlap > 1.0 - 1.0e-12


def test_root_overlap_rejects_an_occupied_virtual_subspace_swap(
    zvector_algebra_case,
):
    reference = zvector_algebra_case.reference
    factory = RHFScannerReferenceFactory(reference)
    root = factory.initial_root
    occupations = np.asarray(reference.mo_occ)
    virtual_coefficients = np.asarray(reference.mo_coeff)[:, occupations == 0]
    swapped = root.occupied_coefficients.copy()
    swapped[:, 0] = virtual_coefficients[:, 0]
    swapped_root = _sealed_root(root, swapped)

    with pytest.raises(RHFScannerReferenceError, match="discontinuous"):
        factory.build(zvector_algebra_case.coordinates, swapped_root)


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        pytest.param("diis", object(), "diis must be boolean", id="custom-diis"),
        pytest.param(
            "DIIS",
            object(),
            "custom DIIS implementation",
            id="custom-diis-implementation",
        ),
        pytest.param("init_guess", "chkfile", "must not use a checkpoint", id="checkpoint"),
        pytest.param("damp", 1.0, "damp must be below", id="damping"),
    ],
)
def test_factory_rejects_unsafe_scf_controls(
    zvector_algebra_case,
    name,
    value,
    message,
):
    reference = zvector_algebra_case.reference
    had_instance_value = name in reference.__dict__
    original = getattr(reference, name)
    setattr(reference, name, value)
    try:
        with pytest.raises(RHFScannerReferenceError, match=message):
            RHFScannerReferenceFactory(reference)
    finally:
        if had_instance_value:
            setattr(reference, name, original)
        else:
            reference.__dict__.pop(name, None)
