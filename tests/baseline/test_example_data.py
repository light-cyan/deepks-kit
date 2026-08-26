from pathlib import Path

import numpy as np

from deepks.data.fields import select_fields
from deepks.data.io import build_molecule, iter_system
from deepks.deepks.run import solve_molecule


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = REPOSITORY_ROOT / "examples"
EXAMPLE_SYSTEMS = {
    "water_single/systems/group.00": (300, 3, True),
    "water_single/systems/group.01": (300, 3, True),
    "water_single/systems/group.02": (300, 3, True),
    "water_single/systems/group.03": (100, 3, True),
    "water_cluster/systems/train.n1": (100, 3, False),
    "water_cluster/systems/train.n2": (100, 6, False),
    "water_cluster/systems/train.n3": (100, 9, False),
    "water_cluster/systems/valid.n4": (50, 12, False),
    "water_cluster/systems/test.n6": (21, 18, False),
}
EXPECTED_WATER_FORCE = np.array(
    [
        [0.0930768210858968, 0.0413643324237498, -0.0382744104686354],
        [-0.0628933656224590, -0.0290013274195663, 0.0244659322805099],
        [-0.0301834554634233, -0.0123630050041789, 0.0138084781881153],
    ]
)


def test_example_system_assets_have_consistent_frames_and_labels():
    total_frames = 0
    for relative_path, (expected_frames, expected_atoms, has_density) in EXAMPLE_SYSTEMS.items():
        system_path = EXAMPLE_ROOT / relative_path
        label_names = {"energy", "force"}
        if has_density:
            label_names.add("dm")

        frame_count = 0
        for atoms, attributes, labels in iter_system(str(system_path), label_names):
            frame_count += 1
            coordinates = np.asarray([atom[1] for atom in atoms], dtype=float)
            assert coordinates.shape == (expected_atoms, 3)
            assert np.isfinite(coordinates).all()
            assert np.asarray(labels["energy"]).shape == (1,)
            assert np.asarray(labels["force"]).shape == (expected_atoms, 3)
            assert np.isfinite(labels["energy"]).all()
            assert np.isfinite(labels["force"]).all()
            if has_density:
                density = np.asarray(labels["dm"])
                assert density.shape == (24, 24)
                assert np.isfinite(density).all()
            else:
                assert attributes["unit"] == "Angstrom"

        assert frame_count == expected_frames
        total_frames += frame_count

    assert total_frames == 1371


def test_water_example_first_frame_scf_and_force_snapshot():
    system_path = EXAMPLE_ROOT / "water_cluster" / "systems" / "train.n1"
    atoms, attributes, labels = next(
        iter_system(str(system_path), {"energy", "force"})
    )
    molecule = build_molecule(atoms, basis="ccpvdz", verbose=0, **attributes)
    fields = select_fields(
        [
            "e_base",
            "e_tot",
            "descriptor",
            "converged",
            "f_reference_variational",
            "f_tot",
            "dq_dR_explicit",
            "e_corr_target",
            "f_corr_explicit_target",
        ]
    )

    metadata, result = solve_molecule(
        molecule,
        None,
        fields,
        labels,
        conv_tol=1.0e-11,
        max_cycle=100,
    )

    np.testing.assert_array_equal(metadata, np.array([3, 3, 24, 108]))
    assert result["converged"]
    np.testing.assert_allclose(result["e_base"], -76.02290493759624, rtol=0.0, atol=2.0e-10)
    np.testing.assert_allclose(result["e_tot"], result["e_base"], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(result["f_reference_variational"], EXPECTED_WATER_FORCE, rtol=0.0, atol=2.0e-8)
    np.testing.assert_allclose(result["f_tot"], result["f_reference_variational"], rtol=0.0, atol=0.0)
    assert result["descriptor"].shape == (3, 108)
    assert result["dq_dR_explicit"].shape == (3, 3, 3, 108)
    np.testing.assert_allclose(np.linalg.norm(result["descriptor"]), 3.1483482127496454, rtol=0.0, atol=5.0e-9)
    np.testing.assert_allclose(np.linalg.norm(result["dq_dR_explicit"]), 4.283780712012324, rtol=0.0, atol=3.0e-8)
    np.testing.assert_allclose(result["f_reference_variational"].sum(axis=0), np.zeros(3), rtol=0.0, atol=3.0e-12)
    np.testing.assert_allclose(result["dq_dR_explicit"].sum(axis=0), np.zeros_like(result["dq_dR_explicit"][0]), rtol=0.0, atol=3.0e-12)
