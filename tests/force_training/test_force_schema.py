import copy
from pathlib import Path

import numpy as np

from deepks.data.force_schema import (
    CANONICAL_FORCE_FIELDS,
    ForceDataContract,
    force_checkpoint_metadata,
    load_force_dataset,
    validate_force_checkpoint_metadata,
    write_force_dataset,
)


def make_schema_inputs(frame_count=2):
    raw_atoms = descriptor_atoms = 2
    features = 3
    atom = np.zeros((frame_count, raw_atoms, 4), dtype=np.float64)
    atom[:, :, 0] = np.array([1.0, 1.0])
    atom[:, :, 1:] = np.arange(
        frame_count * raw_atoms * 3,
        dtype=np.float64,
    ).reshape(frame_count, raw_atoms, 3) / 10.0
    descriptor = np.arange(
        frame_count * descriptor_atoms * features,
        dtype=np.float64,
    ).reshape(frame_count, descriptor_atoms, features) / 100.0
    e_base = -np.arange(1, frame_count + 1, dtype=np.float64).reshape(-1, 1)
    e_target = e_base + 0.125
    f_base = np.arange(
        frame_count * raw_atoms * 3,
        dtype=np.float64,
    ).reshape(frame_count, raw_atoms, 3) / 1000.0
    f_target = f_base + 0.025
    arrays = {
        "atom": atom,
        "descriptor": descriptor,
        "e_base": e_base,
        "f_base": f_base,
        "e_target": e_target,
        "f_target": f_target,
        "e_corr_target": e_target - e_base,
        "f_corr_target": f_target - f_base,
        "dq_dR_relaxed": np.arange(
            frame_count * raw_atoms * 3 * descriptor_atoms * features,
            dtype=np.float64,
        ).reshape(frame_count, raw_atoms, 3, descriptor_atoms, features)
        / 10000.0,
    }
    provenance = {
        "atom_mapping": {
            "descriptor_to_raw": [0, 1],
            "raw_to_descriptor": [0, 1],
            "nuclear_charges": [1, 1],
            "ghost_policy": "rejected",
        },
        "descriptor": {
            "definition": "ordered_projected_density_eigenvalues",
            "spin_semantics": "spin_summed",
            "shell_sizes": [1, 2],
            "projector_basis": [[0, [0.8, 1.0]], [1, [0.3, 1.0]]],
            "differentiability_controls": {
                "gap_atol": 1.0e-9,
                "gap_rtol": 1.0e-7,
            },
        },
        "reference": {
            "family": "RHF",
            "python_class": "pyscf.scf.hf.RHF",
            "basis_content": {"H": [[0, [1.24, 1.0]]]},
            "ecp": None,
            "charge": 0,
            "spin": 0,
            "occupations": [2.0, 0.0],
            "scf_controls": {"conv_tol": 1.0e-13, "max_cycle": 100},
        },
        "response": {
            "backend": "rhf_direct",
            "adapter": "deepks.deephf.pyscf_rhf.RHFResponseAdapter",
            "controls": {"residual_tolerance": 1.0e-9},
        },
        "frames": [
            {
                "reference_state_fingerprint": f"{index + 1:064x}",
                "reference_converged": True,
                "response_converged": True,
                "response_diagnostics": {
                    "maximum_residual": 1.0e-11 + index * 1.0e-12,
                    "residual_tolerance": 1.0e-9,
                },
                "descriptor_diagnostics": {
                    "minimum_scaled_gap": 10.0 + index,
                    "structural_zero_blocks": [],
                },
            }
            for index in range(frame_count)
        ],
        "generation": {
            "deepks_version": "0.1.dev-test",
            "deepks_commit": None,
            "pyscf_version": "2.14.0",
            "torch_version": "2.test",
            "numpy_version": np.__version__,
            "python_version": "3.11.test",
            "producer": "deepks.deephf.force_data.rhf_direct",
        },
    }
    return arrays, provenance


def test_force_schema_round_trip_is_complete_and_canonical(tmp_path):
    arrays, provenance = make_schema_inputs()
    directory = tmp_path / "force-data"

    written = write_force_dataset(
        directory,
        arrays=arrays,
        provenance=provenance,
    )
    loaded, actual = load_force_dataset(directory)

    assert isinstance(written, ForceDataContract)
    assert written.manifest_fingerprint == loaded.manifest_fingerprint
    assert written.compatibility_fingerprint == loaded.compatibility_fingerprint
    assert written.force_contract_fingerprint == written.compatibility_fingerprint
    assert written.jacobian_semantics == "dq_dR_relaxed"
    assert written.dimensions == {
        "n_frame": 2,
        "n_raw_atom": 2,
        "n_descriptor_atom": 2,
        "n_feature": 3,
    }
    assert {path.name for path in directory.iterdir()} == {
        "force_data.json",
        *(f"{name}.npy" for name in CANONICAL_FORCE_FIELDS),
    }
    for name in CANONICAL_FORCE_FIELDS:
        np.testing.assert_array_equal(actual[name], arrays[name])

    manifest = loaded.manifest
    assert manifest["schema"] == {
        "id": "deepks.deephf.rhf-force-data",
        "version": 1,
    }
    assert manifest["conventions"] == {
        "cartesian_order": ["x", "y", "z"],
        "energy_unit": "Eh",
        "force_sign": "force=-dE/dR",
        "force_unit": "Eh/Bohr",
        "jacobian_name": "dq_dR_relaxed",
        "jacobian_semantics": "complete_relaxed_reference_response",
        "jacobian_sign": "+dq/dR",
        "jacobian_unit": "Bohr^-1",
        "length_unit": "Bohr",
    }
    jacobian = manifest["fields"]["dq_dR_relaxed"]
    assert jacobian["axes"] == [
        "frame",
        "raw_atom",
        "cartesian",
        "descriptor_atom",
        "feature",
    ]
    assert jacobian["shape"] == [2, 2, 3, 2, 3]
    assert jacobian["dtype"] == "float64"
    assert jacobian["unit"] == "Bohr^-1"
    assert jacobian["sign"] == "+dq/dR"
    assert jacobian["semantics"] == "complete_relaxed_reference_response"
    assert len(jacobian["sha256"]) == 64
    assert len(manifest["descriptor"]["projector_sha256"]) == 64
    assert len(manifest["reference"]["basis_sha256"]) == 64
    assert all(len(frame["sample_id"]) == 64 for frame in manifest["frames"])

    mutable_copy = loaded.manifest
    mutable_copy["dimensions"]["n_frame"] = 99
    assert loaded.dimensions["n_frame"] == 2


def test_force_schema_output_and_fingerprints_are_deterministic(tmp_path):
    arrays, provenance = make_schema_inputs()
    first = write_force_dataset(
        tmp_path / "first",
        arrays=arrays,
        provenance=provenance,
    )
    second = write_force_dataset(
        tmp_path / "second",
        arrays=copy.deepcopy(arrays),
        provenance=copy.deepcopy(provenance),
    )

    assert first.manifest_fingerprint == second.manifest_fingerprint
    assert first.compatibility_fingerprint == second.compatibility_fingerprint
    assert (tmp_path / "first" / "force_data.json").read_bytes() == (
        tmp_path / "second" / "force_data.json"
    ).read_bytes()


def test_force_checkpoint_metadata_round_trip_and_ignores_system_size(tmp_path):
    arrays, provenance = make_schema_inputs(frame_count=2)
    contract = write_force_dataset(
        tmp_path / "two-frames",
        arrays=arrays,
        provenance=provenance,
    )
    metadata = force_checkpoint_metadata(contract)

    validate_force_checkpoint_metadata(metadata, contract)
    assert metadata["jacobian_semantics"] == "dq_dR_relaxed"
    assert metadata["n_feature"] == 3
    assert metadata["projector_sha256"] == contract.manifest["descriptor"][
        "projector_sha256"
    ]

    single_arrays, single_provenance = make_schema_inputs(frame_count=1)
    single_contract = write_force_dataset(
        tmp_path / "one-frame",
        arrays=single_arrays,
        provenance=single_provenance,
    )
    assert contract.compatibility_fingerprint == single_contract.compatibility_fingerprint
    assert force_checkpoint_metadata(contract) == force_checkpoint_metadata(single_contract)


def test_contract_accepts_a_path_object(tmp_path):
    arrays, provenance = make_schema_inputs(frame_count=1)
    directory = Path(tmp_path) / "path-object"
    write_force_dataset(directory, arrays=arrays, provenance=provenance)

    contract, _ = load_force_dataset(directory)
    assert contract.dimensions["n_frame"] == 1
