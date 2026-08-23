import copy
from pathlib import Path

import numpy as np

from deepks.data.force_schema import (
    CANONICAL_FORCE_FIELDS,
    ForceDataContract,
    force_checkpoint_metadata,
    load_force_dataset,
    validate_force_checkpoint_metadata,
    _write_force_dataset as write_force_dataset,
)


def make_schema_inputs(frame_count=2):
    raw_atoms = descriptor_atoms = 2
    features = 4
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
            "shell_sizes": [1, 3],
            "projector_basis": [[0, [0.8, 1.0]], [1, [0.3, 1.0]]],
            "differentiability_controls": {
                "gap_atol": 1.0e-9,
                "gap_rtol": 1.0e-7,
                "zero_atol": 1.0e-9,
                "sensitivity_atol": 1.0e-8,
                "structural_zero_blocks": "rejected",
                "validator": "deepks.descriptor.validate_differentiability",
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
            "scf_controls": {
                "conv_tol": 1.0e-13,
                "conv_tol_grad": None,
                "conv_tol_cpscf": 1.0e-9,
                "max_cycle": 100,
                "level_shift": 0.0,
                "diis_space": 8,
                "direct_scf": True,
                "conv_check": True,
            },
        },
        "response": {
            "backend": "rhf_direct",
            "adapter": "deepks.deephf.pyscf_rhf.RHFResponseAdapter",
            "controls": {
                "cphf_tolerance": 1.0e-12,
                "residual_tolerance": 1.0e-9,
                "invariant_tolerance": 1.0e-9,
                "orbital_gap_tolerance": 1.0e-8,
                "max_cycle": 50,
                "max_refinement_cycles": 3,
                "level_shift": 0.0,
                "operator_stability_tolerance": 1.0e-6,
                "operator_condition_tolerance": 1.0e8,
                "operator_symmetry_tolerance": 1.0e-10,
                "operator_dimension_limit": 512,
            },
        },
        "frames": [
            {
                "reference_state_fingerprint": f"{index + 1:064x}",
                "reference_converged": True,
                "response_converged": True,
                "response_integrity_fingerprint": f"{index + 101:064x}",
                "response_diagnostics": {
                    "minimum_orbital_gap": 0.5,
                    "pyscf_version": "2.14.0",
                    "cphf_tolerance": 1.0e-12,
                    "maximum_residual": 1.0e-11 + index * 1.0e-12,
                    "residual_rms": 5.0e-12 + index * 5.0e-13,
                    "residual_tolerance": 1.0e-9,
                    "invariant_tolerance": 1.0e-9,
                    "orbital_gap_tolerance": 1.0e-8,
                    "max_cycle": 50,
                    "max_refinement_cycles": 3,
                    "level_shift": 0.0,
                    "response_dimension": 1,
                    "operator_stability_tolerance": 1.0e-6,
                    "operator_condition_tolerance": 1.0e8,
                    "operator_symmetry_tolerance": 1.0e-10,
                    "operator_dimension_limit": 512,
                    "operator_diagnostics_are_estimates": True,
                    "operator_minimum_eigenvalue": 0.25,
                    "operator_maximum_eigenvalue": 2.0,
                    "operator_condition_number": 8.0,
                    "operator_symmetry_residual": 1.0e-15,
                    "metric_residual": 1.0e-14,
                    "idempotency_residual": 1.0e-14,
                    "particle_number_residual": 1.0e-14,
                    "refinement_cycles": 0,
                    "residual_history": [1.0e-11 + index * 1.0e-12],
                },
                "descriptor_diagnostics": {
                    "minimum_scaled_gap": 10.0 + index,
                    "structural_zero_blocks": [],
                },
                "geometry_bohr": atom[index, :, 1:].tolist(),
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
            "producer_version": 1,
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
        "n_feature": 4,
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
    assert jacobian["shape"] == [2, 2, 3, 2, 4]
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


def test_coordinate_chunking_does_not_change_scientific_compatibility(tmp_path):
    arrays, provenance = make_schema_inputs(frame_count=1)
    one_atom_blocks = copy.deepcopy(provenance)
    one_atom_blocks["response"].update(
        coordinate_block_size=1,
        response_block_count=2,
    )
    two_atom_block = copy.deepcopy(provenance)
    two_atom_block["response"].update(
        coordinate_block_size=2,
        response_block_count=1,
    )

    first = write_force_dataset(
        tmp_path / "one-atom-blocks",
        arrays=copy.deepcopy(arrays),
        provenance=one_atom_blocks,
    )
    second = write_force_dataset(
        tmp_path / "two-atom-block",
        arrays=copy.deepcopy(arrays),
        provenance=two_atom_block,
    )

    assert first.compatibility_fingerprint == second.compatibility_fingerprint
    assert first.manifest_fingerprint != second.manifest_fingerprint


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
    assert metadata["n_feature"] == 4
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


def test_compatibility_fingerprint_binds_scientific_controls(tmp_path):
    arrays, provenance = make_schema_inputs(frame_count=1)
    baseline = write_force_dataset(
        tmp_path / "baseline",
        arrays=arrays,
        provenance=provenance,
    )
    mutations = {
        "basis": lambda value: value["reference"].update(
            basis_content={"H": [[0, [1.35, 1.0]]]}
        ),
        "scf": lambda value: value["reference"]["scf_controls"].update(
            conv_tol=2.0e-13
        ),
        "response": lambda value: (
            value["response"]["controls"].update(residual_tolerance=2.0e-9),
            value["frames"][0]["response_diagnostics"].update(
                residual_tolerance=2.0e-9
            ),
        ),
        "differentiability": lambda value: value["descriptor"][
            "differentiability_controls"
        ].update(gap_atol=2.0e-9),
        "generation": lambda value: value["generation"].update(
            deepks_version="0.2.dev-test"
        ),
    }
    for name, mutate in mutations.items():
        changed_arrays = copy.deepcopy(arrays)
        changed_provenance = copy.deepcopy(provenance)
        changed_provenance["reference"].pop("basis_sha256", None)
        mutate(changed_provenance)
        changed = write_force_dataset(
            tmp_path / name,
            arrays=changed_arrays,
            provenance=changed_provenance,
        )
        assert changed.compatibility_fingerprint != baseline.compatibility_fingerprint


def test_compatible_contracts_may_have_different_atom_counts(tmp_path):
    arrays, provenance = make_schema_inputs(frame_count=1)
    two_atom = write_force_dataset(
        tmp_path / "two-atom",
        arrays=arrays,
        provenance=provenance,
    )

    four_arrays = copy.deepcopy(arrays)
    four_arrays["atom"] = np.zeros((1, 4, 4), dtype=np.float64)
    four_arrays["atom"][..., 0] = 1.0
    four_arrays["atom"][..., 1:] = np.arange(12, dtype=np.float64).reshape(1, 4, 3) / 10.0
    four_arrays["descriptor"] = np.arange(16, dtype=np.float64).reshape(1, 4, 4) / 100.0
    for name in ("f_base", "f_target", "f_corr_target"):
        four_arrays[name] = np.zeros((1, 4, 3), dtype=np.float64)
    four_arrays["dq_dR_relaxed"] = np.arange(
        1 * 4 * 3 * 4 * 4,
        dtype=np.float64,
    ).reshape(1, 4, 3, 4, 4) / 10000.0
    four_provenance = copy.deepcopy(provenance)
    four_provenance["atom_mapping"] = {
        "descriptor_to_raw": [0, 1, 2, 3],
        "raw_to_descriptor": [0, 1, 2, 3],
        "nuclear_charges": [1, 1, 1, 1],
        "ghost_policy": "rejected",
    }
    four_provenance["reference"]["occupations"] = [2.0, 2.0, 0.0]
    four_provenance["frames"][0]["response_diagnostics"]["response_dimension"] = 2
    four_provenance["frames"][0]["geometry_bohr"] = four_arrays["atom"][0, :, 1:].tolist()
    four_atom = write_force_dataset(
        tmp_path / "four-atom",
        arrays=four_arrays,
        provenance=four_provenance,
    )

    assert four_atom.compatibility_fingerprint == two_atom.compatibility_fingerprint
    assert four_atom.manifest["frames"][0]["sample_id"] != (
        two_atom.manifest["frames"][0]["sample_id"]
    )
