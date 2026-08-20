import copy

import numpy as np
import pytest
import torch

from deepks.data.force_schema import ForceDataError, write_force_dataset
from deepks.data.stats import concat_data, make_label
from deepks.model.reader import GroupReader, Reader
from test_force_schema import make_schema_inputs


def _write_schema_dataset(path, *, frame_count=2, projector_basis=None):
    arrays, provenance = make_schema_inputs(frame_count=frame_count)
    if projector_basis is not None:
        provenance = copy.deepcopy(provenance)
        provenance["descriptor"]["projector_basis"] = projector_basis
    contract = write_force_dataset(path, arrays=arrays, provenance=provenance)
    return contract, arrays


def test_reader_exposes_only_canonical_relaxed_force_training_fields(tmp_path):
    directory = tmp_path / "force-data"
    contract, arrays = _write_schema_dataset(directory)

    reader = Reader(
        directory,
        batch_size=1,
        force_mode="deephf_relaxed",
    )
    sample = reader.sample_all()

    assert reader.force_contract.compatibility_fingerprint == (
        contract.compatibility_fingerprint
    )
    assert set(sample) == {
        "energy",
        "descriptor",
        "force",
        "dq_dR_relaxed",
        "force_contract_fingerprint",
    }
    torch.testing.assert_close(sample["energy"], torch.from_numpy(arrays["e_corr_target"]))
    torch.testing.assert_close(sample["descriptor"], torch.from_numpy(arrays["descriptor"]))
    torch.testing.assert_close(sample["force"], torch.from_numpy(arrays["f_corr_target"]))
    torch.testing.assert_close(
        sample["dq_dR_relaxed"],
        torch.from_numpy(arrays["dq_dR_relaxed"]),
    )
    expected_marker = torch.tensor(
        list(bytes.fromhex(contract.compatibility_fingerprint)),
        dtype=torch.uint8,
    ).expand(arrays["descriptor"].shape[0], -1)
    torch.testing.assert_close(sample["force_contract_fingerprint"], expected_marker)


def test_reader_rejects_explicit_aliases_and_missing_strict_manifest(tmp_path):
    directory = tmp_path / "force-data"
    _write_schema_dataset(directory, frame_count=1)

    with pytest.raises(ForceDataError, match="canonical f_corr_target"):
        Reader(
            directory,
            batch_size=1,
            force_mode="deephf_relaxed",
            force_name="f_corr_explicit_target",
        )
    with pytest.raises(ForceDataError, match="canonical dq_dR_relaxed"):
        Reader(
            directory,
            batch_size=1,
            force_mode="deephf_relaxed",
            jacobian_name="dq_dR_explicit",
        )
    with pytest.raises(ForceDataError, match="force_mode='deephf_relaxed'"):
        Reader(
            directory,
            batch_size=1,
            force_name="f_corr_target",
        )

    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    np.save(incomplete / "e_corr_target.npy", np.zeros((1, 1), dtype=np.float64))
    np.save(incomplete / "descriptor.npy", np.zeros((1, 2, 3), dtype=np.float64))
    with pytest.raises(ForceDataError, match="cannot read force_data.json"):
        Reader(incomplete, batch_size=1, force_mode="deephf_relaxed")


def test_group_reader_requires_one_compatible_force_contract(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    foreign = tmp_path / "foreign"
    contract, _ = _write_schema_dataset(first, frame_count=1)
    _write_schema_dataset(second, frame_count=2)
    _write_schema_dataset(
        foreign,
        frame_count=1,
        projector_basis=[[0, [1.1, 1.0]], [1, [0.45, 1.0]]],
    )

    grouped = GroupReader(
        [first, second],
        batch_size=1,
        force_mode="deephf_relaxed",
    )
    assert grouped.force_contract.compatibility_fingerprint == (
        contract.compatibility_fingerprint
    )
    assert grouped.get_train_size() == 3
    assert sum(batch["energy"].shape[0] for batch in grouped.sample_all_batch()) == 3

    with pytest.raises(ForceDataError, match="incompatible provenance"):
        GroupReader(
            [first, foreign],
            batch_size=1,
            force_mode="deephf_relaxed",
        )


def test_energy_only_reader_remains_valid(tmp_path):
    directory = tmp_path / "energy-only"
    directory.mkdir()
    energy = np.array([[0.1], [0.2]], dtype=np.float64)
    descriptor = np.arange(12, dtype=np.float64).reshape(2, 2, 3)
    np.save(directory / "e_corr_target.npy", energy)
    np.save(directory / "descriptor.npy", descriptor)

    reader = Reader(directory, batch_size=1, converged_filter=False)

    assert reader.force_contract is None
    assert set(reader.sample_all()) == {"energy", "descriptor"}
    torch.testing.assert_close(reader.sample_all()["energy"], torch.from_numpy(energy))


def test_legacy_stats_tools_refuse_to_rewrite_strict_force_data(tmp_path):
    directory = tmp_path / "force-data"
    _, arrays = _write_schema_dataset(directory, frame_count=1)

    with pytest.raises(ForceDataError, match="cannot rewrite"):
        concat_data([directory], dump_dir=tmp_path / "combined")
    with pytest.raises(ForceDataError, match="cannot rewrite"):
        make_label(directory, arrays["e_target"])
