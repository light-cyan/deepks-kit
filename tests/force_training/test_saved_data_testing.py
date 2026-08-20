import copy

import numpy as np
import pytest
import torch

from deepks.data.force_schema import (
    ForceDataError,
    force_checkpoint_metadata,
    _write_force_dataset as write_force_dataset,
)
from deepks.model.evaluate import predict_correction
from deepks.model.model import CorrNet
from deepks.model.reader import FORCE_MODE_DEEPHF_RELAXED, GroupReader
from deepks.model.test import main as saved_data_main
from deepks.model.test import test as run_saved_data_test
from deepks.utils import save_yaml
from test_force_schema import make_schema_inputs
from force_contract_helpers import write_force_contract_sample


TEST_PROJECTOR_BASIS = [[0, [0.8, 1.0]], [0, [0.3, 1.0]]]


def _make_force_checkpoint(path, contract, provenance):
    torch.manual_seed(23)
    model = CorrNet(
        input_dim=contract.dimensions["n_feature"],
        hidden_sizes=(4,),
        actv_fn="tanh",
        use_resnet=False,
        proj_basis=provenance["descriptor"]["projector_basis"],
    ).double().eval()
    with torch.no_grad():
        values = torch.linspace(
            -0.19,
            0.23,
            sum(parameter.numel() for parameter in model.parameters()),
            dtype=torch.float64,
        )
        offset = 0
        for parameter in model.parameters():
            count = parameter.numel()
            if parameter.requires_grad:
                parameter.copy_(values[offset : offset + count].reshape(parameter.shape))
            offset += count
    model.save(path, force_training=force_checkpoint_metadata(contract))
    return model


def test_strict_saved_data_testing_reports_force_metrics_and_predictions(
    tmp_path,
    capsys,
):
    arrays, provenance = make_schema_inputs(frame_count=2)
    data_directory = tmp_path / "force-data"
    contract = write_force_dataset(
        data_directory,
        arrays=arrays,
        provenance=provenance,
    )
    checkpoint = tmp_path / "force-model.pth"
    model = _make_force_checkpoint(checkpoint, contract, provenance)

    results = saved_data_main(
        [str(data_directory)],
        model_file=str(checkpoint),
        output_prefix="saved-test",
        group=False,
        force_mode=FORCE_MODE_DEEPHF_RELAXED,
    )

    assert len(results) == 1
    result = results[0]
    descriptor = torch.from_numpy(arrays["descriptor"].copy())
    relaxed_jacobian = torch.from_numpy(arrays["dq_dR_relaxed"].copy())
    expected = predict_correction(
        model,
        descriptor,
        dq_dR_relaxed=relaxed_jacobian,
        require_force=True,
    )
    expected_energy = expected.energy.detach().numpy().reshape(-1)
    expected_force = expected.force.detach().numpy()
    np.testing.assert_allclose(
        result.energy_predictions[0],
        expected_energy,
        rtol=1.0e-13,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        result.force_predictions[0],
        expected_force,
        rtol=1.0e-13,
        atol=1.0e-14,
    )
    energy_difference = expected_energy - arrays["e_corr_target"].reshape(-1)
    force_difference = expected_force - arrays["f_corr_target"]
    assert result.energy_mae == pytest.approx(np.mean(np.abs(energy_difference)))
    assert result.energy_rmse == pytest.approx(
        np.sqrt(np.mean(np.square(energy_difference)))
    )
    assert result.force_mae == pytest.approx(np.mean(np.abs(force_difference)))
    assert result.force_rmse == pytest.approx(
        np.sqrt(np.mean(np.square(force_difference)))
    )
    assert (tmp_path / "saved-test.00.out").is_file()
    assert (tmp_path / "saved-test.00.force.out").is_file()
    output = capsys.readouterr().out
    assert "all systems energy MAE" in output
    assert "all systems energy RMSE" in output
    assert "all systems force MAE" in output
    assert "all systems force RMSE" in output


def test_energy_only_saved_data_testing_remains_valid(tmp_path):
    data_directory = tmp_path / "energy-data"
    data_directory.mkdir()
    descriptor = np.array(
        [
            [[0.2, -0.1, 0.4], [0.5, 0.3, -0.2]],
            [[-0.4, 0.1, 0.7], [0.3, -0.6, 0.2]],
        ],
        dtype=np.float64,
    )
    model = CorrNet(input_dim=3, hidden_sizes=(3,)).double().eval()
    energy = model(torch.from_numpy(descriptor)).detach().numpy()
    np.save(data_directory / "descriptor.npy", descriptor, allow_pickle=False)
    np.save(data_directory / "e_corr_target.npy", energy, allow_pickle=False)
    checkpoint = tmp_path / "energy-model.pth"
    model.save(checkpoint, purpose="energy-only")

    result = saved_data_main(
        [str(data_directory)],
        model_file=str(checkpoint),
        output_prefix=None,
    )[0]

    assert result.energy_mae == 0.0
    assert result.energy_rmse == 0.0
    assert result.force_mae is None
    assert result.force_rmse is None
    assert result.force_predictions is None
    unpacked_mae, unpacked_rmse = result
    assert unpacked_mae == 0.0
    assert unpacked_rmse == 0.0


def test_force_saved_data_testing_rejects_checkpoint_without_force_metadata(tmp_path):
    arrays, provenance = make_schema_inputs(frame_count=1)
    data_directory = tmp_path / "force-data"
    write_force_dataset(data_directory, arrays=arrays, provenance=provenance)
    checkpoint = tmp_path / "energy-only.pth"
    CorrNet(
        input_dim=3,
        hidden_sizes=(2,),
        proj_basis=provenance["descriptor"]["projector_basis"],
    ).double().save(checkpoint)

    with pytest.raises(ValueError, match="missing force_training metadata"):
        saved_data_main(
            [str(data_directory)],
            model_file=str(checkpoint),
            output_prefix=None,
            force_mode=FORCE_MODE_DEEPHF_RELAXED,
        )


def test_force_saved_data_testing_rejects_incompatible_checkpoint_contract(tmp_path):
    arrays, provenance = make_schema_inputs(frame_count=1)
    first_directory = tmp_path / "first"
    first_contract = write_force_dataset(
        first_directory,
        arrays=arrays,
        provenance=provenance,
    )
    checkpoint = tmp_path / "first.pth"
    _make_force_checkpoint(checkpoint, first_contract, provenance)

    second_provenance = copy.deepcopy(provenance)
    second_provenance["descriptor"]["projector_basis"][0][1][0] = 0.91
    second_directory = tmp_path / "second"
    write_force_dataset(
        second_directory,
        arrays=copy.deepcopy(arrays),
        provenance=second_provenance,
    )

    with pytest.raises(ValueError, match="fingerprint does not match"):
        saved_data_main(
            [str(second_directory)],
            model_file=str(checkpoint),
            output_prefix=None,
            force_mode=FORCE_MODE_DEEPHF_RELAXED,
        )


def test_force_saved_data_testing_rejects_missing_manifest_or_relaxed_field(tmp_path):
    missing_manifest = tmp_path / "missing-manifest"
    missing_manifest.mkdir()
    arrays, provenance = make_schema_inputs(frame_count=1)
    np.save(
        missing_manifest / "e_corr_target.npy",
        arrays["e_corr_target"],
        allow_pickle=False,
    )
    np.save(
        missing_manifest / "descriptor.npy",
        arrays["descriptor"],
        allow_pickle=False,
    )
    with pytest.raises(ForceDataError, match="force_data.json"):
        GroupReader(
            [str(missing_manifest)],
            force_mode=FORCE_MODE_DEEPHF_RELAXED,
        )

    incomplete = tmp_path / "incomplete"
    write_force_dataset(incomplete, arrays=arrays, provenance=provenance)
    (incomplete / "dq_dR_relaxed.npy").unlink()
    with pytest.raises(ForceDataError, match="dq_dR_relaxed"):
        GroupReader(
            [str(incomplete)],
            force_mode=FORCE_MODE_DEEPHF_RELAXED,
        )


def test_strict_saved_data_directory_cannot_default_to_energy_only(tmp_path):
    arrays, provenance = make_schema_inputs(frame_count=1)
    data_directory = tmp_path / "force-data"
    contract = write_force_dataset(
        data_directory,
        arrays=arrays,
        provenance=provenance,
    )
    checkpoint = tmp_path / "force.pth"
    _make_force_checkpoint(checkpoint, contract, provenance)

    with pytest.raises(ForceDataError, match="force_mode='deephf_relaxed'"):
        saved_data_main(
            [str(data_directory)],
            model_file=str(checkpoint),
            output_prefix=None,
        )


def test_strict_reader_cannot_be_forced_through_energy_only_test_path(tmp_path):
    arrays, provenance = make_schema_inputs(frame_count=1)
    data_directory = tmp_path / "force-data"
    contract = write_force_dataset(
        data_directory,
        arrays=arrays,
        provenance=provenance,
    )
    checkpoint = tmp_path / "force.pth"
    _make_force_checkpoint(checkpoint, contract, provenance)
    model = CorrNet.load(
        checkpoint,
        require_force_metadata=True,
        expected_force_contract=contract,
    ).double()
    reader = GroupReader(
        [str(data_directory)],
        force_mode=FORCE_MODE_DEEPHF_RELAXED,
    )

    with pytest.raises(ForceDataError, match="cannot be tested.*energy-only"):
        run_saved_data_test(
            model,
            reader,
            dump_prefix=None,
            force_aware=False,
        )


def test_grouped_saved_data_uses_all_sample_contracts(tmp_path):
    torch.manual_seed(61)
    model = CorrNet(
        input_dim=2,
        hidden_sizes=(3,),
        actv_fn="tanh",
        use_resnet=False,
        proj_basis=TEST_PROJECTOR_BASIS,
    ).double().eval()
    descriptor = torch.tensor(
        [[[0.2, -0.1], [0.4, 0.3]]],
        dtype=torch.float64,
    )
    jacobian = torch.arange(
        1 * 2 * 3 * 2 * 2,
        dtype=torch.float64,
    ).reshape(1, 2, 3, 2, 2) / 100.0
    contracts = []
    for index, shifted_descriptor in enumerate(
        (descriptor, descriptor + 0.037)
    ):
        prediction = predict_correction(
            model,
            shifted_descriptor,
            dq_dR_relaxed=jacobian,
            require_force=True,
        )
        contract, _ = write_force_contract_sample(
            tmp_path / f"system-{index}",
            energy=prediction.energy.detach(),
            descriptor=shifted_descriptor,
            force=prediction.force.detach(),
            jacobian=jacobian,
            projector_basis=TEST_PROJECTOR_BASIS,
            shell_sizes=[1, 1],
        )
        contracts.append(contract)
    assert contracts[0].compatibility_fingerprint == (
        contracts[1].compatibility_fingerprint
    )
    checkpoint = tmp_path / "grouped-force.pth"
    model.save(
        checkpoint,
        force_training=force_checkpoint_metadata(contracts[0]),
    )

    (result,) = saved_data_main(
        [str(tmp_path / "system-0"), str(tmp_path / "system-1")],
        model_file=str(checkpoint),
        output_prefix=None,
        force_mode=FORCE_MODE_DEEPHF_RELAXED,
    )

    assert len(result.systems) == 2
    assert result.energy_rmse < 1.0e-14
    assert result.force_rmse < 1.0e-14


def test_saved_data_cli_forwards_strict_force_mode_from_training_yaml(
    tmp_path,
    monkeypatch,
):
    configuration = {
        "train_paths": ["unused-train"],
        "test_paths": ["strict-validation"],
        "data_args": {
            "energy_name": "e_corr_target",
            "descriptor_name": "descriptor",
            "force_mode": "deephf_relaxed",
        },
        "train_args": {
            "ckpt_file": "strict-model.pth",
            "force_factor": 1.0,
        },
    }
    input_file = tmp_path / "force-train.yaml"
    save_yaml(configuration, input_file)
    captured = {}

    def fake_main(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("deepks.model.test.main", fake_main)
    from deepks.main import test_cli

    test_cli([str(input_file)])

    assert captured["data_paths"] == ["strict-validation"]
    assert captured["model_file"] == "strict-model.pth"
    assert captured["force_mode"] == "deephf_relaxed"
