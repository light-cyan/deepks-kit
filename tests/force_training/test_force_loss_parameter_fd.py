import numpy as np
import pytest
import torch

from deepks.deephf import write_rhf_force_dataset
from deepks.model.model import CorrNet
from deepks.model.train import (
    Evaluator,
    ForceTrainingError,
    TrainingResult,
    _training_batches,
    main as train_main,
    train,
)

from conftest import ORACLE_PROJECTOR_BASIS
from force_contract_helpers import write_force_contract_sample


TEST_PROJECTOR_BASIS = [[0, [0.8, 1.0]], [0, [0.3, 1.0]]]


def _nonlinear_force_case(directory):
    model = CorrNet(
        input_dim=2,
        hidden_sizes=(2,),
        actv_fn="tanh",
        use_resnet=False,
        proj_basis=TEST_PROJECTOR_BASIS,
    ).double().eval()
    with torch.no_grad():
        model.linear.weight[:] = torch.tensor(
            [[0.08, -0.05]],
            dtype=torch.float64,
        )
        model.linear.bias.fill_(0.01)
        first_layer, output_layer = model.densenet.layers
        first_layer.weight[:] = torch.tensor(
            [[0.31, -0.22], [0.17, 0.29]],
            dtype=torch.float64,
        )
        first_layer.bias[:] = torch.tensor([0.03, -0.04], dtype=torch.float64)
        output_layer.weight[:] = torch.tensor(
            [[0.27, -0.19]],
            dtype=torch.float64,
        )
        output_layer.bias.fill_(0.02)

    descriptor = torch.tensor(
        [
            [[0.2, -0.3], [0.4, 0.1]],
            [[-0.1, 0.5], [0.3, -0.2]],
        ],
        dtype=torch.float64,
    )
    generator = torch.Generator().manual_seed(43)
    jacobian = torch.randn(
        (2, 2, 3, 2, 2),
        generator=generator,
        dtype=torch.float64,
    )
    target_force = torch.tensor(
        [
            [[0.04, -0.03, 0.02], [-0.01, 0.05, -0.02]],
            [[-0.02, 0.01, 0.03], [0.06, -0.04, 0.01]],
        ],
        dtype=torch.float64,
    )
    contract, sample = write_force_contract_sample(
        directory,
        energy=torch.zeros((2, 1), dtype=torch.float64),
        descriptor=descriptor,
        force=target_force,
        jacobian=jacobian,
        projector_basis=TEST_PROJECTOR_BASIS,
        shell_sizes=[1, 1],
    )
    return model, contract, sample


@pytest.mark.parametrize(
    ("parameter_name", "parameter_index"),
    [
        ("linear.weight", (0, 1)),
        ("densenet.layers.0.weight", (0, 1)),
    ],
)
def test_force_loss_parameter_gradient_matches_central_finite_difference(
    parameter_name,
    parameter_index,
    tmp_path,
):
    model, contract, sample = _nonlinear_force_case(tmp_path / "strict-sample")
    evaluator = Evaluator(
        energy_factor=0.0,
        force_factor=1.0,
        force_contract=contract,
    )
    parameter = (
        model.linear.weight
        if parameter_name == "linear.weight"
        else model.densenet.layers[0].weight
    )

    loss = evaluator(model, sample)
    (gradient,) = torch.autograd.grad(loss, parameter)
    analytic = gradient[parameter_index].item()

    step = 1.0e-6
    original = parameter[parameter_index].item()
    finite_difference_values = []
    for direction in (1.0, -1.0):
        with torch.no_grad():
            parameter[parameter_index] = original + direction * step
        finite_difference_values.append(evaluator(model, sample).item())
    with torch.no_grad():
        parameter[parameter_index] = original
    finite_difference = (
        finite_difference_values[0] - finite_difference_values[1]
    ) / (2.0 * step)

    np.testing.assert_allclose(
        analytic,
        finite_difference,
        rtol=2.0e-6,
        atol=2.0e-8,
    )


class _SingleFrameReader:
    group_batch = 1

    def __init__(self):
        self.sample_count = 0
        self.sample = {
            "energy": torch.zeros((1, 1), dtype=torch.float64),
            "descriptor": torch.zeros((1, 1, 1), dtype=torch.float64),
        }

    def get_batch_size(self):
        return 1

    def get_train_size(self):
        return 1

    def sample_train(self):
        self.sample_count += 1
        return self.sample

    def sample_all_batch(self):
        yield self.sample


def test_single_frame_training_epoch_yields_exactly_one_batch():
    reader = _SingleFrameReader()

    batches = list(_training_batches(reader))

    assert len(batches) == 1
    assert reader.sample_count == 1


def test_train_returns_separate_energy_and_force_metrics(tmp_path):
    model, contract, two_frame_sample = _nonlinear_force_case(
        tmp_path / "strict-sample"
    )
    reader = _SingleFrameReader()
    reader.sample = {
        name: value[:1].clone()
        for name, value in two_frame_sample.items()
    }
    reader.force_contract = contract

    result = train(
        model,
        reader,
        n_epoch=1,
        test_reader=reader,
        energy_factor=1.0,
        force_factor=1.0,
        start_lr=1.0e-5,
        display_epoch=2,
        ckpt_file=None,
        device="cpu",
        force_contract=contract,
    )

    assert isinstance(result, TrainingResult)
    assert np.isfinite(result.training_metrics.energy_rmse)
    assert np.isfinite(result.training_metrics.force_rmse)
    assert np.isfinite(result.validation_metrics.energy_rmse)
    assert np.isfinite(result.validation_metrics.force_rmse)
    assert reader.sample_count == 1


def test_public_train_rejects_strict_reader_in_energy_only_mode(tmp_path):
    model, contract, two_frame_sample = _nonlinear_force_case(
        tmp_path / "strict-sample"
    )
    reader = _SingleFrameReader()
    reader.sample = {
        name: value[:1].clone()
        for name, value in two_frame_sample.items()
    }
    reader.force_contract = contract

    with pytest.raises(ForceTrainingError, match="energy-only path"):
        train(
            model,
            reader,
            n_epoch=0,
            test_reader=reader,
            force_factor=0.0,
            ckpt_file=None,
            device="cpu",
        )


def test_train_main_strict_force_contract_initialization_and_restart(
    tmp_path,
    force_generation_case,
):
    case = force_generation_case
    data_path = tmp_path / "force-data"
    contract = write_rhf_force_dataset(
        data_path,
        case.reference,
        projector_basis=ORACLE_PROJECTOR_BASIS,
        e_target=case.target_energy,
        f_target=case.target_force,
    )
    checkpoint = tmp_path / "force-model.pth"
    common = {
        "train_paths": [str(data_path)],
        "model_args": {"hidden_sizes": (2,), "actv_fn": "tanh"},
        "data_args": {"batch_size": 1},
        "preprocess_args": {
            "preshift": False,
            "prescale": False,
            "prefit": False,
        },
        "seed": 91,
        "device": "cpu",
    }

    result = train_main(
        **common,
        ckpt_file=checkpoint,
        train_args={
            "n_epoch": 1,
            "display_epoch": 1,
            "energy_factor": 1.0,
            "force_factor": 1.0,
            "start_lr": 1.0e-5,
        },
    )

    assert isinstance(result, TrainingResult)
    assert checkpoint.is_file()
    assert np.isfinite(result.training_metrics.energy_rmse)
    assert np.isfinite(result.training_metrics.force_rmse)
    loaded = CorrNet.load(
        checkpoint,
        require_force_metadata=True,
        expected_force_contract=contract,
    )
    assert loaded._pbas == result.model._pbas

    restarted_checkpoint = tmp_path / "force-model-restarted.pth"
    restarted = train_main(
        **common,
        restart=checkpoint,
        ckpt_file=restarted_checkpoint,
        train_args={
            "n_epoch": 0,
            "energy_factor": 1.0,
            "force_factor": 1.0,
        },
    )
    assert isinstance(restarted, TrainingResult)
    assert restarted_checkpoint.is_file()

    with pytest.raises(
        ForceTrainingError,
        match="requires force_mode='deephf_relaxed'",
    ):
        train_main(
            **{**common, "data_args": {"batch_size": 1, "force_mode": "none"}},
            train_args={"n_epoch": 0, "force_factor": 1.0},
        )

    wrong_projector = [[0, [0.7, 1.0]], [1, [0.3, 1.0]]]
    with pytest.raises(ForceTrainingError, match="projector metadata does not match"):
        train_main(
            **{
                **common,
                "model_args": {
                    "hidden_sizes": (2,),
                    "actv_fn": "tanh",
                    "proj_basis": wrong_projector,
                },
            },
            train_args={"n_epoch": 0, "force_factor": 1.0},
        )
