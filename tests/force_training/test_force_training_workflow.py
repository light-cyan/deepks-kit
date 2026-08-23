import math

import numpy as np
import torch

from deepks.deephf import DeePHF, write_rhf_force_dataset
from deepks.model.evaluate import predict_correction
from deepks.model.model import CorrNet
from deepks.model.reader import FORCE_MODE_DEEPHF_RELAXED, GroupReader
from deepks.model.test import main as saved_data_main
from deepks.model.train import Evaluator, main as train_main, train
ORACLE_PROJECTOR_BASIS = [[0, [0.8, 1.0]], [1, [0.3, 1.0]]]


def _zero_linear_model():
    model = CorrNet(
        input_dim=4,
        hidden_sizes=(2,),
        actv_fn="tanh",
        use_resnet=False,
        proj_basis=ORACLE_PROJECTOR_BASIS,
    ).double()
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.zero_()
    for parameter in model.densenet.parameters():
        parameter.requires_grad_(False)
    return model


def _prediction(model, reader):
    sample = reader.sample_all(0)
    return predict_correction(
        model,
        sample["descriptor"],
        dq_dR_relaxed=sample["dq_dR_relaxed"],
        require_force=True,
    )


def _assert_finite_metrics(metrics):
    assert math.isfinite(metrics.energy_mae)
    assert math.isfinite(metrics.energy_rmse)
    assert math.isfinite(metrics.force_mae)
    assert math.isfinite(metrics.force_rmse)


def test_rhf_force_training_checkpoint_and_fresh_deephf_workflow(
    tmp_path,
    force_generation_case,
):
    torch.manual_seed(20260820)
    np.random.seed(20260820)
    case = force_generation_case
    second_teacher = DeePHF(
        case.forward_reference,
        case.teacher_method.model,
        projector_basis=ORACLE_PROJECTOR_BASIS,
    )
    second_energy = np.float64(second_teacher.kernel())
    second_force = np.asarray(
        -second_teacher.nuc_grad_method().kernel(),
        dtype=np.float64,
    )
    data_path = tmp_path / "rhf-force-data"
    contract = write_rhf_force_dataset(
        data_path,
        [case.reference, case.forward_reference],
        projector_basis=ORACLE_PROJECTOR_BASIS,
        e_target=np.array(
            [case.target_energy, second_energy],
            dtype=np.float64,
        ),
        f_target=np.stack([case.target_force, second_force], axis=0),
    )

    reader_args = {
        "batch_size": 2,
        "force_mode": FORCE_MODE_DEEPHF_RELAXED,
    }
    training_reader = GroupReader([data_path], **reader_args)
    validation_reader = GroupReader([data_path], **reader_args)
    assert training_reader.get_train_size() == 2
    assert training_reader.force_contract.manifest_fingerprint == (
        contract.manifest_fingerprint
    )
    assert validation_reader.force_contract.compatibility_fingerprint == (
        contract.compatibility_fingerprint
    )

    model = _zero_linear_model()
    evaluator = Evaluator(
        energy_factor=1.0,
        force_factor=1.0,
        force_contract=validation_reader.force_contract,
    )
    initial = evaluator.evaluate(
        model,
        validation_reader.sample_all(0),
        create_graph=False,
    )
    initial_energy_rmse = initial.energy_metrics.rmse.item()
    initial_force_rmse = initial.force_metrics.rmse.item()
    assert initial_energy_rmse > 0.0
    assert initial_force_rmse > 0.0

    checkpoint = tmp_path / "rhf-force-model.pth"
    training_result = train(
        model,
        training_reader,
        n_epoch=160,
        test_reader=validation_reader,
        energy_factor=1.0,
        force_factor=1.0,
        start_lr=5.0e-3,
        decay_steps=80,
        decay_rate=0.5,
        display_epoch=160,
        ckpt_file=checkpoint,
        device="cpu",
        force_contract=contract,
    )
    _assert_finite_metrics(training_result.training_metrics)
    _assert_finite_metrics(training_result.validation_metrics)
    assert training_result.training_metrics.energy_rmse < initial_energy_rmse * 0.2
    assert training_result.training_metrics.force_rmse < initial_force_rmse * 0.5
    assert training_result.validation_metrics.energy_rmse < initial_energy_rmse * 0.2
    assert training_result.validation_metrics.force_rmse < initial_force_rmse * 0.5

    before_reload = _prediction(training_result.model.eval(), validation_reader)
    loaded = CorrNet.load(
        checkpoint,
        require_force_metadata=True,
        expected_force_contract=contract,
    ).double().eval()
    after_reload = _prediction(loaded, validation_reader)
    torch.testing.assert_close(
        after_reload.energy,
        before_reload.energy,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        after_reload.force,
        before_reload.force,
        rtol=0.0,
        atol=0.0,
    )

    saved_result = saved_data_main(
        [str(data_path)],
        model_file=str(checkpoint),
        output_prefix=None,
        force_mode=FORCE_MODE_DEEPHF_RELAXED,
    )[0]
    assert math.isfinite(saved_result.energy_mae)
    assert math.isfinite(saved_result.energy_rmse)
    assert math.isfinite(saved_result.force_mae)
    assert math.isfinite(saved_result.force_rmse)
    np.testing.assert_allclose(
        saved_result.energy_predictions[0],
        after_reload.energy.detach().cpu().numpy().reshape(-1),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        saved_result.force_predictions[0],
        after_reload.force.detach().cpu().numpy(),
        rtol=0.0,
        atol=0.0,
    )

    restart_checkpoint = tmp_path / "rhf-force-restart.pth"
    restart_result = train_main(
        [str(data_path)],
        test_paths=[str(data_path)],
        restart=str(checkpoint),
        ckpt_file=str(restart_checkpoint),
        data_args=reader_args,
        preprocess_args={
            "preshift": False,
            "prescale": False,
            "prefit": False,
        },
        train_args={
            "n_epoch": 1,
            "energy_factor": 1.0,
            "force_factor": 1.0,
            "start_lr": 1.0e-5,
            "display_epoch": 1,
        },
        seed=20260820,
        device="cpu",
    )
    _assert_finite_metrics(restart_result.training_metrics)
    _assert_finite_metrics(restart_result.validation_metrics)
    assert restart_checkpoint.is_file()
    CorrNet.load(
        restart_checkpoint,
        require_force_metadata=True,
        expected_force_contract=contract,
    )

    fresh_method = DeePHF(
        case.backward_reference,
        loaded,
        projector_basis=ORACLE_PROJECTOR_BASIS,
    )
    total_energy = fresh_method.kernel()
    fresh_gradient = fresh_method.nuc_grad_method().run()
    fresh_prediction = predict_correction(
        loaded,
        torch.from_numpy(fresh_method.descriptor()).unsqueeze(0),
        dq_dR_relaxed=torch.from_numpy(
            fresh_gradient.dq_dR_relaxed,
        ).unsqueeze(0),
        require_force=True,
    )
    assert total_energy == fresh_method.e_base + fresh_method.e_corr
    np.testing.assert_allclose(
        fresh_prediction.energy.detach().cpu().numpy().reshape(()),
        fresh_method.e_corr,
        rtol=1.0e-13,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        fresh_prediction.force.detach().cpu().numpy()[0],
        -fresh_gradient.correction_gradient,
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    contracted_force = -np.einsum(
        "bxik,ik->bx",
        fresh_gradient.dq_dR_relaxed,
        fresh_method.correction_sensitivity(),
    )
    np.testing.assert_allclose(
        fresh_prediction.force.detach().cpu().numpy()[0],
        contracted_force,
        rtol=2.0e-12,
        atol=2.0e-12,
    )
