from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest
import torch

from deepks.data import force_checkpoint_metadata
from deepks.deephf import write_rhf_force_dataset
from deepks.model.evaluate import predict_correction
from deepks.model.model import CorrNet
from deepks.model.reader import GroupReader
from deepks.model.train import (
    Evaluator,
    ForceTrainingError,
)
from force_contract_helpers import write_force_contract_sample


CONTRACT_FINGERPRINT = bytes(range(32)).hex()
ORACLE_PROJECTOR_BASIS = [[0, [0.8, 1.0]], [1, [0.3, 1.0]]]
TEST_PROJECTOR_BASIS = [[0, [0.8, 1.0]], [0, [0.3, 1.0]]]


def _canonical_digest(value):
    encoded = json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _linear_model():
    model = CorrNet(
        input_dim=2,
        hidden_sizes=(2,),
        proj_basis=TEST_PROJECTOR_BASIS,
    ).double().eval()
    with torch.no_grad():
        model.linear.weight[:] = torch.tensor(
            [[0.25, -0.4]],
            dtype=torch.float64,
        )
        model.linear.bias.fill_(0.07)
        model.energy_const.fill_(0.03)
        for parameter in model.densenet.parameters():
            parameter.zero_()
    return model


def _force_case():
    model = _linear_model()
    descriptor = torch.tensor(
        [
            [[0.2, -0.1], [0.5, 0.3]],
            [[-0.4, 0.7], [0.1, -0.2]],
        ],
        dtype=torch.float64,
    )
    dq_dR_relaxed = torch.arange(
        2 * 2 * 3 * 2 * 2,
        dtype=torch.float64,
    ).reshape(2, 2, 3, 2, 2) / 100.0
    differentiable_descriptor = descriptor.clone().requires_grad_(True)
    sensitivity = torch.autograd.grad(
        model(differentiable_descriptor).sum(),
        differentiable_descriptor,
    )[0]
    expected_force = -torch.einsum(
        "fbxik,fik->fbx",
        dq_dR_relaxed,
        sensitivity,
    )
    return model, descriptor, dq_dR_relaxed, sensitivity, expected_force


def _force_checkpoint_metadata():
    return {
        "schema_id": "deepks.deephf.rhf-force-data",
        "schema_version": 1,
        "compatibility_fingerprint": CONTRACT_FINGERPRINT,
        "jacobian_semantics": "dq_dR_relaxed",
        "n_feature": 2,
        "descriptor_definition": "ordered_projected_density_eigenvalues",
        "descriptor_spin_semantics": "spin_summed",
        "descriptor_shell_sizes": [1, 1],
        "projector_sha256": _canonical_digest(TEST_PROJECTOR_BASIS),
        "reference_family": "RHF",
        "response_backend": "rhf_direct",
    }


def test_predict_correction_uses_only_the_complete_relaxed_jacobian():
    model, descriptor, jacobian, sensitivity, expected_force = _force_case()

    prediction = predict_correction(
        model,
        descriptor,
        dq_dR_relaxed=jacobian,
        require_force=True,
        create_graph=True,
    )

    torch.testing.assert_close(
        prediction.descriptor_gradient,
        sensitivity,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        prediction.force,
        expected_force,
        rtol=0.0,
        atol=0.0,
    )
    assert prediction.energy.shape == (2, 1)


def test_nonlinear_normalized_force_matches_descriptor_displacement_energy_fd():
    model = CorrNet(
        input_dim=2,
        hidden_sizes=(3,),
        actv_fn="tanh",
        use_resnet=False,
        input_shift=[0.17, -0.23],
        input_scale=[0.71, 1.29],
        output_scale=1.37,
    ).double().eval()
    with torch.no_grad():
        model.linear.weight[:] = torch.tensor([[0.08, -0.05]], dtype=torch.float64)
        model.linear.bias.fill_(0.013)
        first_layer, output_layer = model.densenet.layers
        first_layer.weight[:] = torch.tensor(
            [[0.31, -0.22], [0.17, 0.29], [-0.26, 0.14]],
            dtype=torch.float64,
        )
        first_layer.bias[:] = torch.tensor([0.03, -0.04, 0.02], dtype=torch.float64)
        output_layer.weight[:] = torch.tensor(
            [[0.27, -0.19, 0.16]],
            dtype=torch.float64,
        )
        output_layer.bias.fill_(0.021)
    descriptor = torch.tensor(
        [[[0.2, -0.3], [0.4, 0.1]]],
        dtype=torch.float64,
    )
    generator = torch.Generator().manual_seed(47)
    jacobian = torch.randn(
        (1, 2, 3, 2, 2),
        generator=generator,
        dtype=torch.float64,
    ) * 0.2

    prediction = predict_correction(
        model,
        descriptor,
        dq_dR_relaxed=jacobian,
        require_force=True,
    )
    finite_difference_force = torch.empty((2, 3), dtype=torch.float64)
    step = 3.0e-5
    with torch.no_grad():
        for atom_index in range(2):
            for coordinate_index in range(3):
                direction = jacobian[0, atom_index, coordinate_index]
                forward = model(descriptor + step * direction.unsqueeze(0))[0, 0]
                backward = model(descriptor - step * direction.unsqueeze(0))[0, 0]
                finite_difference_force[atom_index, coordinate_index] = -(
                    forward - backward
                ) / (2.0 * step)

    np.testing.assert_allclose(
        prediction.force.detach().numpy()[0],
        finite_difference_force.numpy(),
        rtol=2.0e-8,
        atol=2.0e-10,
    )


@pytest.mark.parametrize(
    ("replacement", "error", "match"),
    [
        (None, ValueError, "requires dq_dR_relaxed"),
        (
            torch.zeros((2, 3, 3, 2, 2), dtype=torch.float32),
            TypeError,
            "torch.float64",
        ),
        (
            torch.zeros((2, 3, 2, 3, 2), dtype=torch.float64),
            ValueError,
            "must have shape",
        ),
    ],
)
def test_predict_correction_rejects_missing_or_invalid_relaxed_data(
    replacement,
    error,
    match,
):
    model, descriptor, _, _, _ = _force_case()

    with pytest.raises(error, match=match):
        predict_correction(
            model,
            descriptor,
            dq_dR_relaxed=replacement,
            require_force=True,
        )


def test_predict_correction_rejects_nonfinite_descriptor_and_jacobian():
    model, descriptor, jacobian, _, _ = _force_case()
    bad_descriptor = descriptor.clone()
    bad_descriptor[0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="descriptor must contain only finite"):
        predict_correction(model, bad_descriptor)

    bad_jacobian = jacobian.clone()
    bad_jacobian[0, 0, 0, 0, 0] = torch.inf
    with pytest.raises(ValueError, match="dq_dR_relaxed must contain only finite"):
        predict_correction(model, descriptor, bad_jacobian)


def test_force_evaluator_requires_target_relaxed_jacobian_and_matching_contract(
    tmp_path,
):
    model, descriptor, jacobian, _, expected_force = _force_case()
    expected_energy = model(descriptor).detach()
    contract, sample = write_force_contract_sample(
        tmp_path / "strict-sample",
        energy=expected_energy,
        descriptor=descriptor,
        force=expected_force,
        jacobian=jacobian,
        projector_basis=TEST_PROJECTOR_BASIS,
        shell_sizes=[1, 1],
    )
    evaluator = Evaluator(
        energy_factor=1.0,
        force_factor=1.0,
        force_contract=contract,
    )

    result = evaluator.evaluate(model, sample)
    assert result.total_loss.item() < 1.0e-28
    assert result.energy_metrics.rmse.item() == 0.0
    assert result.force_metrics.rmse.item() < 1.0e-14

    for required_name in (
        "force",
        "dq_dR_relaxed",
        "force_contract_fingerprint",
        "force_sample_fingerprint",
    ):
        incomplete = dict(sample)
        incomplete.pop(required_name)
        with pytest.raises((ValueError, ForceTrainingError), match=required_name):
            evaluator(model, incomplete)

    explicit_only = dict(sample)
    explicit_only["dq_dR_explicit"] = explicit_only.pop("dq_dR_relaxed")
    with pytest.raises(ValueError, match="dq_dR_relaxed"):
        evaluator(model, explicit_only)

    foreign = dict(sample)
    foreign["force_contract_fingerprint"] = torch.full(
        (descriptor.shape[0], 32),
        255,
        dtype=torch.uint8,
    )
    with pytest.raises(ForceTrainingError, match="does not match"):
        evaluator(model, foreign)

    tampered = dict(sample)
    tampered["dq_dR_relaxed"] = sample["dq_dR_relaxed"].clone()
    tampered["dq_dR_relaxed"][0, 0, 0, 0, 0] += 1.0e-5
    with pytest.raises(ForceTrainingError, match="does not belong"):
        evaluator(model, tampered)

    wrong_projector_model = CorrNet(
        input_dim=2,
        hidden_sizes=(2,),
        proj_basis=[[0, [0.7, 1.0]], [0, [0.3, 1.0]]],
    ).double()
    with pytest.raises(ForceTrainingError, match="projector metadata"):
        evaluator(wrong_projector_model, sample)

    with pytest.raises(ForceTrainingError, match="validated ForceDataContract"):
        Evaluator(
            force_factor=1.0,
            force_contract={
                "jacobian_semantics": "dq_dR_relaxed",
                "fingerprint": contract.compatibility_fingerprint,
            },
        )

    with pytest.raises(ForceTrainingError, match="requires force_factor"):
        Evaluator(force_factor=0.0, force_contract=contract)
    with pytest.raises(ForceTrainingError, match="energy-only path"):
        Evaluator(force_factor=0.0).evaluate(model, sample)


def test_force_evaluator_reports_nonzero_component_losses_and_metrics(tmp_path):
    model, descriptor, jacobian, _, predicted_force = _force_case()
    predicted_energy = model(descriptor).detach()
    energy_offset = torch.tensor([[0.2], [-0.4]], dtype=torch.float64)
    force_offset = torch.tensor(
        [
            [[0.1, -0.2, 0.3], [-0.4, 0.5, -0.6]],
            [[-1.0, 1.1, -1.2], [1.3, -1.4, 1.5]],
        ],
        dtype=torch.float64,
    )
    energy_factor = 2.5
    force_factor = 0.4
    contract, sample = write_force_contract_sample(
        tmp_path / "strict-offset-sample",
        energy=predicted_energy + energy_offset,
        descriptor=descriptor,
        force=predicted_force + force_offset,
        jacobian=jacobian,
        projector_basis=TEST_PROJECTOR_BASIS,
        shell_sizes=[1, 1],
    )
    evaluator = Evaluator(
        energy_factor=energy_factor,
        force_factor=force_factor,
        force_contract=contract,
    )

    result = evaluator.evaluate(model, sample)
    expected_energy_loss = energy_offset.square().mean()
    expected_force_loss = force_offset.square().mean()
    expected_total = (
        energy_factor * expected_energy_loss
        + force_factor * expected_force_loss
    )

    torch.testing.assert_close(result.energy_loss, expected_energy_loss)
    torch.testing.assert_close(result.force_loss, expected_force_loss)
    torch.testing.assert_close(result.total_loss, expected_total)
    torch.testing.assert_close(
        result.energy_metrics.mae,
        energy_offset.abs().mean(),
    )
    torch.testing.assert_close(
        result.energy_metrics.rmse,
        energy_offset.square().mean().sqrt(),
    )
    torch.testing.assert_close(
        result.force_metrics.mae,
        force_offset.abs().mean(),
    )
    torch.testing.assert_close(
        result.force_metrics.rmse,
        force_offset.square().mean().sqrt(),
    )


def test_grouped_evaluator_uses_each_validated_sample_contract(tmp_path):
    model, descriptor, jacobian, _, expected_force = _force_case()
    first_energy = model(descriptor).detach()
    first_contract, _ = write_force_contract_sample(
        tmp_path / "first",
        energy=first_energy,
        descriptor=descriptor,
        force=expected_force,
        jacobian=jacobian,
        projector_basis=TEST_PROJECTOR_BASIS,
        shell_sizes=[1, 1],
    )
    second_descriptor = descriptor + 0.031
    second_prediction = predict_correction(
        model,
        second_descriptor,
        dq_dR_relaxed=jacobian,
        require_force=True,
    )
    second_contract, _ = write_force_contract_sample(
        tmp_path / "second",
        energy=second_prediction.energy.detach(),
        descriptor=second_descriptor,
        force=second_prediction.force.detach(),
        jacobian=jacobian,
        projector_basis=TEST_PROJECTOR_BASIS,
        shell_sizes=[1, 1],
    )
    grouped = GroupReader(
        [tmp_path / "first", tmp_path / "second"],
        batch_size=2,
        force_mode="deephf_relaxed",
    )
    assert first_contract.compatibility_fingerprint == (
        second_contract.compatibility_fingerprint
    )
    evaluator = Evaluator(
        force_factor=1.0,
        force_contract=grouped.force_contracts,
    )

    for system_index in range(2):
        result = evaluator.evaluate(model, grouped.sample_all(system_index))
        assert result.energy_metrics.rmse.item() < 1.0e-14
        assert result.force_metrics.rmse.item() < 1.0e-14


def test_energy_only_evaluator_keeps_legacy_checkpoint_path_valid(tmp_path):
    model, descriptor, _, _, _ = _force_case()
    sample = {
        "energy": model(descriptor).detach(),
        "descriptor": descriptor,
    }
    result = Evaluator(force_factor=0.0).evaluate(model, sample)
    assert result.force_loss is None
    assert result.force_metrics is None

    checkpoint = tmp_path / "energy-only.pth"
    model.save(checkpoint, purpose="energy-only")
    loaded = CorrNet.load(checkpoint)
    assert loaded._checkpoint_extra_info == {"purpose": "energy-only"}
    torch.testing.assert_close(loaded(descriptor), model(descriptor))


def test_force_checkpoint_requires_matching_metadata_and_strict_state(tmp_path):
    model, descriptor, _, _, _ = _force_case()
    checkpoint_path = tmp_path / "force-model.pth"
    model.save(
        checkpoint_path,
        force_training=_force_checkpoint_metadata(),
    )

    loaded = CorrNet.load(
        checkpoint_path,
        require_force_metadata=True,
        expected_force_contract_fingerprint=CONTRACT_FINGERPRINT,
    )
    torch.testing.assert_close(loaded(descriptor), model(descriptor))
    assert loaded._checkpoint_extra_info["force_training"][
        "jacobian_semantics"
    ] == "dq_dR_relaxed"

    with pytest.raises(ValueError, match="does not match"):
        CorrNet.load(
            checkpoint_path,
            require_force_metadata=True,
            expected_force_contract_fingerprint=bytes(reversed(range(32))),
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    corrupted = deepcopy(checkpoint)
    corrupted["state_dict"].pop("linear.weight")
    with pytest.raises(ValueError, match="missing linear.weight"):
        CorrNet.load_dict(corrupted)

    corrupted = deepcopy(checkpoint)
    corrupted["state_dict"]["unexpected.weight"] = torch.zeros(
        1,
        dtype=torch.float64,
    )
    with pytest.raises(ValueError, match="unexpected unexpected.weight"):
        CorrNet.load_dict(corrupted)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("schema_id", "foreign.schema"),
        ("schema_version", 2),
        ("jacobian_semantics", "dq_dR_explicit"),
        ("descriptor_definition", "unordered_descriptor"),
        ("descriptor_spin_semantics", "alpha_only"),
        ("reference_family", "UHF"),
        ("response_backend", "finite_difference"),
    ],
)
def test_force_checkpoint_rejects_noncanonical_metadata_values(
    field,
    replacement,
):
    checkpoint = _linear_model().save_dict(
        force_training=_force_checkpoint_metadata(),
    )
    checkpoint["extra_info"]["force_training"][field] = replacement

    with pytest.raises(ValueError, match=field):
        CorrNet.load_dict(checkpoint)


@pytest.mark.parametrize(
    ("field", "replacement", "match"),
    [
        ("n_feature", 3, "n_feature does not match"),
        ("descriptor_shell_sizes", [2], "descriptor_shell_sizes do not match"),
        ("projector_sha256", "0" * 64, "projector_sha256 does not match"),
    ],
)
def test_force_checkpoint_rejects_metadata_inconsistent_with_model(
    field,
    replacement,
    match,
):
    checkpoint = _linear_model().save_dict(
        force_training=_force_checkpoint_metadata(),
    )
    checkpoint["extra_info"]["force_training"][field] = replacement
    if field == "n_feature":
        checkpoint["extra_info"]["force_training"][
            "descriptor_shell_sizes"
        ] = [1, 2]

    with pytest.raises(ValueError, match=match):
        CorrNet.load_dict(checkpoint)


@pytest.mark.parametrize("corruption", ["shape", "dtype", "nonfinite"])
def test_force_checkpoint_rejects_inexact_or_nonfinite_state(corruption):
    checkpoint = deepcopy(
        _linear_model().save_dict(force_training=_force_checkpoint_metadata())
    )
    parameter = checkpoint["state_dict"]["linear.weight"]
    if corruption == "shape":
        checkpoint["state_dict"]["linear.weight"] = parameter[:, :1]
        error = ValueError
        match = "has shape"
    elif corruption == "dtype":
        checkpoint["state_dict"]["linear.weight"] = parameter.float()
        error = TypeError
        match = "has dtype"
    else:
        parameter[0, 0] = torch.nan
        error = ValueError
        match = "only finite values"

    with pytest.raises(error, match=match):
        CorrNet.load_dict(checkpoint)


def test_force_checkpoint_rejects_external_element_table():
    model = CorrNet(
        input_dim=2,
        hidden_sizes=(2,),
        proj_basis=TEST_PROJECTOR_BASIS,
        elem_table=([1], [0.1]),
    ).double()
    checkpoint = model.save_dict(force_training=_force_checkpoint_metadata())

    with pytest.raises(ValueError, match="external element table"):
        CorrNet.load_dict(checkpoint, require_force_metadata=True)


def test_real_contract_reader_prediction_and_checkpoint_reload(
    tmp_path,
    force_generation_case,
):
    case = force_generation_case
    data_path = tmp_path / "strict-force-data"
    contract = write_rhf_force_dataset(
        data_path,
        case.reference,
        projector_basis=ORACLE_PROJECTOR_BASIS,
        e_target=case.target_energy,
        f_target=case.target_force,
    )
    reader = GroupReader(
        [data_path],
        batch_size=1,
        force_mode="deephf_relaxed",
    )
    sample = reader.sample_all(0)
    evaluator = Evaluator(
        energy_factor=1.0,
        force_factor=1.0,
        force_contract=reader.force_contract,
    )

    result = evaluator.evaluate(case.teacher_method.model, sample)

    assert result.energy_metrics.rmse.item() < 1.0e-14
    assert result.force_metrics.rmse.item() < 2.0e-12
    checkpoint_path = tmp_path / "strict-force-model.pth"
    case.teacher_method.model.save(
        checkpoint_path,
        force_training=force_checkpoint_metadata(contract),
    )
    loaded = CorrNet.load(
        checkpoint_path,
        require_force_metadata=True,
        expected_force_contract=contract,
    )
    loaded_result = evaluator.evaluate(loaded, sample)
    torch.testing.assert_close(
        loaded_result.prediction.energy,
        result.prediction.energy,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        loaded_result.prediction.force,
        result.prediction.force,
        rtol=0.0,
        atol=0.0,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    wrong_projector_checkpoint = deepcopy(checkpoint)
    wrong_projector_checkpoint["init_args"]["proj_basis"] = [
        [0, [0.7, 1.0]],
        [1, [0.3, 1.0]],
    ]
    with pytest.raises(ValueError, match="projector_sha256 does not match"):
        CorrNet.load_dict(
            wrong_projector_checkpoint,
            require_force_metadata=True,
            expected_force_contract=contract,
        )
