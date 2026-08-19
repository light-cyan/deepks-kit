import numpy as np
import pytest
import torch

from deepks.model.model import CorrNet, masked_softmax
from deepks.model.train import make_loss


def test_corrnet_forward_and_input_gradient():
    model = CorrNet(input_dim=3, hidden_sizes=(4,)).double()
    descriptors = torch.randn(2, 2, 3, dtype=torch.float64, requires_grad=True)

    energy = model(descriptors)
    gradient = torch.autograd.grad(energy.sum(), descriptors)[0]

    assert energy.shape == (2, 1)
    assert gradient.shape == descriptors.shape
    assert torch.isfinite(energy).all()
    assert torch.isfinite(gradient).all()


def test_masked_softmax_ignores_masked_entries():
    values = torch.tensor([[1.0, 2.0, 3.0]])
    mask = torch.tensor([[True, False, True]])

    result = masked_softmax(values, mask)

    assert result[0, 1] == 0
    torch.testing.assert_close(result.sum(dim=-1), torch.ones(1))


def test_capped_loss_is_quadratic_then_linear():
    loss = make_loss(cap=1.0, reduction="none")

    result = loss(torch.tensor([0.5, 2.0]), torch.zeros(2))

    torch.testing.assert_close(result, torch.tensor([0.25, 3.0]))


def test_corrnet_checkpoint_round_trip_preserves_predictions(tmp_path):
    torch.manual_seed(7)
    model = CorrNet(
        input_dim=np.int64(3),
        hidden_sizes=(4,),
        proj_basis=[[0, [1.2, 1.0]]],
        elem_table=(np.array([1, 2]), np.array([-0.25, -0.5])),
    ).double().eval()
    descriptors = torch.tensor(
        [[[0.2, -0.1, 0.4], [0.5, 0.3, -0.2]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    expected_energy = model(descriptors)
    expected_gradient = torch.autograd.grad(expected_energy.sum(), descriptors)[0]

    checkpoint = tmp_path / "model.pth"
    model.save(checkpoint, purpose="baseline-round-trip")
    serialized = torch.load(checkpoint, map_location="cpu", weights_only=True)
    loaded = CorrNet.load(checkpoint).double().eval()

    loaded_descriptors = descriptors.detach().clone().requires_grad_(True)
    actual_energy = loaded(loaded_descriptors)
    actual_gradient = torch.autograd.grad(actual_energy.sum(), loaded_descriptors)[0]

    torch.testing.assert_close(actual_energy, expected_energy, rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0.0, atol=0.0)
    assert serialized["format_version"] == 1
    assert serialized["init_args"]["input_dim"] == 3
    assert serialized["extra_info"] == {"purpose": "baseline-round-trip"}
    assert loaded.elem_dict == {1: -0.25, 2: -0.5}


def test_corrnet_compiled_checkpoint_remains_loadable(tmp_path):
    torch.manual_seed(13)
    model = CorrNet(input_dim=3, hidden_sizes=(4,)).double().eval()
    descriptors = torch.tensor(
        [[[0.2, -0.1, 0.4], [0.5, 0.3, -0.2]]],
        dtype=torch.float64,
    )
    expected_energy = model(descriptors)

    checkpoint = tmp_path / "compiled-model.pt"
    model.compile_save(str(checkpoint))
    loaded = CorrNet.load(checkpoint)

    torch.testing.assert_close(
        loaded(descriptors),
        expected_energy,
        rtol=0.0,
        atol=0.0,
    )


def test_corrnet_rejects_unversioned_parameter_checkpoint(tmp_path):
    checkpoint = CorrNet(input_dim=2, hidden_sizes=(3,)).save_dict()
    del checkpoint["format_version"]
    checkpoint_path = tmp_path / "unversioned-model.pth"
    torch.save(checkpoint, checkpoint_path)

    with pytest.raises(ValueError, match="unsupported CorrNet checkpoint format"):
        CorrNet.load(checkpoint_path)
