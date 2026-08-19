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
