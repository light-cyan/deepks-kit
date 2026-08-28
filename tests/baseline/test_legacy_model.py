from io import BytesIO

import numpy as np
import torch

from deepks.model.legacy import load_legacy_corrnet_bytes
from deepks.model.model import CorrNet, corrnet_is_shell_permutation_invariant


def test_load_legacy_corrnet_normalizes_numpy_metadata():
    source = CorrNet(
        input_dim=1,
        hidden_sizes=(2,),
        actv_fn="tanh",
        use_resnet=False,
        proj_basis=[[0, [0.8, 1.0]]],
        elem_table=(np.asarray([1]), np.asarray([0.125])),
    ).double().eval()
    legacy = source.save_dict()
    legacy.pop("format_version")
    legacy["init_args"]["input_dim"] = np.int64(1)
    legacy["init_args"]["elem_table"] = (
        np.asarray([1]),
        np.asarray([0.125]),
    )
    legacy["init_args"]["layer_norm"] = False
    stream = BytesIO()
    torch.save(legacy, stream)

    loaded = load_legacy_corrnet_bytes(stream.getvalue()).eval()

    descriptor = torch.tensor([[[0.4]]], dtype=torch.float64)
    torch.testing.assert_close(loaded(descriptor), source(descriptor))
    assert loaded.input_dim == 1
    assert loaded.elem_dict == {1: 0.125}


def test_thermal_corrnet_shell_symmetry_requires_equal_feature_controls():
    model = CorrNet(
        input_dim=3,
        hidden_sizes=(2,),
        actv_fn="tanh",
        use_resnet=False,
        embedding={"type": "thermal"},
        proj_basis=[[1, [0.8, 1.0]]],
    ).double().eval()
    with torch.no_grad():
        model.linear.weight.fill_(0.25)

    assert corrnet_is_shell_permutation_invariant(model)

    with torch.no_grad():
        model.input_scale[1] = 2.0

    assert not corrnet_is_shell_permutation_invariant(model)


def test_thermal_corrnet_energy_and_sensitivity_obey_shell_permutation():
    model = CorrNet(
        input_dim=3,
        hidden_sizes=(2,),
        actv_fn="tanh",
        use_resnet=False,
        embedding={"type": "thermal"},
        proj_basis=[[1, [0.8, 1.0]]],
    ).double().eval()
    with torch.no_grad():
        model.linear.weight.fill_(0.25)
    descriptor = torch.tensor(
        [[[0.2, 0.5, 0.8]]], dtype=torch.float64, requires_grad=True
    )
    permutation = torch.tensor([2, 0, 1])
    permuted = descriptor.detach()[..., permutation].requires_grad_(True)

    energy = model(descriptor)
    permuted_energy = model(permuted)
    (sensitivity,) = torch.autograd.grad(energy.sum(), descriptor)
    (permuted_sensitivity,) = torch.autograd.grad(
        permuted_energy.sum(), permuted
    )

    torch.testing.assert_close(permuted_energy, energy)
    torch.testing.assert_close(
        permuted_sensitivity,
        sensitivity[..., permutation],
    )


def test_load_legacy_corrnet_rejects_unknown_fields():
    stream = BytesIO()
    torch.save(
        {"state_dict": {}, "init_args": {}, "extra_info": {}, "object": "x"},
        stream,
    )

    try:
        load_legacy_corrnet_bytes(stream.getvalue())
    except ValueError as error:
        assert str(error) == "legacy CorrNet checkpoint fields are invalid"
    else:
        raise AssertionError("unknown legacy checkpoint fields were accepted")
