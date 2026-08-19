import numpy as np
import torch
from pyscf import gto

from deepks.model.model import CorrNet
from deepks.model.reader import Reader
from deepks.model.train import Evaluator, main as train_model
from deepks.scf.fields import select_fields
from deepks.scf.run import collect_fields, dump_data, dump_meta, solve_mol


PROJECTOR_BASIS = [[0, [1.0, 1.0]]]


def _make_linear_model():
    model = CorrNet(
        input_dim=1,
        hidden_sizes=(2,),
        proj_basis=PROJECTOR_BASIS,
    ).double()
    with torch.no_grad():
        model.linear.weight.fill_(1.0e-3)
        model.linear.bias.fill_(2.0e-4)
        for parameter in model.densenet.parameters():
            parameter.zero_()
    return model


def _write_force_dataset(tmp_path):
    molecule = gto.M(
        atom="H 0 0 0; H 0 0 1.4",
        basis="sto-3g",
        unit="Bohr",
        verbose=0,
    )
    model = _make_linear_model()
    selected = select_fields(
        [
            "atom",
            "e_base",
            "e_tot",
            "dm_eig",
            "conv",
            "f_base",
            "f_tot",
            "grad_vx",
        ]
    )

    metadata, result = solve_mol(
        molecule,
        model,
        selected,
        conv_tol=1.0e-10,
        max_cycle=50,
    )
    exported = collect_fields(selected, metadata, result)
    exported["l_e_delta"] = exported["e_tot"] - exported["e_base"]
    exported["l_f_delta"] = exported["f_tot"] - exported["f_base"]

    data_directory = tmp_path / "h2"
    dump_meta(data_directory, metadata)
    dump_data(data_directory, **exported)
    return molecule, model, data_directory


def test_scf_export_reader_and_force_evaluator_pipeline(tmp_path):
    _, model, data_directory = _write_force_dataset(tmp_path)

    expected_files = {
        "atom.npy",
        "conv.npy",
        "dm_eig.npy",
        "e_base.npy",
        "e_tot.npy",
        "f_base.npy",
        "f_tot.npy",
        "grad_vx.npy",
        "l_e_delta.npy",
        "l_f_delta.npy",
        "system.raw",
    }
    assert {path.name for path in data_directory.iterdir()} == expected_files
    np.testing.assert_array_equal(
        np.loadtxt(data_directory / "system.raw", dtype=int),
        np.array([2, 2, 2, 1]),
    )

    reader = Reader(data_directory, batch_size=1)
    sample = reader.sample_all()

    assert set(sample) == {"eig", "gvx", "lb_e", "lb_f"}
    assert sample["eig"].shape == (1, 2, 1)
    assert sample["gvx"].shape == (1, 2, 3, 2, 1)
    assert sample["lb_e"].shape == (1, 1)
    assert sample["lb_f"].shape == (1, 2, 3)

    descriptors = sample["eig"].clone().requires_grad_(True)
    predicted_energy = model(descriptors)
    (energy_descriptor_gradient,) = torch.autograd.grad(
        predicted_energy,
        descriptors,
        grad_outputs=torch.ones_like(predicted_energy),
    )
    predicted_force = -torch.einsum(
        "...bxap,...ap->...bx",
        sample["gvx"],
        energy_descriptor_gradient,
    )

    torch.testing.assert_close(predicted_energy, sample["lb_e"], atol=1.0e-10, rtol=0.0)
    torch.testing.assert_close(predicted_force, sample["lb_f"], atol=1.0e-10, rtol=0.0)

    perturbed_sample = dict(sample)
    perturbed_sample["lb_f"] = sample["lb_f"].clone()
    perturbed_sample["lb_f"][0, 0, 2] += 1.0e-3

    evaluator = Evaluator(energy_factor=1.0, force_factor=1.0)
    loss = evaluator(model, perturbed_sample)
    assert loss.item() > 1.0e-10
    loss.backward()
    assert model.linear.weight.grad is not None
    assert torch.isfinite(model.linear.weight.grad).all()
    assert torch.linalg.vector_norm(model.linear.weight.grad) > 0


def test_force_training_checkpoint_can_be_reloaded_for_scf(tmp_path):
    molecule, _, data_directory = _write_force_dataset(tmp_path)
    checkpoint = tmp_path / "trained-model.pth"

    train_model(
        train_paths=[str(data_directory)],
        test_paths=[str(data_directory)],
        ckpt_file=str(checkpoint),
        model_args={
            "hidden_sizes": (2,),
            "proj_basis": PROJECTOR_BASIS,
        },
        data_args={"batch_size": 1},
        preprocess_args={
            "preshift": False,
            "prescale": False,
            "prefit": False,
        },
        train_args={
            "n_epoch": 1,
            "display_epoch": 1,
            "energy_factor": 1.0,
            "force_factor": 1.0,
            "start_lr": 1.0e-4,
        },
        seed=11,
        device="cpu",
    )

    loaded = CorrNet.load(checkpoint).double().eval()
    selected = select_fields(["e_tot", "dm_eig", "conv", "f_tot"])
    _, result = solve_mol(
        molecule,
        loaded,
        selected,
        conv_tol=1.0e-10,
        max_cycle=50,
    )

    assert result["conv"]
    assert np.isfinite(result["e_tot"])
    assert np.isfinite(result["dm_eig"]).all()
    assert np.isfinite(result["f_tot"]).all()
