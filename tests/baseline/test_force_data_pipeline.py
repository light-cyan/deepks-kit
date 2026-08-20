import numpy as np
import torch
from pyscf import gto

from deepks.model.model import CorrNet
from deepks.model.reader import Reader
from deepks.model.train import main as train_model
from deepks.data.fields import select_fields
from deepks.data.io import collect_field_results, dump_data, dump_metadata
from deepks.deepks.run import solve_molecule


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
            "e_corr",
            "e_tot",
            "descriptor",
            "converged",
            "f_reference_variational",
            "f_corr_explicit",
            "f_tot",
            "dq_dR_explicit",
        ]
    )

    metadata, result = solve_molecule(
        molecule,
        model,
        selected,
        conv_tol=1.0e-10,
        max_cycle=50,
    )
    exported = collect_field_results(selected, metadata, result)
    exported["e_corr_target"] = exported["e_tot"] - exported["e_base"]
    exported["f_corr_explicit_target"] = (
        exported["f_tot"] - exported["f_reference_variational"]
    )
    np.testing.assert_allclose(
        exported["e_corr"],
        exported["e_corr_target"],
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        exported["f_corr_explicit"],
        exported["f_corr_explicit_target"],
        rtol=0.0,
        atol=1.0e-12,
    )

    data_directory = tmp_path / "h2"
    dump_metadata(data_directory, metadata)
    dump_data(data_directory, **exported)
    return molecule, model, data_directory


def test_deepks_scf_export_keeps_explicit_force_fields(tmp_path):
    _, model, data_directory = _write_force_dataset(tmp_path)

    expected_files = {
        "atom.npy",
        "converged.npy",
        "descriptor.npy",
        "e_base.npy",
        "e_corr.npy",
        "e_tot.npy",
        "f_reference_variational.npy",
        "f_corr_explicit.npy",
        "f_tot.npy",
        "dq_dR_explicit.npy",
        "e_corr_target.npy",
        "f_corr_explicit_target.npy",
        "system.raw",
    }
    assert {path.name for path in data_directory.iterdir()} == expected_files
    np.testing.assert_array_equal(
        np.loadtxt(data_directory / "system.raw", dtype=int),
        np.array([2, 2, 2, 1]),
    )

    reader = Reader(data_directory, batch_size=1)
    sample = reader.sample_all()

    assert set(sample) == {"descriptor", "energy"}
    assert sample["descriptor"].shape == (1, 2, 1)
    assert sample["energy"].shape == (1, 1)
    explicit_jacobian = torch.from_numpy(
        np.load(data_directory / "dq_dR_explicit.npy")
    )
    explicit_force = torch.from_numpy(
        np.load(data_directory / "f_corr_explicit_target.npy")
    )
    assert explicit_jacobian.shape == (1, 2, 3, 2, 1)
    assert explicit_force.shape == (1, 2, 3)

    descriptors = sample["descriptor"].clone().requires_grad_(True)
    predicted_energy = model(descriptors)
    (energy_descriptor_gradient,) = torch.autograd.grad(
        predicted_energy,
        descriptors,
        grad_outputs=torch.ones_like(predicted_energy),
    )
    predicted_force = -torch.einsum(
        "...bxap,...ap->...bx",
        explicit_jacobian,
        energy_descriptor_gradient,
    )

    torch.testing.assert_close(predicted_energy, sample["energy"], atol=1.0e-10, rtol=0.0)
    torch.testing.assert_close(predicted_force, explicit_force, atol=1.0e-10, rtol=0.0)


def test_energy_training_checkpoint_can_be_reloaded_for_deepks_scf(tmp_path):
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
            "force_factor": 0.0,
            "start_lr": 1.0e-4,
        },
        seed=11,
        device="cpu",
    )

    loaded = CorrNet.load(checkpoint).double().eval()
    selected = select_fields(["e_tot", "descriptor", "converged", "f_tot"])
    _, result = solve_molecule(
        molecule,
        loaded,
        selected,
        conv_tol=1.0e-10,
        max_cycle=50,
    )

    assert result["converged"]
    assert np.isfinite(result["e_tot"])
    assert np.isfinite(result["descriptor"]).all()
    assert np.isfinite(result["f_tot"]).all()
