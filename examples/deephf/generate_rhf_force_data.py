"""Generate a deterministic strict RHF relaxed-force training example."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from pyscf import gto, scf

from deepks.deephf import DeePHF, write_rhf_force_dataset
from deepks.model.model import CorrNet


PROJECTOR_BASIS = [[0, [0.8, 1.0]], [1, [0.3, 1.0]]]
ATOMS = ("O", "H", "H")
TRAIN_COORDINATES = np.array(
    [[0.13, -0.21, 0.07], [1.51, 0.12, -0.19], [-0.46, 1.62, 0.31]],
    dtype=np.float64,
)
VALIDATION_COORDINATES = np.array(
    [[0.15, -0.21, 0.07], [1.51, 0.10, -0.19], [-0.46, 1.62, 0.29]],
    dtype=np.float64,
)
TEACHER_WEIGHTS = np.array([0.037, -0.021, 0.013, 0.029], dtype=np.float64)


def make_reference(coordinates: np.ndarray):
    molecule = gto.M(
        atom=list(zip(ATOMS, coordinates)),
        basis="sto-3g",
        unit="Bohr",
        symmetry=False,
        cart=False,
        verbose=0,
    )
    reference = scf.RHF(molecule)
    reference.conv_tol = 1.0e-13
    reference.conv_tol_grad = 1.0e-10
    reference.conv_tol_cpscf = 1.0e-12
    reference.max_cycle = 100
    reference.kernel()
    if not reference.converged:
        raise RuntimeError("the example RHF reference did not converge")
    return reference


def make_teacher() -> CorrNet:
    model = CorrNet(
        input_dim=4,
        hidden_sizes=(2,),
        proj_basis=PROJECTOR_BASIS,
    ).double()
    with torch.no_grad():
        model.linear.weight.copy_(torch.as_tensor(TEACHER_WEIGHTS).reshape(1, 4))
        model.linear.bias.fill_(0.011)
        for parameter in model.densenet.parameters():
            parameter.zero_()
    return model.eval()


def teacher_targets(reference, teacher: CorrNet) -> tuple[np.float64, np.ndarray]:
    method = DeePHF(
        reference,
        teacher,
        projector_basis=PROJECTOR_BASIS,
    )
    energy = np.float64(method.kernel())
    gradient = method.nuc_grad_method(backend="direct").run()
    force = np.ascontiguousarray(-gradient.de_full, dtype=np.float64)
    return energy, force


def write_example_dataset(directory: Path, coordinates: np.ndarray, teacher: CorrNet):
    reference = make_reference(coordinates)
    energy, force = teacher_targets(reference, teacher)
    return write_rhf_force_dataset(
        directory,
        reference,
        projector_basis=PROJECTOR_BASIS,
        e_target=energy,
        f_target=force,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate deterministic strict RHF relaxed-force data."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
        help="parent directory for the train and validation datasets",
    )
    args = parser.parse_args()
    teacher = make_teacher()
    train_contract = write_example_dataset(
        args.output / "train",
        TRAIN_COORDINATES,
        teacher,
    )
    validation_contract = write_example_dataset(
        args.output / "validation",
        VALIDATION_COORDINATES,
        teacher,
    )
    if (
        train_contract.compatibility_fingerprint
        != validation_contract.compatibility_fingerprint
    ):
        raise RuntimeError("the train and validation force contracts differ")
    print(f"wrote strict force data under {args.output}")
    print(f"compatibility fingerprint: {train_contract.compatibility_fingerprint}")


if __name__ == "__main__":
    main()
