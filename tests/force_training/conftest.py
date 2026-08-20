from dataclasses import dataclass

import numpy as np
import pytest
import torch
from pyscf import gto, scf

from deepks.deephf import DeePHF, generate_rhf_force_frame
from deepks.model.model import CorrNet


ORACLE_COORDINATES = np.array(
    [
        [0.13, -0.21, 0.07],
        [1.51, 0.12, -0.19],
        [-0.46, 1.62, 0.31],
    ],
    dtype=np.float64,
)
ORACLE_ATOMS = ("O", "H", "H")
ORACLE_PROJECTOR_BASIS = [[0, [0.8, 1.0]], [1, [0.3, 1.0]]]
ORACLE_MODEL_WEIGHTS = np.array([0.037, -0.021, 0.013, 0.029])
FINITE_DIFFERENCE_STEP = 1.0e-4
FINITE_DIFFERENCE_ATOM = 0
FINITE_DIFFERENCE_COORDINATE = 0


def make_oracle_molecule(coordinates=ORACLE_COORDINATES):
    return gto.M(
        atom=list(zip(ORACLE_ATOMS, np.asarray(coordinates))),
        basis="sto-3g",
        unit="Bohr",
        symmetry=False,
        cart=False,
        verbose=0,
    )


def run_oracle_rhf(coordinates=ORACLE_COORDINATES):
    reference = scf.RHF(make_oracle_molecule(coordinates))
    reference.conv_tol = 1.0e-13
    reference.conv_tol_grad = 1.0e-10
    reference.conv_tol_cpscf = 1.0e-12
    reference.max_cycle = 100
    reference.kernel()
    assert reference.converged
    return reference


def make_oracle_model():
    model = CorrNet(
        input_dim=4,
        hidden_sizes=(2,),
        proj_basis=ORACLE_PROJECTOR_BASIS,
    ).double()
    with torch.no_grad():
        model.linear.weight.copy_(
            torch.as_tensor(ORACLE_MODEL_WEIGHTS).reshape(1, 4)
        )
        model.linear.bias.fill_(0.011)
        for parameter in model.densenet.parameters():
            parameter.zero_()
    return model.eval()


@dataclass(frozen=True)
class ForceGenerationCase:
    reference: object
    teacher_method: DeePHF
    teacher_gradient: object
    target_energy: np.float64
    target_force: np.ndarray
    forward_reference: object
    backward_reference: object
    forward_descriptor: np.ndarray
    backward_descriptor: np.ndarray
    forward_correction_energy: float
    backward_correction_energy: float

    @property
    def descriptor_finite_difference(self):
        return (
            self.forward_descriptor - self.backward_descriptor
        ) / (2.0 * FINITE_DIFFERENCE_STEP)

    @property
    def correction_energy_finite_difference(self):
        return (
            self.forward_correction_energy - self.backward_correction_energy
        ) / (2.0 * FINITE_DIFFERENCE_STEP)


@pytest.fixture(scope="session")
def force_generation_case():
    reference = run_oracle_rhf()
    model = make_oracle_model()
    teacher_method = DeePHF(
        reference,
        model,
        projector_basis=ORACLE_PROJECTOR_BASIS,
    )
    target_energy = np.float64(teacher_method.kernel())
    teacher_gradient = teacher_method.nuc_grad_method().run()
    target_force = np.asarray(-teacher_gradient.de_full, dtype=np.float64)

    displaced = []
    for direction in (1, -1):
        coordinates = ORACLE_COORDINATES.copy()
        coordinates[
            FINITE_DIFFERENCE_ATOM,
            FINITE_DIFFERENCE_COORDINATE,
        ] += direction * FINITE_DIFFERENCE_STEP
        displaced_reference = run_oracle_rhf(coordinates)
        displaced_method = DeePHF(
            displaced_reference,
            model,
            projector_basis=ORACLE_PROJECTOR_BASIS,
        )
        displaced_method.kernel()
        displaced.append(
            (
                displaced_reference,
                displaced_method.descriptor(),
                displaced_method.e_corr,
            )
        )

    return ForceGenerationCase(
        reference=reference,
        teacher_method=teacher_method,
        teacher_gradient=teacher_gradient,
        target_energy=target_energy,
        target_force=target_force,
        forward_reference=displaced[0][0],
        backward_reference=displaced[1][0],
        forward_descriptor=displaced[0][1],
        backward_descriptor=displaced[1][1],
        forward_correction_energy=displaced[0][2],
        backward_correction_energy=displaced[1][2],
    )


@pytest.fixture(scope="session")
def generated_force_frame(force_generation_case):
    case = force_generation_case
    return generate_rhf_force_frame(
        case.reference,
        projector_basis=ORACLE_PROJECTOR_BASIS,
        e_target=case.target_energy,
        f_target=case.target_force,
    )
