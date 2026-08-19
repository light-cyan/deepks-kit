from dataclasses import dataclass

import numpy as np
import pytest
import torch
from pyscf import gto, scf

from deepks.deephf import DeePHF
from deepks.model.model import CorrNet


ORACLE_COORDINATES = np.array(
    [
        [0.13, -0.21, 0.07],
        [1.51, 0.12, -0.19],
        [-0.46, 1.62, 0.31],
    ]
)
ORACLE_ATOMS = ("O", "H", "H")
ORACLE_PROJECTOR_BASIS = [[0, [0.8, 1.0]], [1, [0.3, 1.0]]]
ORACLE_MODEL_WEIGHTS = np.array([0.037, -0.021, 0.013, 0.029])
ORACLE_STEPS = (1.0e-3, 3.0e-4, 1.0e-4)


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


def make_oracle_model(weights=ORACLE_MODEL_WEIGHTS, bias=0.011):
    model = CorrNet(
        input_dim=4,
        hidden_sizes=(2,),
        proj_basis=ORACLE_PROJECTOR_BASIS,
    ).double()
    weights = torch.as_tensor(weights, dtype=torch.float64).reshape(1, 4)
    with torch.no_grad():
        model.linear.weight.copy_(weights)
        model.linear.bias.fill_(bias)
        for parameter in model.densenet.parameters():
            parameter.zero_()
    return model.eval()


@dataclass(frozen=True)
class OracleState:
    density: np.ndarray
    overlap: np.ndarray
    descriptor: np.ndarray
    total_energy: float


@dataclass(frozen=True)
class RHFOracleCase:
    coordinates: np.ndarray
    steps: tuple[float, ...]
    reference: object
    method: DeePHF
    response: object
    model: torch.nn.Module
    displaced: dict[tuple[float, int, int, int], OracleState]
    minimum_descriptor_gap: float

    def finite_difference(self, field: str, step: float) -> np.ndarray:
        sample = np.asarray(getattr(self.displaced[(step, 0, 0, 1)], field))
        result = np.empty((self.reference.mol.natm, 3, *sample.shape))
        for atom_index in range(self.reference.mol.natm):
            for coordinate_index in range(3):
                forward = np.asarray(
                    getattr(
                        self.displaced[(step, atom_index, coordinate_index, 1)],
                        field,
                    )
                )
                backward = np.asarray(
                    getattr(
                        self.displaced[(step, atom_index, coordinate_index, -1)],
                        field,
                    )
                )
                result[atom_index, coordinate_index] = (
                    forward - backward
                ) / (2.0 * step)
        return result


@pytest.fixture(scope="session")
def rhf_oracle_case():
    reference = run_oracle_rhf()
    model = make_oracle_model()
    method = DeePHF(
        reference,
        model,
        projector_basis=ORACLE_PROJECTOR_BASIS,
    )
    method.kernel()
    response = method.response()
    reference_occupations = np.asarray(reference.mo_occ).copy()
    reference_ao_labels = tuple(reference.mol.ao_labels())
    displaced = {}
    minimum_descriptor_gap = np.inf
    for step in ORACLE_STEPS:
        for atom_index in range(reference.mol.natm):
            for coordinate_index in range(3):
                for direction in (-1, 1):
                    coordinates = ORACLE_COORDINATES.copy()
                    coordinates[atom_index, coordinate_index] += direction * step
                    displaced_reference = run_oracle_rhf(coordinates)
                    np.testing.assert_array_equal(
                        displaced_reference.mo_occ,
                        reference_occupations,
                    )
                    assert tuple(displaced_reference.mol.ao_labels()) == reference_ao_labels
                    occupied = displaced_reference.mo_occ > 0
                    virtual = displaced_reference.mo_occ == 0
                    orbital_gap = np.min(
                        displaced_reference.mo_energy[virtual, None]
                        - displaced_reference.mo_energy[occupied]
                    )
                    assert orbital_gap > 0.8
                    displaced_method = DeePHF(
                        displaced_reference,
                        model,
                        projector_basis=ORACLE_PROJECTOR_BASIS,
                    )
                    total_energy = displaced_method.kernel()
                    descriptor = displaced_method.descriptor()
                    p_block = descriptor[:, 1:]
                    minimum_descriptor_gap = min(
                        minimum_descriptor_gap,
                        float(np.min(np.diff(p_block, axis=-1))),
                    )
                    displaced[(step, atom_index, coordinate_index, direction)] = (
                        OracleState(
                            density=displaced_method.ao_density(),
                            overlap=np.asarray(displaced_reference.get_ovlp()),
                            descriptor=descriptor,
                            total_energy=total_energy,
                        )
                    )
    return RHFOracleCase(
        coordinates=ORACLE_COORDINATES.copy(),
        steps=ORACLE_STEPS,
        reference=reference,
        method=method,
        response=response,
        model=model,
        displaced=displaced,
        minimum_descriptor_gap=minimum_descriptor_gap,
    )
