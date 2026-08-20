from dataclasses import dataclass

import numpy as np
import pytest
import torch
from pyscf import ao2mo, gto, scf

from deepks.deephf import DeePHF
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
FINITE_DIFFERENCE_STEPS = (1.0e-3, 3.0e-4, 1.0e-4)


def make_oracle_molecule(coordinates=ORACLE_COORDINATES):
    return gto.M(
        atom=list(zip(ORACLE_ATOMS, np.asarray(coordinates, dtype=np.float64))),
        basis="sto-3g",
        unit="Bohr",
        symmetry=False,
        cart=False,
        verbose=0,
    )


def run_fresh_rhf(coordinates=ORACLE_COORDINATES):
    reference = scf.RHF(make_oracle_molecule(coordinates))
    reference.conv_tol = 1.0e-13
    reference.conv_tol_grad = 1.0e-10
    reference.conv_tol_cpscf = 1.0e-12
    reference.max_cycle = 100
    reference.kernel()
    assert reference.converged
    return reference


def make_nonlinear_model():
    model = CorrNet(
        input_dim=4,
        hidden_sizes=(3,),
        actv_fn="tanh",
        use_resnet=False,
        proj_basis=ORACLE_PROJECTOR_BASIS,
        input_shift=[0.17, -0.23, 0.11, -0.07],
        input_scale=[0.71, 1.29, 0.83, 1.17],
        output_scale=1.37,
    ).double()
    with torch.no_grad():
        model.linear.weight[:] = torch.tensor(
            [[0.037, -0.021, 0.013, 0.029]],
            dtype=torch.float64,
        )
        model.linear.bias.fill_(0.011)
        first_layer, output_layer = model.densenet.layers
        first_layer.weight[:] = torch.tensor(
            [
                [0.31, -0.22, 0.17, 0.09],
                [0.17, 0.29, -0.14, 0.23],
                [-0.26, 0.14, 0.28, -0.19],
            ],
            dtype=torch.float64,
        )
        first_layer.bias[:] = torch.tensor(
            [0.03, -0.04, 0.02],
            dtype=torch.float64,
        )
        output_layer.weight[:] = torch.tensor(
            [[0.27, -0.19, 0.16]],
            dtype=torch.float64,
        )
        output_layer.bias.fill_(0.021)
        model.energy_const.fill_(0.019)
    return model.eval()


def independent_rhf_operator(reference):
    coefficient = np.asarray(reference.mo_coeff)
    occupied = np.asarray(reference.mo_occ) > 0
    virtual = np.asarray(reference.mo_occ) == 0
    occupied_coefficients = coefficient[:, occupied]
    virtual_coefficients = coefficient[:, virtual]
    n_occupied = int(np.count_nonzero(occupied))
    n_virtual = int(np.count_nonzero(virtual))
    coulomb = ao2mo.general(
        reference.mol,
        (
            virtual_coefficients,
            occupied_coefficients,
            virtual_coefficients,
            occupied_coefficients,
        ),
        compact=False,
    ).reshape(n_virtual, n_occupied, n_virtual, n_occupied)
    exchange_abij = ao2mo.general(
        reference.mol,
        (
            virtual_coefficients,
            virtual_coefficients,
            occupied_coefficients,
            occupied_coefficients,
        ),
        compact=False,
    ).reshape(n_virtual, n_virtual, n_occupied, n_occupied)
    exchange_ajib = ao2mo.general(
        reference.mol,
        (
            virtual_coefficients,
            occupied_coefficients,
            occupied_coefficients,
            virtual_coefficients,
        ),
        compact=False,
    ).reshape(n_virtual, n_occupied, n_occupied, n_virtual)
    operator = (
        4.0 * coulomb
        - exchange_abij.transpose(0, 2, 1, 3)
        - exchange_ajib.transpose(0, 2, 3, 1)
    ).reshape(n_virtual * n_occupied, n_virtual * n_occupied)
    orbital_gaps = (
        np.asarray(reference.mo_energy)[virtual, None]
        - np.asarray(reference.mo_energy)[occupied]
    )
    return operator + np.diag(orbital_gaps.reshape(-1))


@dataclass(frozen=True)
class ZVectorAlgebraCase:
    coordinates: np.ndarray
    steps: tuple[float, ...]
    reference: object
    model: torch.nn.Module
    method: DeePHF
    direct_response: object
    independent_operator: np.ndarray
    displaced_total_energies: dict[tuple[float, int, int, int], float]

    def total_energy_finite_difference(self, step: float) -> np.ndarray:
        result = np.empty((self.reference.mol.natm, 3), dtype=np.float64)
        for atom_index in range(self.reference.mol.natm):
            for coordinate_index in range(3):
                forward = self.displaced_total_energies[
                    (step, atom_index, coordinate_index, 1)
                ]
                backward = self.displaced_total_energies[
                    (step, atom_index, coordinate_index, -1)
                ]
                result[atom_index, coordinate_index] = (
                    forward - backward
                ) / (2.0 * step)
        return result


@pytest.fixture(scope="session")
def zvector_algebra_case():
    reference = run_fresh_rhf()
    model = make_nonlinear_model()
    method = DeePHF(
        reference,
        model,
        projector_basis=ORACLE_PROJECTOR_BASIS,
    )
    method.kernel()
    direct_response = method.response()
    reference_occupations = np.asarray(reference.mo_occ).copy()
    reference_ao_labels = tuple(reference.mol.ao_labels())
    displaced_total_energies = {}
    for step in FINITE_DIFFERENCE_STEPS:
        for atom_index in range(reference.mol.natm):
            for coordinate_index in range(3):
                for direction in (-1, 1):
                    coordinates = ORACLE_COORDINATES.copy()
                    coordinates[atom_index, coordinate_index] += direction * step
                    displaced_reference = run_fresh_rhf(coordinates)
                    np.testing.assert_array_equal(
                        displaced_reference.mo_occ,
                        reference_occupations,
                    )
                    assert tuple(displaced_reference.mol.ao_labels()) == reference_ao_labels
                    displaced_method = DeePHF(
                        displaced_reference,
                        model,
                        projector_basis=ORACLE_PROJECTOR_BASIS,
                    )
                    displaced_total_energies[
                        (step, atom_index, coordinate_index, direction)
                    ] = float(displaced_method.kernel())
    return ZVectorAlgebraCase(
        coordinates=ORACLE_COORDINATES.copy(),
        steps=FINITE_DIFFERENCE_STEPS,
        reference=reference,
        model=model,
        method=method,
        direct_response=direct_response,
        independent_operator=independent_rhf_operator(reference),
        displaced_total_energies=displaced_total_energies,
    )
