from dataclasses import dataclass

import numpy as np
import pytest
import torch
from pyscf import dft, gto

from deepks.deephf import UKSDeePHF
from deepks.model.model import CorrNet


COORDINATES = np.array(
    [
        [0.13, -0.21, 0.07],
        [1.731385881594, 0.240389779198, -0.280303161599],
        [-0.367019388329, 0.780700019240, 1.613206142906],
    ],
    dtype=np.float64,
)
ATOMS = ("N", "H", "H")
PROJECTOR_BASIS = [[0, [0.8, 1.0]], [1, [0.3, 1.0]]]
STEPS = (1.0e-3, 3.0e-4, 1.0e-4)


def make_molecule(coordinates=COORDINATES):
    return gto.M(
        atom=list(zip(ATOMS, np.asarray(coordinates, dtype=np.float64))),
        basis="sto-3g",
        unit="Bohr",
        spin=1,
        charge=0,
        symmetry=False,
        cart=False,
        verbose=0,
    )


def run_fresh_uks(coordinates=COORDINATES):
    reference = dft.UKS(make_molecule(coordinates))
    reference.xc = "LDA_X + LDA_C_VWN"
    reference.grids.atom_grid = {"N": (20, 50), "H": (20, 50)}
    reference.grids.prune = None
    reference.grids.alignment = 1
    reference.grids.cutoff = 1.0e-15
    reference.grids.build(with_non0tab=True, sort_grids=False)
    reference.small_rho_cutoff = 0.0
    reference.conv_tol = 1.0e-13
    reference.conv_tol_grad = 1.0e-10
    reference.conv_tol_cpscf = 1.0e-12
    reference.max_cycle = 100
    reference.kernel()
    assert reference.converged
    return reference


def make_model():
    model = CorrNet(
        input_dim=4,
        hidden_sizes=(3,),
        actv_fn="tanh",
        use_resnet=False,
        proj_basis=PROJECTOR_BASIS,
        input_shift=[0.17, -0.23, 0.11, -0.07],
        input_scale=[0.71, 1.29, 0.83, 1.17],
        output_scale=1.37,
    ).double()
    with torch.no_grad():
        model.linear.weight[:] = torch.tensor([[0.037, -0.021, 0.013, 0.029]], dtype=torch.float64)
        model.linear.bias.fill_(0.011)
        first, output = model.densenet.layers
        first.weight[:] = torch.tensor([[0.31, -0.22, 0.17, 0.09], [0.17, 0.29, -0.14, 0.23], [-0.26, 0.14, 0.28, -0.19]], dtype=torch.float64)
        first.bias[:] = torch.tensor([0.03, -0.04, 0.02], dtype=torch.float64)
        output.weight[:] = torch.tensor([[0.27, -0.19, 0.16]], dtype=torch.float64)
        output.bias.fill_(0.021)
        model.energy_const.fill_(0.019)
    return model.eval()


@dataclass(frozen=True)
class UKSCase:
    reference: object
    model: object
    method: UKSDeePHF
    response: object
    direct_gradient: np.ndarray
    zvector_gradient: np.ndarray
    displaced: dict

    def finite_difference(self, field: str, step: float):
        result = np.empty((3, 3, *np.asarray(self.displaced[(step, 0, 0, 1)][field]).shape), dtype=np.float64)
        for atom in range(3):
            for axis in range(3):
                plus = np.asarray(self.displaced[(step, atom, axis, 1)][field])
                minus = np.asarray(self.displaced[(step, atom, axis, -1)][field])
                result[atom, axis] = (plus - minus) / (2.0 * step)
        return result


@pytest.fixture(scope="session")
def uks_case():
    reference = run_fresh_uks()
    model = make_model()
    method = UKSDeePHF(reference, model, projector_basis=PROJECTOR_BASIS)
    response = method.response()
    direct_gradient = method.gradient(backend="direct")
    zvector_gradient = method.gradient(backend="zvector")
    displaced = {}
    for step in STEPS:
        for atom in range(3):
            for axis in range(3):
                for sign in (-1, 1):
                    coordinates = COORDINATES.copy()
                    coordinates[atom, axis] += sign * step
                    displaced_reference = run_fresh_uks(coordinates)
                    displaced_method = UKSDeePHF(displaced_reference, model, projector_basis=PROJECTOR_BASIS)
                    displaced[(step, atom, axis, sign)] = {
                        "alpha_density": np.asarray(displaced_reference.make_rdm1()[0]),
                        "beta_density": np.asarray(displaced_reference.make_rdm1()[1]),
                        "density": np.asarray(displaced_reference.make_rdm1()).sum(axis=0),
                        "descriptor": displaced_method.descriptor(),
                        "energy": np.asarray(displaced_method.kernel()),
                    }
    return UKSCase(reference, model, method, response, direct_gradient, zvector_gradient, displaced)
