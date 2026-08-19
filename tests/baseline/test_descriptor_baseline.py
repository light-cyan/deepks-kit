import numpy as np
import torch
from pyscf import gto, scf

from deepks.scf.scf import DSCF, t_make_eig, t_make_grad_eig_dm, t_make_pdm


SYNTHETIC_DENSITY = torch.tensor(
    [[1.2, 0.1, -0.2], [0.1, 0.8, 0.05], [-0.2, 0.05, 0.6]],
    dtype=torch.float64,
)
SYNTHETIC_PROJECTION = torch.tensor(
    [
        [[0.8, 0.1], [0.2, -0.4]],
        [[0.3, 0.5], [-0.7, 0.2]],
        [[-0.2, 0.4], [0.6, 0.3]],
    ],
    dtype=torch.float64,
)
EXPECTED_PROJECTED_DENSITY = torch.tensor(
    [
        [[0.9700, 0.1520], [0.1520, 0.3220]],
        [[0.5380, -0.0365], [-0.0365, 0.3160]],
    ],
    dtype=torch.float64,
)
EXPECTED_DESCRIPTOR_VALUES = torch.tensor(
    [
        [0.288117337664983, 1.003882662335017],
        [0.310152877656315, 0.543847122343684],
    ],
    dtype=torch.float64,
)
MOLECULE_COORDINATES = np.array([[0.10, -0.20, 0.05], [0.70, 0.30, 0.40]])
PROJECTOR_BASIS = "sto-3g@H"
EXPECTED_FIXED_DENSITY_COORDINATE_JACOBIAN = np.array(
    [
        [
            [[0.138497162818611], [0.754781030111533]],
            [[0.115414302348842], [0.628984191759611]],
            [[0.080790011644190], [0.440288934231728]],
        ],
        [
            [[-0.138497162818611], [-0.754781030111533]],
            [[-0.115414302348842], [-0.628984191759611]],
            [[-0.080790011644190], [-0.440288934231728]],
        ],
    ]
)


def make_molecule(coordinates):
    return gto.M(
        atom=[("He", coordinates[0]), ("H", coordinates[1])],
        basis="sto-3g",
        charge=1,
        spin=0,
        unit="Bohr",
        verbose=0,
    )


def test_projected_density_and_descriptor_values_are_characterized():
    projected_density = t_make_pdm(
        SYNTHETIC_DENSITY, (SYNTHETIC_PROJECTION,)
    )[0]
    descriptor_values = t_make_eig(
        SYNTHETIC_DENSITY, (SYNTHETIC_PROJECTION,)
    )

    torch.testing.assert_close(
        projected_density,
        EXPECTED_PROJECTED_DENSITY,
        rtol=1e-13,
        atol=1e-13,
    )
    torch.testing.assert_close(
        descriptor_values,
        EXPECTED_DESCRIPTOR_VALUES,
        rtol=1e-13,
        atol=1e-13,
    )
    torch.testing.assert_close(
        descriptor_values,
        torch.linalg.eigvalsh(projected_density),
        rtol=1e-13,
        atol=1e-13,
    )
    torch.testing.assert_close(
        descriptor_values.sum(dim=-1),
        projected_density.diagonal(dim1=-2, dim2=-1).sum(dim=-1),
        rtol=1e-13,
        atol=1e-13,
    )


def test_descriptor_ao_density_jacobian_matches_directional_differences():
    jacobian = t_make_grad_eig_dm(
        SYNTHETIC_DENSITY.clone(), (SYNTHETIC_PROJECTION,)
    )
    directions = (
        torch.tensor(
            [[0.4, -0.2, 0.1], [-0.2, 0.3, 0.05], [0.1, 0.05, -0.6]],
            dtype=torch.float64,
        ),
        torch.tensor(
            [[-0.1, 0.7, -0.3], [0.7, 0.2, 0.4], [-0.3, 0.4, 0.5]],
            dtype=torch.float64,
        ),
    )

    assert jacobian.shape == (2, 2, 3, 3)
    torch.testing.assert_close(
        jacobian,
        jacobian.transpose(-1, -2),
        rtol=1e-13,
        atol=1e-13,
    )

    step = 1e-5
    for direction in directions:
        forward = t_make_eig(
            SYNTHETIC_DENSITY + step * direction,
            (SYNTHETIC_PROJECTION,),
        )
        backward = t_make_eig(
            SYNTHETIC_DENSITY - step * direction,
            (SYNTHETIC_PROJECTION,),
        )
        finite_difference = (forward - backward) / (2 * step)
        analytic_directional_derivative = torch.einsum(
            "avrs,rs->av", jacobian, direction
        )
        torch.testing.assert_close(
            analytic_directional_derivative,
            finite_difference,
            rtol=1e-8,
            atol=5e-10,
        )

    projected_trace_jacobian = torch.einsum(
        "rap,sap->ars", SYNTHETIC_PROJECTION, SYNTHETIC_PROJECTION
    )
    torch.testing.assert_close(
        jacobian.sum(dim=1),
        projected_trace_jacobian,
        rtol=1e-13,
        atol=1e-13,
    )


def test_fixed_density_coordinate_jacobian_matches_central_difference():
    molecule = make_molecule(MOLECULE_COORDINATES)
    reference = scf.RHF(molecule)
    reference.conv_tol = 1e-12
    reference.kernel()
    assert reference.converged
    fixed_ao_density = np.asarray(reference.make_rdm1())

    method = DSCF(molecule, None, proj_basis=PROJECTOR_BASIS)
    analytic = method.nuc_grad_method().make_grad_eig_x(fixed_ao_density)

    np.testing.assert_allclose(
        analytic,
        EXPECTED_FIXED_DENSITY_COORDINATE_JACOBIAN,
        rtol=2e-11,
        atol=2e-11,
    )
    np.testing.assert_allclose(
        analytic.sum(axis=0),
        np.zeros_like(analytic[0]),
        rtol=0,
        atol=2e-13,
    )

    step = 5e-5
    finite_difference = np.empty_like(analytic)
    for atom_index in range(molecule.natm):
        for coordinate_index in range(3):
            forward_coordinates = MOLECULE_COORDINATES.copy()
            backward_coordinates = MOLECULE_COORDINATES.copy()
            forward_coordinates[atom_index, coordinate_index] += step
            backward_coordinates[atom_index, coordinate_index] -= step
            forward = DSCF(
                make_molecule(forward_coordinates),
                None,
                proj_basis=PROJECTOR_BASIS,
            ).make_eig(fixed_ao_density)
            backward = DSCF(
                make_molecule(backward_coordinates),
                None,
                proj_basis=PROJECTOR_BASIS,
            ).make_eig(fixed_ao_density)
            finite_difference[atom_index, coordinate_index] = (
                forward - backward
            ) / (2 * step)

    np.testing.assert_allclose(
        analytic,
        finite_difference,
        rtol=1e-8,
        atol=3e-9,
    )
