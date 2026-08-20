from dataclasses import dataclass

import numpy as np
import pytest
import torch
from pyscf import gto, scf

from deepks.deephf import UHFDeePHF
from deepks.descriptor import AtomicDensityDescriptor
from deepks.model.model import CorrNet


ORACLE_COORDINATES = np.array(
    [
        [0.130000000000, -0.210000000000, 0.070000000000],
        [1.731385881594, 0.240389779198, -0.280303161599],
        [-0.367019388329, 0.780700019240, 1.613206142906],
    ],
    dtype=np.float64,
)
ORACLE_ATOMS = ("N", "H", "H")
ORACLE_PROJECTOR_BASIS = [[0, [0.8, 1.0]], [1, [0.3, 1.0]]]
FINITE_DIFFERENCE_STEPS = (1.0e-3, 3.0e-4, 1.0e-4)


def make_oracle_molecule(coordinates=ORACLE_COORDINATES):
    return gto.M(
        atom=list(
            zip(
                ORACLE_ATOMS,
                np.asarray(coordinates, dtype=np.float64),
            )
        ),
        basis="sto-3g",
        unit="Bohr",
        charge=0,
        spin=1,
        symmetry=False,
        cart=False,
        verbose=0,
    )


def run_fresh_uhf(coordinates=ORACLE_COORDINATES):
    reference = scf.UHF(make_oracle_molecule(coordinates))
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


def occupied_virtual_shapes(reference):
    occupations = np.asarray(reference.mo_occ)
    return tuple(
        (
            int(np.count_nonzero(spin_occupations == 0.0)),
            int(np.count_nonzero(spin_occupations > 0.0)),
        )
        for spin_occupations in occupations
    )


def occupied_subspace_minimum_singular_values(reference, displaced_reference):
    cross_overlap = np.asarray(
        gto.intor_cross(
            "int1e_ovlp",
            reference.mol,
            displaced_reference.mol,
        )
    )
    minimum_singular_values = []
    for spin in range(2):
        reference_occupied = np.asarray(reference.mo_occ[spin]) > 0.0
        displaced_occupied = (
            np.asarray(displaced_reference.mo_occ[spin]) > 0.0
        )
        occupied_overlap = (
            np.asarray(reference.mo_coeff[spin])[:, reference_occupied].T
            @ cross_overlap
            @ np.asarray(displaced_reference.mo_coeff[spin])[
                :, displaced_occupied
            ]
        )
        singular_values = np.linalg.svd(
            occupied_overlap,
            compute_uv=False,
        )
        minimum_singular_values.append(float(np.min(singular_values)))
    return tuple(minimum_singular_values)


def independent_coupled_uhf_operator(reference):
    coefficients = np.asarray(reference.mo_coeff)
    occupations = np.asarray(reference.mo_occ)
    occupied = occupations > 0.0
    occupied_coefficients = [
        coefficients[spin][:, occupied[spin]] for spin in range(2)
    ]
    virtual_coefficients = [
        coefficients[spin][:, ~occupied[spin]] for spin in range(2)
    ]
    shapes = occupied_virtual_shapes(reference)
    dimensions = tuple(nvirtual * noccupied for nvirtual, noccupied in shapes)
    offsets = (0, dimensions[0])
    operator = np.zeros(
        (sum(dimensions), sum(dimensions)),
        dtype=np.float64,
    )
    electron_repulsion = np.asarray(
        reference.mol.intor("int2e", aosym="s1")
    )
    for source_spin in range(2):
        source_virtual = virtual_coefficients[source_spin]
        source_occupied = occupied_coefficients[source_spin]
        for source_index in range(dimensions[source_spin]):
            orbital_response = np.zeros(
                shapes[source_spin],
                dtype=np.float64,
            )
            orbital_response.reshape(-1)[source_index] = 1.0
            one_sided_density = (
                source_virtual
                @ orbital_response
                @ source_occupied.T
            )
            density_response = one_sided_density + one_sided_density.T
            coulomb = np.einsum(
                "mnkl,lk->mn",
                electron_repulsion,
                density_response,
            )
            exchange = np.einsum(
                "mkln,lk->mn",
                electron_repulsion,
                density_response,
            )
            for target_spin in range(2):
                induced_potential = coulomb.copy()
                if target_spin == source_spin:
                    induced_potential -= exchange
                target_block = (
                    virtual_coefficients[target_spin].T
                    @ induced_potential
                    @ occupied_coefficients[target_spin]
                )
                row_start = offsets[target_spin]
                row_stop = row_start + dimensions[target_spin]
                column = offsets[source_spin] + source_index
                operator[row_start:row_stop, column] = target_block.reshape(-1)
    for spin in range(2):
        orbital_gaps = (
            np.asarray(reference.mo_energy[spin])[~occupied[spin], None]
            - np.asarray(reference.mo_energy[spin])[occupied[spin]]
        )
        start = offsets[spin]
        stop = start + dimensions[spin]
        operator[start:stop, start:stop] += np.diag(
            orbital_gaps.reshape(-1)
        )
    return operator


def independent_overlap_derivative(reference):
    molecule = reference.mol
    derivative_integrals = -np.asarray(
        molecule.intor("int1e_ipovlp", comp=3)
    )
    ao_slices = molecule.aoslice_by_atom()
    derivative = np.zeros(
        (molecule.natm, 3, molecule.nao, molecule.nao),
        dtype=np.float64,
    )
    for atom_index in range(molecule.natm):
        ao_start, ao_stop = ao_slices[atom_index, 2:]
        derivative[atom_index, :, ao_start:ao_stop] += (
            derivative_integrals[:, ao_start:ao_stop]
        )
        derivative[atom_index, :, :, ao_start:ao_stop] += (
            derivative_integrals[:, ao_start:ao_stop].transpose(0, 2, 1)
        )
    return derivative


def independent_metric_density_response(reference, overlap_derivative=None):
    if overlap_derivative is None:
        overlap_derivative = independent_overlap_derivative(reference)
    overlap_derivative = np.asarray(overlap_derivative)
    coefficients = np.asarray(reference.mo_coeff)
    occupations = np.asarray(reference.mo_occ)
    result = np.empty(
        (
            reference.mol.natm,
            3,
            2,
            reference.mol.nao,
            reference.mol.nao,
        ),
        dtype=np.float64,
    )
    for spin in range(2):
        occupied_coefficients = coefficients[spin][
            :, occupations[spin] > 0.0
        ]
        density = occupied_coefficients @ occupied_coefficients.T
        result[:, :, spin] = -np.einsum(
            "ij,bxjk,kl->bxil",
            density,
            overlap_derivative,
            density,
        )
    return result


def independent_coupled_response_residual(
    reference,
    response,
    operator_matrix=None,
    metric_density_response=None,
):
    if operator_matrix is None:
        operator_matrix = independent_coupled_uhf_operator(reference)
    if metric_density_response is None:
        metric_density_response = independent_metric_density_response(reference)
    operator_matrix = np.asarray(operator_matrix)
    metric_density_response = np.asarray(metric_density_response)
    coefficients = np.asarray(reference.mo_coeff)
    occupations = np.asarray(reference.mo_occ)
    occupied = occupations > 0.0
    occupied_coefficients = [
        coefficients[spin][:, occupied[spin]] for spin in range(2)
    ]
    virtual_coefficients = [
        coefficients[spin][:, ~occupied[spin]] for spin in range(2)
    ]
    dimensions = tuple(
        virtual_coefficients[spin].shape[1]
        * occupied_coefficients[spin].shape[1]
        for spin in range(2)
    )
    electron_repulsion = np.asarray(
        reference.mol.intor("int2e", aosym="s1")
    )
    hamiltonian_derivatives = (
        response.alpha_hamiltonian_derivative,
        response.beta_hamiltonian_derivative,
    )
    mo_responses = (
        response.alpha_mo_response,
        response.beta_mo_response,
    )
    combined_residual = np.empty(
        (reference.mol.natm, 3, sum(dimensions)),
        dtype=np.float64,
    )
    for atom_index in range(reference.mol.natm):
        for coordinate_index in range(3):
            metric_spin = metric_density_response[
                atom_index,
                coordinate_index,
            ]
            metric_total = metric_spin.sum(axis=0)
            coulomb = np.einsum(
                "mnkl,lk->mn",
                electron_repulsion,
                metric_total,
            )
            fixed_terms = []
            orbital_response = []
            for spin in range(2):
                exchange = np.einsum(
                    "mkln,lk->mn",
                    electron_repulsion,
                    metric_spin[spin],
                )
                induced_metric_potential = coulomb - exchange
                occupied_energy = np.asarray(reference.mo_energy[spin])[
                    occupied[spin]
                ]
                projected_hamiltonian = (
                    virtual_coefficients[spin].T
                    @ (
                        hamiltonian_derivatives[spin][
                            atom_index,
                            coordinate_index,
                        ]
                        + induced_metric_potential
                    )
                    @ occupied_coefficients[spin]
                )
                projected_overlap = (
                    virtual_coefficients[spin].T
                    @ response.overlap_derivative[
                        atom_index,
                        coordinate_index,
                    ]
                    @ occupied_coefficients[spin]
                )
                fixed_terms.append(
                    (
                        projected_hamiltonian
                        - projected_overlap * occupied_energy
                    ).reshape(-1)
                )
                orbital_response.append(
                    mo_responses[spin][
                        atom_index,
                        coordinate_index,
                    ][~occupied[spin]].reshape(-1)
                )
            combined_residual[atom_index, coordinate_index] = (
                operator_matrix @ np.concatenate(orbital_response)
                + np.concatenate(fixed_terms)
            )
    alpha_stop = dimensions[0]
    alpha_shape = (
        reference.mol.natm,
        3,
        virtual_coefficients[0].shape[1],
        occupied_coefficients[0].shape[1],
    )
    beta_shape = (
        reference.mol.natm,
        3,
        virtual_coefficients[1].shape[1],
        occupied_coefficients[1].shape[1],
    )
    return (
        combined_residual[..., :alpha_stop].reshape(alpha_shape),
        combined_residual[..., alpha_stop:].reshape(beta_shape),
    )


@dataclass(frozen=True)
class UHFOracleState:
    density_alpha: np.ndarray
    density_beta: np.ndarray
    density_total: np.ndarray
    overlap: np.ndarray
    descriptor: np.ndarray
    base_energy: float
    correction_energy: float
    total_energy: float


def evaluate_oracle_state(reference, model):
    density_spin = np.asarray(reference.make_rdm1())
    density_total = density_spin.sum(axis=0)
    descriptor = AtomicDensityDescriptor(
        reference.mol,
        ORACLE_PROJECTOR_BASIS,
    ).descriptor(density_total)
    with torch.no_grad():
        correction_energy = float(
            model(torch.from_numpy(descriptor)).detach().cpu().item()
        )
    base_energy = float(reference.e_tot)
    return UHFOracleState(
        density_alpha=density_spin[0].copy(),
        density_beta=density_spin[1].copy(),
        density_total=density_total.copy(),
        overlap=np.asarray(reference.get_ovlp()).copy(),
        descriptor=descriptor.copy(),
        base_energy=base_energy,
        correction_energy=correction_energy,
        total_energy=base_energy + correction_energy,
    )


@dataclass(frozen=True)
class UHFOracleCase:
    coordinates: np.ndarray
    steps: tuple[float, ...]
    reference: object
    model: torch.nn.Module
    method: UHFDeePHF
    response: object
    gradient_driver: object
    gradient: np.ndarray
    state: UHFOracleState
    displaced: dict[tuple[float, int, int, int], UHFOracleState]
    occupied_subspace_minimum_singular_values: dict[
        tuple[float, int, int, int],
        tuple[float, float],
    ]
    coupled_operator: np.ndarray
    overlap_derivative: np.ndarray
    metric_density_response: np.ndarray
    minimum_alpha_orbital_gap: float
    minimum_beta_orbital_gap: float
    minimum_descriptor_gap: float

    def finite_difference(self, field: str, step: float) -> np.ndarray:
        sample = np.asarray(
            getattr(self.displaced[(step, 0, 0, 1)], field)
        )
        result = np.empty(
            (self.reference.mol.natm, 3, *sample.shape),
            dtype=np.float64,
        )
        for atom_index in range(self.reference.mol.natm):
            for coordinate_index in range(3):
                forward = np.asarray(
                    getattr(
                        self.displaced[
                            (step, atom_index, coordinate_index, 1)
                        ],
                        field,
                    )
                )
                backward = np.asarray(
                    getattr(
                        self.displaced[
                            (step, atom_index, coordinate_index, -1)
                        ],
                        field,
                    )
                )
                result[atom_index, coordinate_index] = (
                    forward - backward
                ) / (2.0 * step)
        return result

    def coupled_response_residual(
        self,
        operator_matrix=None,
        metric_density_response=None,
    ):
        if operator_matrix is None:
            operator_matrix = self.coupled_operator
        if metric_density_response is None:
            metric_density_response = self.metric_density_response
        return independent_coupled_response_residual(
            self.reference,
            self.response,
            operator_matrix,
            metric_density_response,
        )


@pytest.fixture(scope="session")
def uhf_oracle_case():
    reference = run_fresh_uhf()
    model = make_nonlinear_model()
    state = evaluate_oracle_state(reference, model)
    reference_occupations = np.asarray(reference.mo_occ).copy()
    reference_ao_labels = tuple(reference.mol.ao_labels())
    _, _, internally_stable, externally_stable = reference.stability(
        internal=True,
        external=True,
        return_status=True,
        verbose=0,
    )
    assert internally_stable
    assert externally_stable
    displaced = {}
    occupied_subspace_minima = {}
    minimum_orbital_gaps = np.full(2, np.inf, dtype=np.float64)
    minimum_descriptor_gap = float(
        np.min(np.diff(state.descriptor[:, 1:], axis=-1))
    )
    for step in FINITE_DIFFERENCE_STEPS:
        for atom_index in range(reference.mol.natm):
            for coordinate_index in range(3):
                for direction in (-1, 1):
                    coordinates = ORACLE_COORDINATES.copy()
                    coordinates[atom_index, coordinate_index] += (
                        direction * step
                    )
                    displaced_reference = run_fresh_uhf(coordinates)
                    np.testing.assert_array_equal(
                        displaced_reference.mo_occ,
                        reference_occupations,
                    )
                    assert (
                        tuple(displaced_reference.mol.ao_labels())
                        == reference_ao_labels
                    )
                    displacement_key = (
                        step,
                        atom_index,
                        coordinate_index,
                        direction,
                    )
                    spin_subspace_minima = (
                        occupied_subspace_minimum_singular_values(
                            reference,
                            displaced_reference,
                        )
                    )
                    for minimum_singular_value in spin_subspace_minima:
                        assert np.isfinite(minimum_singular_value)
                        assert minimum_singular_value > 0.99
                    occupied_subspace_minima[displacement_key] = (
                        spin_subspace_minima
                    )
                    for spin in range(2):
                        occupations = displaced_reference.mo_occ[spin]
                        occupied = occupations > 0.0
                        virtual = occupations == 0.0
                        minimum_orbital_gaps[spin] = min(
                            minimum_orbital_gaps[spin],
                            float(
                                np.min(
                                    displaced_reference.mo_energy[spin][
                                        virtual,
                                        None,
                                    ]
                                    - displaced_reference.mo_energy[spin][
                                        occupied
                                    ]
                                )
                            ),
                        )
                    displaced_state = evaluate_oracle_state(
                        displaced_reference,
                        model,
                    )
                    minimum_descriptor_gap = min(
                        minimum_descriptor_gap,
                        float(
                            np.min(
                                np.diff(
                                    displaced_state.descriptor[:, 1:],
                                    axis=-1,
                                )
                            )
                        ),
                    )
                    displaced[displacement_key] = displaced_state
    overlap_derivative = independent_overlap_derivative(reference)
    method = UHFDeePHF(
        reference,
        model,
        projector_basis=ORACLE_PROJECTOR_BASIS,
    )
    np.testing.assert_allclose(
        method.kernel(),
        state.total_energy,
        rtol=0.0,
        atol=2.0e-14,
    )
    gradient_driver = method.nuc_grad_method()
    gradient = gradient_driver.kernel()
    return UHFOracleCase(
        coordinates=ORACLE_COORDINATES.copy(),
        steps=FINITE_DIFFERENCE_STEPS,
        reference=reference,
        model=model,
        method=method,
        response=gradient_driver.response_result,
        gradient_driver=gradient_driver,
        gradient=gradient,
        state=state,
        displaced=displaced,
        occupied_subspace_minimum_singular_values=(
            occupied_subspace_minima
        ),
        coupled_operator=independent_coupled_uhf_operator(reference),
        overlap_derivative=overlap_derivative,
        metric_density_response=independent_metric_density_response(
            reference,
            overlap_derivative,
        ),
        minimum_alpha_orbital_gap=float(minimum_orbital_gaps[0]),
        minimum_beta_orbital_gap=float(minimum_orbital_gaps[1]),
        minimum_descriptor_gap=minimum_descriptor_gap,
    )
