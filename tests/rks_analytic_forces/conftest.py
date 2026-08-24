from dataclasses import dataclass

import numpy as np
import pytest
import torch
from pyscf import dft, gto
from pyscf.dft import gen_grid, libxc
from pyscf.grad import rks as rks_gradient
from pyscf.hessian import rks as rks_hessian

from deepks.descriptor import AtomicDensityDescriptor
from deepks.model.model import CorrNet


ORACLE_COORDINATES = np.array(
    [
        [0.130000000000, -0.210000000000, 0.070000000000],
        [1.510000000000, 0.120000000000, -0.190000000000],
        [-0.460000000000, 1.620000000000, 0.310000000000],
    ],
    dtype=np.float64,
)
ORACLE_ATOMS = ("O", "H", "H")
ORACLE_PROJECTOR_BASIS = [[0, [0.8, 1.0]], [1, [0.3, 1.0]]]
ORACLE_XC = "LDA_X + LDA_C_VWN"
ORACLE_FUNCTIONAL_COMPONENTS = ((1, 1.0), (7, 1.0))
ORACLE_ATOM_GRID = {"O": (20, 50), "H": (20, 50)}
FINITE_DIFFERENCE_STEPS = (1.0e-3, 3.0e-4, 1.0e-4)


def normalized_libxc_components(xc_code):
    hybrid, components = libxc.parse_xc(xc_code)
    totals = {}
    for functional_id, coefficient in components:
        functional_id = int(functional_id)
        totals[functional_id] = totals.get(functional_id, 0.0) + float(
            coefficient
        )
    normalized = tuple(
        (functional_id, coefficient)
        for functional_id, coefficient in sorted(totals.items())
        if coefficient != 0.0
    )
    return tuple(float(value) for value in hybrid), normalized


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
        spin=0,
        symmetry=False,
        cart=False,
        verbose=0,
    )


def configure_oracle_rks(reference, *, build_grid=True):
    reference.xc = ORACLE_XC
    reference.conv_tol = 1.0e-13
    reference.conv_tol_grad = 1.0e-10
    reference.conv_tol_cpscf = 1.0e-12
    reference.max_cycle = 100
    reference.small_rho_cutoff = 0.0
    reference.grids.atom_grid = dict(ORACLE_ATOM_GRID)
    reference.grids.prune = None
    reference.grids.alignment = 1
    if build_grid:
        reference.grids.build(with_non0tab=True, sort_grids=False)
    return reference


def run_fresh_rks(coordinates=ORACLE_COORDINATES):
    reference = configure_oracle_rks(
        dft.RKS(make_oracle_molecule(coordinates))
    )
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


def occupied_subspace_minimum_singular_value(reference, displaced_reference):
    reference_occupied = np.asarray(reference.mo_occ) > 0.0
    displaced_occupied = np.asarray(displaced_reference.mo_occ) > 0.0
    cross_overlap = np.asarray(
        gto.intor_cross(
            "int1e_ovlp",
            reference.mol,
            displaced_reference.mol,
        )
    )
    occupied_overlap = (
        np.asarray(reference.mo_coeff)[:, reference_occupied].T
        @ cross_overlap
        @ np.asarray(displaced_reference.mo_coeff)[:, displaced_occupied]
    )
    return float(
        np.min(np.linalg.svd(occupied_overlap, compute_uv=False))
    )


def independent_overlap_derivative(reference):
    derivative_integrals = -np.asarray(
        reference.mol.intor("int1e_ipovlp", comp=3)
    )
    derivative = np.zeros(
        (
            reference.mol.natm,
            3,
            reference.mol.nao,
            reference.mol.nao,
        ),
        dtype=np.float64,
    )
    for atom_index, atom_slice in enumerate(
        reference.mol.aoslice_by_atom()
    ):
        ao_start, ao_stop = atom_slice[2:]
        derivative[atom_index, :, ao_start:ao_stop] += (
            derivative_integrals[:, ao_start:ao_stop]
        )
        derivative[atom_index, :, :, ao_start:ao_stop] += (
            derivative_integrals[:, ao_start:ao_stop].transpose(0, 2, 1)
        )
    return derivative


def _independent_lda_quantities(reference):
    numerical_integrator = reference._numint
    coordinates = np.asarray(reference.grids.coords)
    ao_values = np.asarray(
        numerical_integrator.eval_ao(
            reference.mol,
            coordinates,
            deriv=1,
        )
    )
    density = np.asarray(reference.make_rdm1())
    rho_and_gradient = np.asarray(
        numerical_integrator.eval_rho(
            reference.mol,
            ao_values,
            density,
            xctype="GGA",
            hermi=1,
        )
    )
    _, vxc, fxc = numerical_integrator.eval_xc_eff(
        reference.xc,
        rho_and_gradient[0],
        2,
        xctype="LDA",
    )[:3]
    return (
        ao_values,
        rho_and_gradient,
        np.asarray(vxc[0]),
        np.asarray(fxc[0, 0]),
    )


def independent_xc_hamiltonian_components(reference):
    molecule = reference.mol
    density = np.asarray(reference.make_rdm1())
    weights = np.asarray(reference.grids.weights)
    atom_indices = np.asarray(reference.grids.atm_idx)
    ao_values, rho_and_gradient, vxc, fxc = (
        _independent_lda_quantities(reference)
    )
    coordinate_chunks = []
    weight_chunks = []
    derivative_weight_chunks = []
    for coordinates, chunk_weights, derivative_weights in (
        rks_gradient.grids_response_cc(reference.grids)
    ):
        coordinate_chunks.append(np.asarray(coordinates))
        weight_chunks.append(np.asarray(chunk_weights))
        derivative_weight_chunks.append(np.asarray(derivative_weights))
    regenerated_coordinates = np.vstack(coordinate_chunks)
    regenerated_weights = np.concatenate(weight_chunks)
    derivative_weights = np.concatenate(
        derivative_weight_chunks,
        axis=-1,
    )
    np.testing.assert_array_equal(
        regenerated_coordinates,
        reference.grids.coords,
    )
    np.testing.assert_array_equal(
        regenerated_weights,
        reference.grids.weights,
    )
    shape = (molecule.natm, 3, molecule.nao, molecule.nao)
    ao_motion = np.zeros(shape, dtype=np.float64)
    grid_coordinate = np.zeros(shape, dtype=np.float64)
    grid_weight = np.zeros(shape, dtype=np.float64)
    for atom_index, atom_slice in enumerate(molecule.aoslice_by_atom()):
        ao_start, ao_stop = atom_slice[2:]
        center_mask = atom_indices == atom_index
        for coordinate_index in range(3):
            derivative_ao = np.zeros_like(ao_values[0])
            derivative_ao[:, ao_start:ao_stop] = -ao_values[
                coordinate_index + 1,
                :,
                ao_start:ao_stop,
            ]
            derivative_rho = np.einsum(
                "gi,ij,gj->g",
                derivative_ao,
                density,
                ao_values[0],
            ) + np.einsum(
                "gi,ij,gj->g",
                ao_values[0],
                density,
                derivative_ao,
            )
            ao_motion[atom_index, coordinate_index] = np.einsum(
                "g,g,g,gi,gj->ij",
                weights,
                fxc,
                derivative_rho,
                ao_values[0],
                ao_values[0],
                optimize=True,
            )
            ao_motion[atom_index, coordinate_index] += np.einsum(
                "g,g,gi,gj->ij",
                weights,
                vxc,
                derivative_ao,
                ao_values[0],
                optimize=True,
            )
            ao_motion[atom_index, coordinate_index] += np.einsum(
                "g,g,gi,gj->ij",
                weights,
                vxc,
                ao_values[0],
                derivative_ao,
                optimize=True,
            )
            grid_coordinate[atom_index, coordinate_index] = np.einsum(
                "g,g,g,gi,gj->ij",
                weights[center_mask],
                fxc[center_mask],
                rho_and_gradient[
                    coordinate_index + 1,
                    center_mask,
                ],
                ao_values[0, center_mask],
                ao_values[0, center_mask],
                optimize=True,
            )
            grid_coordinate[atom_index, coordinate_index] += np.einsum(
                "g,g,gi,gj->ij",
                weights[center_mask],
                vxc[center_mask],
                ao_values[coordinate_index + 1, center_mask],
                ao_values[0, center_mask],
                optimize=True,
            )
            grid_coordinate[atom_index, coordinate_index] += np.einsum(
                "g,g,gi,gj->ij",
                weights[center_mask],
                vxc[center_mask],
                ao_values[0, center_mask],
                ao_values[coordinate_index + 1, center_mask],
                optimize=True,
            )
            grid_weight[atom_index, coordinate_index] = np.einsum(
                "g,g,gi,gj->ij",
                derivative_weights[atom_index, coordinate_index],
                vxc,
                ao_values[0],
                ao_values[0],
                optimize=True,
            )
    return ao_motion, grid_coordinate, grid_weight


def fresh_grid_fixed_density_hamiltonian_components(
    reference,
    step=3.0e-5,
):
    density = np.asarray(reference.make_rdm1())
    central_coordinates = np.asarray(reference.grids.coords)
    central_weights = np.asarray(reference.grids.weights)
    shape = (
        2,
        reference.mol.natm,
        3,
        reference.mol.nao,
        reference.mol.nao,
    )
    values = {
        name: np.empty(shape, dtype=np.float64)
        for name in ("full", "frozen", "coordinate", "weight")
    }

    def effective_potential(molecule, coordinates, weights):
        method = configure_oracle_rks(
            dft.RKS(molecule),
            build_grid=False,
        )
        method.grids.coords = np.ascontiguousarray(coordinates)
        method.grids.weights = np.ascontiguousarray(weights)
        method.grids.non0tab = None
        return np.asarray(method.get_veff(molecule, density))

    for direction_index, direction in enumerate((-1, 1)):
        for atom_index in range(reference.mol.natm):
            for coordinate_index in range(3):
                coordinates = ORACLE_COORDINATES.copy()
                coordinates[atom_index, coordinate_index] += (
                    direction * step
                )
                molecule = make_oracle_molecule(coordinates)
                fresh_method = configure_oracle_rks(dft.RKS(molecule))
                fresh_coordinates = np.asarray(fresh_method.grids.coords)
                fresh_weights = np.asarray(fresh_method.grids.weights)
                values["full"][
                    direction_index,
                    atom_index,
                    coordinate_index,
                ] = effective_potential(
                    molecule,
                    fresh_coordinates,
                    fresh_weights,
                )
                values["frozen"][
                    direction_index,
                    atom_index,
                    coordinate_index,
                ] = effective_potential(
                    molecule,
                    central_coordinates,
                    central_weights,
                )
                values["coordinate"][
                    direction_index,
                    atom_index,
                    coordinate_index,
                ] = effective_potential(
                    molecule,
                    fresh_coordinates,
                    central_weights,
                )
                values["weight"][
                    direction_index,
                    atom_index,
                    coordinate_index,
                ] = effective_potential(
                    molecule,
                    central_coordinates,
                    fresh_weights,
                )
    derivatives = {
        name: (value[1] - value[0]) / (2.0 * step)
        for name, value in values.items()
    }
    return (
        derivatives["coordinate"] - derivatives["frozen"],
        derivatives["weight"] - derivatives["frozen"],
        derivatives["full"] - derivatives["frozen"],
    )


def _independent_induced_potentials(reference):
    molecule = reference.mol
    density = np.asarray(reference.make_rdm1())
    ao_values, _, _, fxc = _independent_lda_quantities(reference)
    electron_repulsion = np.asarray(
        molecule.intor("int2e", aosym="s1")
    )
    weights = np.asarray(reference.grids.weights)

    def components(density_response):
        density_response = np.asarray(density_response)
        coulomb = np.einsum(
            "mnkl,...lk->...mn",
            electron_repulsion,
            density_response,
        )
        rho_response = np.einsum(
            "gi,...ij,gj->...g",
            ao_values[0],
            density_response,
            ao_values[0],
        )
        xc_kernel = np.einsum(
            "g,g,...g,gi,gj->...ij",
            weights,
            fxc,
            rho_response,
            ao_values[0],
            ao_values[0],
            optimize=True,
        )
        return coulomb, xc_kernel

    quadrature_electron_count = float(
        np.einsum(
            "g,g,gi,ij,gj->",
            weights,
            np.ones_like(weights),
            ao_values[0],
            density,
            ao_values[0],
            optimize=True,
        )
    )
    return components, quadrature_electron_count


def _density_from_mo_response(reference, mo_response):
    coefficient = np.asarray(reference.mo_coeff)
    occupied = np.asarray(reference.mo_occ) > 0.0
    occupied_coefficients = coefficient[:, occupied]
    coefficient_response = np.einsum(
        "pm,...mi->...pi",
        coefficient,
        mo_response,
    )
    one_sided = np.einsum(
        "...pi,qi->...pq",
        coefficient_response,
        occupied_coefficients,
    )
    return 2.0 * (one_sided + one_sided.swapaxes(-1, -2))


def independent_cpks_oracle(reference):
    molecule = reference.mol
    coefficient = np.asarray(reference.mo_coeff)
    energy = np.asarray(reference.mo_energy)
    occupation = np.asarray(reference.mo_occ)
    occupied = occupation > 0.0
    virtual = occupation == 0.0
    occupied_coefficients = coefficient[:, occupied]
    occupied_count = int(np.count_nonzero(occupied))
    virtual_count = int(np.count_nonzero(virtual))
    dimension = occupied_count * virtual_count
    overlap_derivative = independent_overlap_derivative(reference)
    overlap_mo = np.einsum(
        "pm,...pq,qi->...mi",
        coefficient,
        overlap_derivative,
        occupied_coefficients,
    )
    metric_mo_response = np.zeros(
        (molecule.natm, 3, coefficient.shape[1], occupied_count),
        dtype=np.float64,
    )
    metric_mo_response[..., occupied, :] = (
        -0.5 * overlap_mo[..., occupied, :]
    )
    density_metric = _density_from_mo_response(
        reference,
        metric_mo_response,
    )
    induced_components, quadrature_electron_count = (
        _independent_induced_potentials(reference)
    )
    identity = np.eye(dimension, dtype=np.float64)
    orbital_gaps = energy[virtual, None] - energy[occupied]
    coulomb_operator = np.zeros((dimension, dimension), dtype=np.float64)
    fxc_operator = np.zeros_like(coulomb_operator)
    for source_index in range(dimension):
        mo_response = np.zeros(
            (coefficient.shape[1], occupied_count),
            dtype=np.float64,
        )
        mo_response[virtual] = identity[:, source_index].reshape(
            virtual_count,
            occupied_count,
        )
        density_response = _density_from_mo_response(
            reference,
            mo_response,
        )
        coulomb, xc_kernel = induced_components(density_response)
        coulomb_operator[:, source_index] = (
            coefficient[:, virtual].T
            @ coulomb
            @ occupied_coefficients
        ).reshape(-1)
        fxc_operator[:, source_index] = (
            coefficient[:, virtual].T
            @ xc_kernel
            @ occupied_coefficients
        ).reshape(-1)
    gap_operator = np.diag(orbital_gaps.reshape(-1))
    operator = gap_operator + coulomb_operator + fxc_operator
    ao_motion, grid_coordinate, grid_weight = (
        independent_xc_hamiltonian_components(reference)
    )
    hamiltonian_fixed_grid = np.asarray(
        rks_hessian.Hessian(reference).make_h1(
            coefficient,
            occupation,
            atmlst=range(molecule.natm),
        )
    )
    hamiltonian_derivative = (
        hamiltonian_fixed_grid + grid_coordinate + grid_weight
    )

    def solve(
        *,
        include_coulomb=True,
        include_fxc=True,
        include_ao_motion=True,
        include_grid_coordinate=True,
        include_grid_weight=True,
        include_metric=True,
    ):
        active_operator = gap_operator.copy()
        if include_coulomb:
            active_operator += coulomb_operator
        if include_fxc:
            active_operator += fxc_operator
        active_hamiltonian = hamiltonian_fixed_grid.copy()
        if not include_ao_motion:
            active_hamiltonian -= ao_motion
        if include_grid_coordinate:
            active_hamiltonian += grid_coordinate
        if include_grid_weight:
            active_hamiltonian += grid_weight
        active_metric_mo = (
            metric_mo_response
            if include_metric
            else np.zeros_like(metric_mo_response)
        )
        active_density_metric = _density_from_mo_response(
            reference,
            active_metric_mo,
        )
        metric_coulomb, metric_fxc = induced_components(
            active_density_metric
        )
        metric_potential = np.zeros_like(metric_coulomb)
        if include_coulomb:
            metric_potential += metric_coulomb
        if include_fxc:
            metric_potential += metric_fxc
        hamiltonian_mo = np.einsum(
            "pm,...pq,qi->...mi",
            coefficient,
            active_hamiltonian,
            occupied_coefficients,
        )
        metric_potential_mo = np.einsum(
            "pm,...pq,qi->...mi",
            coefficient,
            metric_potential,
            occupied_coefficients,
        )
        right_hand_side = -(
            hamiltonian_mo[..., virtual, :]
            + metric_potential_mo[..., virtual, :]
            - overlap_mo[..., virtual, :] * energy[occupied]
        )
        occupied_virtual_solution = np.linalg.solve(
            active_operator,
            right_hand_side.reshape(-1, dimension).T,
        ).T.reshape(
            molecule.natm,
            3,
            virtual_count,
            occupied_count,
        )
        occupied_virtual_mo = np.zeros_like(metric_mo_response)
        occupied_virtual_mo[..., virtual, :] = (
            occupied_virtual_solution
        )
        complete_mo = active_metric_mo + occupied_virtual_mo
        density_occupied_virtual = _density_from_mo_response(
            reference,
            occupied_virtual_mo,
        )
        density_response = active_density_metric + density_occupied_virtual
        induced_coulomb, induced_fxc = induced_components(density_response)
        induced_potential = np.zeros_like(induced_coulomb)
        if include_coulomb:
            induced_potential += induced_coulomb
        if include_fxc:
            induced_potential += induced_fxc
        induced_mo = np.einsum(
            "pm,...pq,qi->...mi",
            coefficient,
            induced_potential,
            occupied_coefficients,
        )
        residual = (
            hamiltonian_mo
            + induced_mo
            - overlap_mo * energy[occupied]
            + (energy[:, None] - energy[occupied]) * complete_mo
        )[..., virtual, :]
        return IndependentRKSResponseSolution(
            mo_response=complete_mo,
            mo_response_metric=active_metric_mo,
            mo_response_occupied_virtual=occupied_virtual_mo,
            density_response=density_response,
            density_response_metric=active_density_metric,
            density_response_occupied_virtual=(
                density_occupied_virtual
            ),
            residual=residual,
        )

    full_solution = solve()
    native_driver = reference.nuc_grad_method()
    native_driver.grid_response = True
    native_gradient = np.asarray(native_driver.kernel())
    (
        fresh_grid_fd_coordinate,
        fresh_grid_fd_weight,
        fresh_grid_fd_total,
    ) = fresh_grid_fixed_density_hamiltonian_components(reference)
    return IndependentRKSResponseOracle(
        operator=operator,
        gap_operator=gap_operator,
        coulomb_operator=coulomb_operator,
        fxc_operator=fxc_operator,
        overlap_derivative=overlap_derivative,
        hamiltonian_derivative=hamiltonian_derivative,
        hamiltonian_derivative_fixed_grid=hamiltonian_fixed_grid,
        xc_hamiltonian_derivative_ao_motion=ao_motion,
        xc_hamiltonian_derivative_grid_coordinate=grid_coordinate,
        xc_hamiltonian_derivative_grid_weight=grid_weight,
        quadrature_electron_count=quadrature_electron_count,
        native_gradient=native_gradient,
        fresh_grid_fd_coordinate=fresh_grid_fd_coordinate,
        fresh_grid_fd_weight=fresh_grid_fd_weight,
        fresh_grid_fd_total=fresh_grid_fd_total,
        solution=full_solution,
        without_coulomb=solve(include_coulomb=False),
        without_fxc=solve(include_fxc=False),
        without_metric=solve(include_metric=False),
        without_ao_motion=solve(include_ao_motion=False),
        without_grid_response=solve(
            include_grid_coordinate=False,
            include_grid_weight=False,
        ),
        without_grid_coordinate=solve(include_grid_coordinate=False),
        without_grid_weight=solve(include_grid_weight=False),
    )


@dataclass(frozen=True)
class IndependentRKSResponseSolution:
    mo_response: np.ndarray
    mo_response_metric: np.ndarray
    mo_response_occupied_virtual: np.ndarray
    density_response: np.ndarray
    density_response_metric: np.ndarray
    density_response_occupied_virtual: np.ndarray
    residual: np.ndarray


@dataclass(frozen=True)
class IndependentRKSResponseOracle:
    operator: np.ndarray
    gap_operator: np.ndarray
    coulomb_operator: np.ndarray
    fxc_operator: np.ndarray
    overlap_derivative: np.ndarray
    hamiltonian_derivative: np.ndarray
    hamiltonian_derivative_fixed_grid: np.ndarray
    xc_hamiltonian_derivative_ao_motion: np.ndarray
    xc_hamiltonian_derivative_grid_coordinate: np.ndarray
    xc_hamiltonian_derivative_grid_weight: np.ndarray
    quadrature_electron_count: float
    native_gradient: np.ndarray
    fresh_grid_fd_coordinate: np.ndarray
    fresh_grid_fd_weight: np.ndarray
    fresh_grid_fd_total: np.ndarray
    solution: IndependentRKSResponseSolution
    without_coulomb: IndependentRKSResponseSolution
    without_fxc: IndependentRKSResponseSolution
    without_metric: IndependentRKSResponseSolution
    without_ao_motion: IndependentRKSResponseSolution
    without_grid_response: IndependentRKSResponseSolution
    without_grid_coordinate: IndependentRKSResponseSolution
    without_grid_weight: IndependentRKSResponseSolution


@dataclass(frozen=True)
class RKSOracleState:
    density: np.ndarray
    overlap: np.ndarray
    descriptor: np.ndarray
    base_energy: float
    total_energy: float


@dataclass(frozen=True)
class RKSOracleCase:
    coordinates: np.ndarray
    steps: tuple[float, ...]
    reference: object
    method: object
    response: object
    gradient_driver: object
    gradient: np.ndarray
    model: torch.nn.Module
    independent: IndependentRKSResponseOracle
    displaced: dict[tuple[float, int, int, int], RKSOracleState]
    occupied_subspace_minimum_singular_values: dict[
        tuple[float, int, int, int], float
    ]
    minimum_orbital_gap: float
    minimum_descriptor_gap: float

    def finite_difference(self, field: str, step: float) -> np.ndarray:
        sample = np.asarray(
            getattr(self.displaced[(step, 0, 0, 1)], field)
        )
        result = np.empty((self.reference.mol.natm, 3, *sample.shape))
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


@pytest.fixture(scope="session")
def rks_oracle_case():
    from deepks.deephf import RKSDeePHF

    hybrid, components = normalized_libxc_components(ORACLE_XC)
    assert hybrid == (0.0, 0.0, 0.0)
    assert components == ORACLE_FUNCTIONAL_COMPONENTS
    assert libxc.xc_type(ORACLE_XC) == "LDA"
    assert libxc.hybrid_coeff(ORACLE_XC) == 0.0
    assert tuple(libxc.rsh_coeff(ORACLE_XC)) == (0.0, 0.0, 0.0)
    assert libxc.nlc_coeff(ORACLE_XC) == ()
    reference = run_fresh_rks()
    assert reference.grids.prune is None
    assert reference.grids.radi_method is dft.radi.treutler
    assert reference.grids.radii_adjust is (
        dft.radi.treutler_atomic_radii_adjust
    )
    assert reference.grids.becke_scheme is gen_grid.original_becke
    assert reference.grids.size == 3000
    assert np.array_equal(
        np.bincount(reference.grids.atm_idx),
        np.array([1000, 1000, 1000]),
    )
    model = make_nonlinear_model()
    method = RKSDeePHF(
        reference,
        model,
        projector_basis=ORACLE_PROJECTOR_BASIS,
    )
    method.kernel()
    response = method.response()
    gradient_driver = method.nuc_grad_method()
    gradient = gradient_driver.kernel()
    independent = independent_cpks_oracle(reference)
    reference_occupations = np.asarray(reference.mo_occ).copy()
    reference_ao_labels = tuple(reference.mol.ao_labels())
    displaced = {}
    occupied_subspace_minimum_singular_values = {}
    minimum_orbital_gap = np.inf
    minimum_descriptor_gap = np.inf
    for step in FINITE_DIFFERENCE_STEPS:
        for atom_index in range(reference.mol.natm):
            for coordinate_index in range(3):
                for direction in (-1, 1):
                    coordinates = ORACLE_COORDINATES.copy()
                    coordinates[atom_index, coordinate_index] += (
                        direction * step
                    )
                    displaced_reference = run_fresh_rks(coordinates)
                    np.testing.assert_array_equal(
                        displaced_reference.mo_occ,
                        reference_occupations,
                    )
                    assert (
                        tuple(displaced_reference.mol.ao_labels())
                        == reference_ao_labels
                    )
                    occupied_subspace_minimum_singular_values[
                        (step, atom_index, coordinate_index, direction)
                    ] = occupied_subspace_minimum_singular_value(
                        reference,
                        displaced_reference,
                    )
                    occupied = displaced_reference.mo_occ > 0.0
                    virtual = displaced_reference.mo_occ == 0.0
                    minimum_orbital_gap = min(
                        minimum_orbital_gap,
                        float(
                            np.min(
                                displaced_reference.mo_energy[virtual, None]
                                - displaced_reference.mo_energy[occupied]
                            )
                        ),
                    )
                    expected_grid_coordinates = np.asarray(
                        reference.grids.coords
                    ).copy()
                    expected_grid_coordinates[
                        reference.grids.atm_idx == atom_index,
                        coordinate_index,
                    ] += direction * step
                    np.testing.assert_allclose(
                        displaced_reference.grids.coords,
                        expected_grid_coordinates,
                        rtol=0.0,
                        atol=2.0e-15,
                    )
                    assert displaced_reference.grids.size == 3000
                    displaced_method = RKSDeePHF(
                        displaced_reference,
                        model,
                        projector_basis=ORACLE_PROJECTOR_BASIS,
                    )
                    total_energy = displaced_method.kernel()
                    descriptor = displaced_method.descriptor()
                    minimum_descriptor_gap = min(
                        minimum_descriptor_gap,
                        float(np.min(np.diff(descriptor[:, 1:], axis=-1))),
                    )
                    displaced[
                        (step, atom_index, coordinate_index, direction)
                    ] = RKSOracleState(
                        density=displaced_method.ao_density(),
                        overlap=np.asarray(displaced_reference.get_ovlp()),
                        descriptor=descriptor,
                        base_energy=float(displaced_reference.e_tot),
                        total_energy=float(total_energy),
                    )
    return RKSOracleCase(
        coordinates=ORACLE_COORDINATES.copy(),
        steps=FINITE_DIFFERENCE_STEPS,
        reference=reference,
        method=method,
        response=response,
        gradient_driver=gradient_driver,
        gradient=gradient,
        model=model,
        independent=independent,
        displaced=displaced,
        occupied_subspace_minimum_singular_values=(
            occupied_subspace_minimum_singular_values
        ),
        minimum_orbital_gap=minimum_orbital_gap,
        minimum_descriptor_gap=minimum_descriptor_gap,
    )
