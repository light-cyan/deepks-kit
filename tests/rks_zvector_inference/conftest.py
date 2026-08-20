from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import torch


_P4B_FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "rks_analytic_forces"
    / "conftest.py"
)
_P4B_FIXTURE_SPEC = importlib.util.spec_from_file_location(
    "_deepks_p4b_rks_oracle_fixtures",
    _P4B_FIXTURE_PATH,
)
if _P4B_FIXTURE_SPEC is None or _P4B_FIXTURE_SPEC.loader is None:
    raise RuntimeError("the P4B independent RKS oracle fixture cannot be loaded")
_P4B_FIXTURES = importlib.util.module_from_spec(_P4B_FIXTURE_SPEC)
sys.modules[_P4B_FIXTURE_SPEC.name] = _P4B_FIXTURES
_P4B_FIXTURE_SPEC.loader.exec_module(_P4B_FIXTURES)
ORACLE_PROJECTOR_BASIS = _P4B_FIXTURES.ORACLE_PROJECTOR_BASIS
RKSOracleCase = _P4B_FIXTURES.RKSOracleCase
rks_oracle_case = _P4B_FIXTURES.rks_oracle_case


def independent_objective_ao_potential(case: RKSOracleCase) -> np.ndarray:
    density = torch.tensor(
        case.method.ao_density(),
        dtype=torch.float64,
        requires_grad=True,
    )
    descriptor = case.method._descriptor.torch_descriptor(density)
    correction = case.model(descriptor).sum()
    (potential,) = torch.autograd.grad(correction, density)
    return potential.detach().cpu().numpy()


def independent_induced_potential(reference, density_response) -> np.ndarray:
    density_response = np.asarray(density_response, dtype=np.float64)
    perturbation_shape = density_response.shape[:-2]
    flat_density = density_response.reshape(
        -1,
        reference.mol.nao,
        reference.mol.nao,
    )
    electron_repulsion = np.asarray(
        reference.mol.intor("int2e", aosym="s1")
    )
    coulomb = np.einsum(
        "mnkl,xlk->xmn",
        electron_repulsion,
        flat_density,
    )
    coordinates = np.asarray(reference.grids.coords)
    weights = np.asarray(reference.grids.weights)
    ao_values = np.asarray(
        reference._numint.eval_ao(
            reference.mol,
            coordinates,
            deriv=0,
        )
    )
    ground_density = np.asarray(reference.make_rdm1())
    ground_rho = np.einsum(
        "gi,ij,gj->g",
        ao_values,
        ground_density,
        ao_values,
        optimize=True,
    )
    fxc = np.asarray(
        reference._numint.eval_xc_eff(
            reference.xc,
            ground_rho,
            deriv=2,
            xctype="LDA",
            spin=0,
        )[2]
    )[0, 0]
    density_rho = np.einsum(
        "gi,xij,gj->xg",
        ao_values,
        flat_density,
        ao_values,
        optimize=True,
    )
    xc_kernel = np.einsum(
        "g,g,xg,gi,gj->xij",
        weights,
        fxc,
        density_rho,
        ao_values,
        ao_values,
        optimize=True,
    )
    return (coulomb + xc_kernel).reshape(
        *perturbation_shape,
        reference.mol.nao,
        reference.mol.nao,
    )


def _mo_occupied_block(reference, ao_tensor) -> np.ndarray:
    coefficient = np.asarray(reference.mo_coeff)
    occupied = np.asarray(reference.mo_occ) > 0.0
    return np.einsum(
        "pm,...pq,qi->...mi",
        coefficient,
        ao_tensor,
        coefficient[:, occupied],
    )


@dataclass(frozen=True)
class IndependentRKSAdjointOracle:
    objective_ao_potential: np.ndarray
    objective_mo_potential: np.ndarray
    objective_orbital_gradient: np.ndarray
    one_sided_objective_gradient: np.ndarray
    operator: np.ndarray
    zvector: np.ndarray
    solver_residual: np.ndarray
    transpose_residual: np.ndarray
    adjoint_ao_density: np.ndarray
    adjoint_ao_potential: np.ndarray
    correction_gradient_metric: np.ndarray
    correction_gradient_adjoint_ao_motion: np.ndarray
    correction_gradient_adjoint_fixed_grid: np.ndarray
    correction_gradient_adjoint_grid_coordinate: np.ndarray
    correction_gradient_adjoint_grid_weight: np.ndarray
    correction_gradient_adjoint_nuclear: np.ndarray
    correction_gradient_adjoint_metric: np.ndarray
    correction_gradient_adjoint_metric_overlap_form: np.ndarray
    correction_gradient_occupied_virtual: np.ndarray
    correction_gradient_response: np.ndarray


def build_independent_rks_adjoint_oracle(
    case: RKSOracleCase,
) -> IndependentRKSAdjointOracle:
    reference = case.reference
    direct_oracle = case.independent
    coefficient = np.asarray(reference.mo_coeff)
    energy = np.asarray(reference.mo_energy)
    occupation = np.asarray(reference.mo_occ)
    occupied = occupation > 0.0
    virtual = occupation == 0.0
    occupied_coefficients = coefficient[:, occupied]
    virtual_coefficients = coefficient[:, virtual]
    objective_ao = independent_objective_ao_potential(case)
    objective_mo = coefficient.T @ objective_ao @ coefficient
    objective_gradient = (
        objective_mo[virtual][:, occupied]
        + objective_mo.T[virtual][:, occupied]
    ) * occupation[occupied]
    one_sided_objective_gradient = (
        objective_mo[virtual][:, occupied] * occupation[occupied]
    )
    operator = np.asarray(direct_oracle.operator)
    zvector = np.linalg.solve(operator.T, objective_gradient.reshape(-1)).reshape(
        objective_gradient.shape
    )
    solver_residual = (
        operator.T @ zvector.reshape(-1) - objective_gradient.reshape(-1)
    ).reshape(objective_gradient.shape)
    transpose_residual = np.empty_like(solver_residual)
    identity = np.eye(operator.shape[0], dtype=np.float64)
    independently_rebuilt_operator = np.empty_like(operator)
    orbital_gaps = energy[virtual, None] - energy[occupied]
    for source_index in range(operator.shape[0]):
        amplitude = identity[:, source_index].reshape(
            objective_gradient.shape
        )
        rotated_occupied = virtual_coefficients @ amplitude
        trial_density = rotated_occupied @ (
            occupied_coefficients * occupation[occupied]
        ).T
        trial_density = trial_density + trial_density.T
        induced = independent_induced_potential(reference, trial_density)
        induced_vo = (
            virtual_coefficients.T @ induced @ occupied_coefficients
        )
        independently_rebuilt_operator[:, source_index] = (
            orbital_gaps * amplitude + induced_vo
        ).reshape(-1)
    transpose_residual[:] = (
        independently_rebuilt_operator.T @ zvector.reshape(-1)
        - objective_gradient.reshape(-1)
    ).reshape(objective_gradient.shape)
    rotated_occupied = virtual_coefficients @ zvector
    adjoint_density = rotated_occupied @ (
        occupied_coefficients * occupation[occupied]
    ).T
    adjoint_density = adjoint_density + adjoint_density.T
    adjoint_potential = independent_induced_potential(
        reference,
        adjoint_density,
    )
    overlap_mo = _mo_occupied_block(
        reference,
        direct_oracle.overlap_derivative,
    )
    fixed_grid_mo = _mo_occupied_block(
        reference,
        direct_oracle.hamiltonian_derivative_fixed_grid,
    )
    grid_coordinate_mo = _mo_occupied_block(
        reference,
        direct_oracle.xc_hamiltonian_derivative_grid_coordinate,
    )
    grid_weight_mo = _mo_occupied_block(
        reference,
        direct_oracle.xc_hamiltonian_derivative_grid_weight,
    )
    fixed_grid_rhs = (
        fixed_grid_mo[..., virtual, :]
        - overlap_mo[..., virtual, :] * energy[occupied]
    )
    correction_fixed_grid = -np.einsum(
        "ai,...ai->...",
        zvector,
        fixed_grid_rhs,
    )
    ao_motion_mo = _mo_occupied_block(
        reference,
        direct_oracle.xc_hamiltonian_derivative_ao_motion,
    )
    correction_ao_motion = -np.einsum(
        "ai,...ai->...",
        zvector,
        ao_motion_mo[..., virtual, :],
    )
    correction_grid_coordinate = -np.einsum(
        "ai,...ai->...",
        zvector,
        grid_coordinate_mo[..., virtual, :],
    )
    correction_grid_weight = -np.einsum(
        "ai,...ai->...",
        zvector,
        grid_weight_mo[..., virtual, :],
    )
    correction_nuclear = (
        correction_fixed_grid
        + correction_grid_coordinate
        + correction_grid_weight
    )
    density_metric = direct_oracle.solution.density_response_metric
    correction_metric = np.einsum(
        "ij,...ij->...",
        objective_ao,
        density_metric,
    )
    metric_induced = independent_induced_potential(
        reference,
        density_metric,
    )
    metric_induced_mo = _mo_occupied_block(reference, metric_induced)
    correction_adjoint_metric = -np.einsum(
        "ai,...ai->...",
        zvector,
        metric_induced_mo[..., virtual, :],
    )
    adjoint_potential_mo = coefficient.T @ adjoint_potential @ coefficient
    adjoint_potential_occupied = adjoint_potential_mo[occupied][:, occupied]
    adjoint_potential_occupied = 0.5 * (
        adjoint_potential_occupied + adjoint_potential_occupied.T
    )
    correction_adjoint_metric_overlap = np.einsum(
        "...ij,ij->...",
        overlap_mo[..., occupied, :],
        0.5 * adjoint_potential_occupied,
    )
    correction_occupied_virtual = (
        correction_nuclear + correction_adjoint_metric
    )
    correction_response = correction_metric + correction_occupied_virtual
    return IndependentRKSAdjointOracle(
        objective_ao_potential=objective_ao,
        objective_mo_potential=objective_mo,
        objective_orbital_gradient=objective_gradient,
        one_sided_objective_gradient=one_sided_objective_gradient,
        operator=independently_rebuilt_operator,
        zvector=zvector,
        solver_residual=solver_residual,
        transpose_residual=transpose_residual,
        adjoint_ao_density=adjoint_density,
        adjoint_ao_potential=adjoint_potential,
        correction_gradient_metric=correction_metric,
        correction_gradient_adjoint_ao_motion=correction_ao_motion,
        correction_gradient_adjoint_fixed_grid=correction_fixed_grid,
        correction_gradient_adjoint_grid_coordinate=(
            correction_grid_coordinate
        ),
        correction_gradient_adjoint_grid_weight=correction_grid_weight,
        correction_gradient_adjoint_nuclear=correction_nuclear,
        correction_gradient_adjoint_metric=correction_adjoint_metric,
        correction_gradient_adjoint_metric_overlap_form=(
            correction_adjoint_metric_overlap
        ),
        correction_gradient_occupied_virtual=correction_occupied_virtual,
        correction_gradient_response=correction_response,
    )


@pytest.fixture(scope="session")
def rks_zvector_oracle(rks_oracle_case):
    return build_independent_rks_adjoint_oracle(rks_oracle_case)


__all__ = [
    "ORACLE_PROJECTOR_BASIS",
    "IndependentRKSAdjointOracle",
    "build_independent_rks_adjoint_oracle",
    "independent_induced_potential",
    "independent_objective_ao_potential",
    "rks_oracle_case",
    "rks_zvector_oracle",
]
