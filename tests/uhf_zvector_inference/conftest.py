from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import torch


_P4A_FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "uhf_analytic_forces"
    / "conftest.py"
)
_P4A_FIXTURE_SPEC = importlib.util.spec_from_file_location(
    "_deepks_p4a_uhf_oracle_fixtures",
    _P4A_FIXTURE_PATH,
)
if _P4A_FIXTURE_SPEC is None or _P4A_FIXTURE_SPEC.loader is None:
    raise RuntimeError("the P4A independent UHF oracle fixture cannot be loaded")
_P4A_FIXTURES = importlib.util.module_from_spec(_P4A_FIXTURE_SPEC)
sys.modules[_P4A_FIXTURE_SPEC.name] = _P4A_FIXTURES
_P4A_FIXTURE_SPEC.loader.exec_module(_P4A_FIXTURES)
ORACLE_PROJECTOR_BASIS = _P4A_FIXTURES.ORACLE_PROJECTOR_BASIS
ORACLE_COORDINATES = _P4A_FIXTURES.ORACLE_COORDINATES
UHFOracleCase = _P4A_FIXTURES.UHFOracleCase
make_nonlinear_model = _P4A_FIXTURES.make_nonlinear_model
run_fresh_uhf = _P4A_FIXTURES.run_fresh_uhf
uhf_oracle_case = _P4A_FIXTURES.uhf_oracle_case


def independent_objective_ao_potential(case: UHFOracleCase) -> np.ndarray:
    density = torch.tensor(
        case.method.ao_density(),
        dtype=torch.float64,
        requires_grad=True,
    )
    descriptor = case.method._descriptor.torch_descriptor(density)
    correction = case.model(descriptor).sum()
    (potential,) = torch.autograd.grad(correction, density)
    return potential.detach().cpu().numpy()


def independent_induced_uhf_potential(reference, densities):
    densities = np.asarray(densities, dtype=np.float64)
    electron_repulsion = np.asarray(
        reference.mol.intor("int2e", aosym="s1")
    )
    total_density = densities.sum(axis=0)
    coulomb = np.einsum(
        "mnkl,...lk->...mn",
        electron_repulsion,
        total_density,
    )
    potentials = []
    for spin in range(2):
        exchange = np.einsum(
            "mkln,...lk->...mn",
            electron_repulsion,
            densities[spin],
        )
        potentials.append(coulomb - exchange)
    return tuple(potentials)


@dataclass(frozen=True)
class IndependentUHFAdjointOracle:
    objective_ao_potential: np.ndarray
    alpha_objective_orbital_gradient: np.ndarray
    beta_objective_orbital_gradient: np.ndarray
    one_sided_objective_gradient: np.ndarray
    operator: np.ndarray
    alpha_zvector: np.ndarray
    beta_zvector: np.ndarray
    solver_residual: np.ndarray
    transpose_residual: np.ndarray
    alpha_adjoint_ao_density: np.ndarray
    beta_adjoint_ao_density: np.ndarray
    alpha_adjoint_ao_potential: np.ndarray
    beta_adjoint_ao_potential: np.ndarray
    correction_gradient_metric_spin: np.ndarray
    correction_gradient_metric: np.ndarray
    correction_gradient_adjoint_nuclear_spin: np.ndarray
    correction_gradient_adjoint_nuclear: np.ndarray
    correction_gradient_adjoint_metric_spin: np.ndarray
    correction_gradient_adjoint_metric: np.ndarray
    correction_gradient_occupied_virtual_spin: np.ndarray
    correction_gradient_occupied_virtual: np.ndarray
    correction_gradient_response: np.ndarray
    direct_correction_gradient_metric_spin: np.ndarray
    direct_correction_gradient_occupied_virtual: np.ndarray
    direct_correction_gradient_response: np.ndarray


def build_independent_uhf_adjoint_oracle(
    case: UHFOracleCase,
) -> IndependentUHFAdjointOracle:
    reference = case.reference
    coefficient = np.asarray(reference.mo_coeff)
    energy = np.asarray(reference.mo_energy)
    occupation = np.asarray(reference.mo_occ)
    occupied = occupation > 0.0
    virtual = occupation == 0.0
    objective_ao = independent_objective_ao_potential(case)
    objective_mo = tuple(
        coefficient[spin].T @ objective_ao @ coefficient[spin]
        for spin in range(2)
    )
    objective_gradients = tuple(
        (
            objective_mo[spin][virtual[spin]][:, occupied[spin]]
            + objective_mo[spin].T[virtual[spin]][:, occupied[spin]]
        )
        * occupation[spin, occupied[spin]]
        for spin in range(2)
    )
    one_sided = np.concatenate(
        tuple(
            (
                objective_mo[spin][virtual[spin]][:, occupied[spin]]
                * occupation[spin, occupied[spin]]
            ).reshape(-1)
            for spin in range(2)
        )
    )
    objective_vector = np.concatenate(
        tuple(value.reshape(-1) for value in objective_gradients)
    )
    operator = np.asarray(case.coupled_operator)
    zflat = np.linalg.solve(operator.T, objective_vector)
    alpha_shape = objective_gradients[0].shape
    beta_shape = objective_gradients[1].shape
    alpha_size = int(np.prod(alpha_shape))
    zvector = (
        zflat[:alpha_size].reshape(alpha_shape),
        zflat[alpha_size:].reshape(beta_shape),
    )
    solver_residual = operator.T @ zflat - objective_vector
    independently_rebuilt_operator = _P4A_FIXTURES.independent_coupled_uhf_operator(
        reference
    )
    transpose_residual = independently_rebuilt_operator.T @ zflat - objective_vector
    adjoint_density = []
    for spin in range(2):
        occupied_coefficients = coefficient[spin][:, occupied[spin]]
        virtual_coefficients = coefficient[spin][:, virtual[spin]]
        rotated = virtual_coefficients @ zvector[spin]
        one_sided_density = rotated @ (
            occupied_coefficients * occupation[spin, occupied[spin]]
        ).T
        adjoint_density.append(one_sided_density + one_sided_density.T)
    adjoint_potential = independent_induced_uhf_potential(
        reference,
        np.stack(adjoint_density),
    )
    overlap_derivative = np.asarray(case.overlap_derivative)
    hamiltonian_derivative = (
        np.asarray(case.response.alpha_hamiltonian_derivative),
        np.asarray(case.response.beta_hamiltonian_derivative),
    )
    metric_spin = []
    nuclear_spin = []
    adjoint_metric_spin = []
    for spin in range(2):
        occupied_coefficients = coefficient[spin][:, occupied[spin]]
        overlap_mo = np.einsum(
            "mp,...mn,ni->...pi",
            coefficient[spin],
            overlap_derivative,
            occupied_coefficients,
        )
        hamiltonian_mo = np.einsum(
            "mp,...mn,ni->...pi",
            coefficient[spin],
            hamiltonian_derivative[spin],
            occupied_coefficients,
        )
        bare_rhs = (
            hamiltonian_mo[..., virtual[spin], :]
            - overlap_mo[..., virtual[spin], :]
            * energy[spin, occupied[spin]]
        )
        nuclear_spin.append(
            -np.einsum("ai,...ai->...", zvector[spin], bare_rhs)
        )
        objective_occupied = objective_mo[spin][occupied[spin]][
            :, occupied[spin]
        ]
        objective_occupied = 0.5 * (
            objective_occupied + objective_occupied.T
        )
        potential_mo = (
            coefficient[spin].T
            @ adjoint_potential[spin]
            @ coefficient[spin]
        )
        potential_occupied = potential_mo[occupied[spin]][:, occupied[spin]]
        potential_occupied = 0.5 * (
            potential_occupied + potential_occupied.T
        )
        overlap_occupied = overlap_mo[..., occupied[spin], :]
        metric_spin.append(
            -np.einsum(
                "...ij,ij->...",
                overlap_occupied,
                objective_occupied,
            )
        )
        adjoint_metric_spin.append(
            0.5
            * np.einsum(
                "...ij,ij->...",
                overlap_occupied,
                potential_occupied,
            )
        )
    metric_spin = np.stack(metric_spin)
    nuclear_spin = np.stack(nuclear_spin)
    adjoint_metric_spin = np.stack(adjoint_metric_spin)
    occupied_virtual_spin = nuclear_spin + adjoint_metric_spin
    metric = metric_spin.sum(axis=0)
    nuclear = nuclear_spin.sum(axis=0)
    adjoint_metric = adjoint_metric_spin.sum(axis=0)
    occupied_virtual = occupied_virtual_spin.sum(axis=0)
    response = metric + occupied_virtual
    direct_metric_spin = np.einsum(
        "ij,sbxij->sbx",
        objective_ao,
        np.moveaxis(case.metric_density_response, 2, 0),
    )
    direct_occupied_virtual = np.einsum(
        "ij,bxij->bx",
        objective_ao,
        case.response.total_density_response_occupied_virtual,
    )
    direct_response = np.einsum(
        "ij,bxij->bx",
        objective_ao,
        case.response.total_density_response,
    )
    return IndependentUHFAdjointOracle(
        objective_ao_potential=objective_ao,
        alpha_objective_orbital_gradient=objective_gradients[0],
        beta_objective_orbital_gradient=objective_gradients[1],
        one_sided_objective_gradient=one_sided,
        operator=operator,
        alpha_zvector=zvector[0],
        beta_zvector=zvector[1],
        solver_residual=solver_residual,
        transpose_residual=transpose_residual,
        alpha_adjoint_ao_density=adjoint_density[0],
        beta_adjoint_ao_density=adjoint_density[1],
        alpha_adjoint_ao_potential=adjoint_potential[0],
        beta_adjoint_ao_potential=adjoint_potential[1],
        correction_gradient_metric_spin=metric_spin,
        correction_gradient_metric=metric,
        correction_gradient_adjoint_nuclear_spin=nuclear_spin,
        correction_gradient_adjoint_nuclear=nuclear,
        correction_gradient_adjoint_metric_spin=adjoint_metric_spin,
        correction_gradient_adjoint_metric=adjoint_metric,
        correction_gradient_occupied_virtual_spin=occupied_virtual_spin,
        correction_gradient_occupied_virtual=occupied_virtual,
        correction_gradient_response=response,
        direct_correction_gradient_metric_spin=direct_metric_spin,
        direct_correction_gradient_occupied_virtual=direct_occupied_virtual,
        direct_correction_gradient_response=direct_response,
    )


@pytest.fixture(scope="session")
def independent_uhf_adjoint_oracle(uhf_oracle_case):
    return build_independent_uhf_adjoint_oracle(uhf_oracle_case)
