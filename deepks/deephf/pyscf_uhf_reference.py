"""Isolated PySCF 2.14 adapter for molecular UHF response and adjoints."""

from dataclasses import dataclass
import hashlib

import numpy as np
import pyscf
from pyscf.gto import mole as gto_mole
from pyscf.scf import hf as scf_hf


from .capabilities import DeePHFCapabilityError, transaction_reference_fingerprint
from .adjoint import AdjointError
from functools import partial

from .contracts import (
    dataclass_fingerprint,
    immutable_array as _immutable_array,
    integer_control,
    real_control,
    update_digest as _update_fingerprint_value,
    validated_float64_array,
    version_series as _canonical_version_series,
)


SUPPORTED_PYSCF_SERIES = (2, 14)


class UHFResponseError(RuntimeError):
    """Raised when the UHF response equations fail the strict contract."""


class UHFAdjointError(AdjointError):
    """Raised when a correction-specific UHF adjoint fails its contract."""


def _native_unrestricted_gradient(reference, driver, atom_indices) -> np.ndarray:
    """Evaluate a selected UHF-family gradient around PySCF's broken selector."""
    molecule = reference.mol
    atom_indices = tuple(atom_indices)
    coefficient = np.asarray(reference.mo_coeff)
    energy = np.asarray(reference.mo_energy)
    occupation = np.asarray(reference.mo_occ)
    hcore_derivative = driver.hcore_generator(molecule)
    overlap_derivative = driver.get_ovlp(molecule)
    density = driver._tag_rdm1(
        reference.make_rdm1(coefficient, occupation),
        mo_coeff=coefficient,
        mo_occ=occupation,
    )
    energy_density = driver.make_rdm1e(energy, coefficient, occupation)
    spin_density = density.sum(axis=0)
    spin_energy_density = energy_density.sum(axis=0)
    effective_derivative = driver.get_veff(molecule, density)
    result = np.zeros((len(atom_indices), 3), dtype=np.float64)
    ao_slices = molecule.aoslice_by_atom()
    for result_index, atom_index in enumerate(atom_indices):
        _shell_start, _shell_stop, ao_start, ao_stop = ao_slices[atom_index]
        hcore = hcore_derivative(atom_index)
        result[result_index] += np.einsum("xij,ij->x", hcore, spin_density)
        result[result_index] += 2.0 * np.einsum(
            "sxij,sij->x",
            effective_derivative[:, :, ao_start:ao_stop],
            density[:, ao_start:ao_stop],
        )
        result[result_index] -= 2.0 * np.einsum(
            "xij,ij->x",
            overlap_derivative[:, ao_start:ao_stop],
            spin_energy_density[ao_start:ao_stop],
        )
        grid_response = getattr(effective_derivative, "exc1_grid", None)
        if grid_response is not None:
            result[result_index] += grid_response[atom_index]
    result += driver.grad_nuc(molecule, atmlst=list(atom_indices))
    return _validated_float64_array(
        result,
        (len(atom_indices), 3),
        "native unrestricted gradient",
    )


@dataclass(frozen=True)
class UHFResponseDiagnostics:
    """Independent diagnostics for one complete coupled UHF response solve."""

    minimum_alpha_orbital_gap: float
    minimum_beta_orbital_gap: float
    pyscf_version: str
    cphf_tolerance: float
    maximum_residual: float
    alpha_maximum_residual: float
    beta_maximum_residual: float
    residual_rms: float
    residual_tolerance: float
    invariant_tolerance: float
    orbital_gap_tolerance: float
    max_cycle: int
    max_refinement_cycles: int
    level_shift: float
    response_dimension: int
    alpha_response_dimension: int
    beta_response_dimension: int
    operator_is_self_adjoint: bool
    alpha_metric_residual: float
    beta_metric_residual: float
    alpha_idempotency_residual: float
    beta_idempotency_residual: float
    alpha_particle_number_residual: float
    beta_particle_number_residual: float
    alpha_translation_residual: float | None
    beta_translation_residual: float | None
    translation_residual: float | None
    refinement_cycles: int
    residual_history: tuple[float, ...]


@dataclass(frozen=True)
class UHFResponse:
    """Own canonical spin MO responses; derived properties allocate without caching."""

    reference_identity: int
    state_fingerprint: str
    integrity_fingerprint: str
    atom_indices: tuple[int, ...]
    alpha_mo_response: np.ndarray
    beta_mo_response: np.ndarray
    _mo_coefficients: np.ndarray
    _mo_occupations: np.ndarray
    overlap_derivative: np.ndarray
    alpha_hamiltonian_derivative: np.ndarray
    beta_hamiltonian_derivative: np.ndarray
    alpha_orbital_response_residual: np.ndarray
    beta_orbital_response_residual: np.ndarray
    diagnostics: UHFResponseDiagnostics

    def _mo_partition(self, spin: int, occupied_virtual: bool) -> np.ndarray:
        response = (self.alpha_mo_response, self.beta_mo_response)[spin]
        occupied = self._mo_occupations[spin] > 0
        selected = ~occupied if occupied_virtual else occupied
        result = np.zeros_like(response)
        result[..., selected, :] = response[..., selected, :]
        return _immutable_array(result)

    def _coefficient_response(self, spin: int, response: np.ndarray) -> np.ndarray:
        return _immutable_array(
            np.einsum("mp,...pi->...mi", self._mo_coefficients[spin], response)
        )

    def _density_response(self, spin: int, response: np.ndarray) -> np.ndarray:
        occupied = self._mo_occupations[spin] > 0
        coefficient = self._mo_coefficients[spin]
        coefficient_response = np.einsum("mp,...pi->...mi", coefficient, response)
        density = np.einsum(
            "...pi,qi->...pq",
            coefficient_response,
            coefficient[:, occupied],
        )
        return _immutable_array(density + density.swapaxes(-1, -2))

    @property
    def alpha_mo_response_occupied_virtual(self):
        return self._mo_partition(0, True)

    @property
    def beta_mo_response_occupied_virtual(self):
        return self._mo_partition(1, True)

    @property
    def alpha_mo_response_metric(self):
        return self._mo_partition(0, False)

    @property
    def beta_mo_response_metric(self):
        return self._mo_partition(1, False)

    @property
    def alpha_coefficient_response(self):
        return self._coefficient_response(0, self.alpha_mo_response)

    @property
    def beta_coefficient_response(self):
        return self._coefficient_response(1, self.beta_mo_response)

    @property
    def alpha_coefficient_response_occupied_virtual(self):
        return self._coefficient_response(0, self.alpha_mo_response_occupied_virtual)

    @property
    def beta_coefficient_response_occupied_virtual(self):
        return self._coefficient_response(1, self.beta_mo_response_occupied_virtual)

    @property
    def alpha_coefficient_response_metric(self):
        return self._coefficient_response(0, self.alpha_mo_response_metric)

    @property
    def beta_coefficient_response_metric(self):
        return self._coefficient_response(1, self.beta_mo_response_metric)

    @property
    def alpha_density_response(self):
        return self._density_response(0, self.alpha_mo_response)

    @property
    def beta_density_response(self):
        return self._density_response(1, self.beta_mo_response)

    @property
    def total_density_response(self):
        return _immutable_array(self.alpha_density_response + self.beta_density_response)

    @property
    def alpha_density_response_occupied_virtual(self):
        return self._density_response(0, self.alpha_mo_response_occupied_virtual)

    @property
    def beta_density_response_occupied_virtual(self):
        return self._density_response(1, self.beta_mo_response_occupied_virtual)

    @property
    def total_density_response_occupied_virtual(self):
        return _immutable_array(
            self.alpha_density_response_occupied_virtual
            + self.beta_density_response_occupied_virtual
        )

    @property
    def alpha_density_response_metric(self):
        return self._density_response(0, self.alpha_mo_response_metric)

    @property
    def beta_density_response_metric(self):
        return self._density_response(1, self.beta_mo_response_metric)

    @property
    def total_density_response_metric(self):
        return _immutable_array(
            self.alpha_density_response_metric + self.beta_density_response_metric
        )


@dataclass(frozen=True)
class UHFAdjointDiagnostics:
    """Independent diagnostics for one coupled alpha/beta UHF adjoint."""

    minimum_alpha_orbital_gap: float
    minimum_beta_orbital_gap: float
    pyscf_version: str
    residual_tolerance: float
    invariant_tolerance: float
    orbital_gap_tolerance: float
    response_dimension: int
    alpha_response_dimension: int
    beta_response_dimension: int
    operator_is_self_adjoint: bool
    objective_symmetry_tolerance: float
    objective_symmetry_residual: float
    alpha_adjoint_density_symmetry_residual: float
    beta_adjoint_density_symmetry_residual: float
    alpha_adjoint_potential_symmetry_residual: float
    beta_adjoint_potential_symmetry_residual: float
    gradient_reconstruction_residual: float
    solver: str
    solve_count: int
    objective_gradient_norm: float
    solution_norm: float
    maximum_residual: float
    residual_rms: float
    max_cycle: int
    krylov_restart: int
    iteration_count: int


@dataclass(frozen=True)
class UHFAdjoint:
    """Immutable coupled UHF Z-vectors and nuclear contractions."""

    reference_identity: int
    state_fingerprint: str
    integrity_fingerprint: str
    operator_fingerprint: str
    atom_indices: tuple[int, ...]
    objective_ao_potential: np.ndarray
    alpha_objective_orbital_gradient: np.ndarray
    beta_objective_orbital_gradient: np.ndarray
    alpha_zvector: np.ndarray
    beta_zvector: np.ndarray
    alpha_residual: np.ndarray
    beta_residual: np.ndarray
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
    diagnostics: UHFAdjointDiagnostics


def _version_series(version: str) -> tuple[int, int]:
    return _canonical_version_series(version, DeePHFCapabilityError)


def validate_pyscf_version() -> None:
    """Require the PySCF series characterized by the UHF adapter."""
    series = _version_series(pyscf.__version__)
    if series != SUPPORTED_PYSCF_SERIES:
        raise DeePHFCapabilityError(
            "the UHF response adapter supports PySCF 2.14; "
            f"found {pyscf.__version__}"
        )


def _direct_effective_potential(molecule, density: np.ndarray) -> np.ndarray:
    try:
        coulomb, exchange = scf_hf.get_jk(molecule, density, hermi=1)
    except Exception as error:
        raise UHFResponseError(
            f"PySCF native UHF integral evaluation failed: {error}"
        ) from error
    coulomb = np.asarray(coulomb)
    exchange = np.asarray(exchange)
    total_coulomb = coulomb[0] + coulomb[1]
    return np.asarray(
        (total_coulomb - exchange[0], total_coulomb - exchange[1])
    )


def validate_uhf_reference(reference):
    from .audits.uhf_reference import validate_uhf_reference as audit
    return audit(reference)


def uhf_molecule_science_fingerprint(molecule) -> str:
    """Fingerprint stable molecular geometry and AO data."""
    if type(molecule) is not gto_mole.Mole:
        raise DeePHFCapabilityError(
            "UHF science-state fingerprints require a native pyscf.gto.Mole"
        )
    environment = np.asarray(molecule._env).copy()
    environment[: gto_mole.PTR_ENV_START] = 0.0
    digest = hashlib.sha256()
    values = (
        pyscf.__version__,
        f"{type(molecule).__module__}.{type(molecule).__qualname__}",
        bool(molecule._built),
        int(molecule.natm),
        int(molecule.nao),
        int(molecule.nbas),
        int(molecule.charge),
        int(molecule.spin),
        int(molecule.nelectron),
        bool(molecule.cart),
        molecule.symmetry,
        float(getattr(molecule, "omega", 0.0)),
        getattr(molecule, "nucmod", None),
        tuple(molecule.atom_symbol(index) for index in range(molecule.natm)),
        np.asarray(molecule.atom_charges()),
        np.asarray(gto_mole.Mole.atom_coords(molecule, unit="Bohr")),
        tuple(molecule.ao_labels()),
        np.asarray(molecule._atm),
        np.asarray(molecule._bas),
        environment,
        molecule._basis,
        molecule._ecp,
        getattr(molecule, "_pseudo", None),
    )
    for value in values:
        _update_fingerprint_value(digest, value)
    return digest.hexdigest()


def uhf_reference_fingerprint(reference) -> str:
    """Return a scratch-independent fingerprint of the scientific UHF state."""
    trusted = transaction_reference_fingerprint(reference)
    if trusted is not None:
        return trusted
    digest = hashlib.sha256()
    values = (
        f"{type(reference).__module__}.{type(reference).__qualname__}",
        bool(reference.converged),
        uhf_molecule_science_fingerprint(reference.mol),
        float(reference.e_tot),
        np.asarray(reference.mo_coeff),
        np.asarray(reference.mo_energy),
        np.asarray(reference.mo_occ),
        np.asarray(reference.make_rdm1()),
    )
    for value in values:
        _update_fingerprint_value(digest, value)
    return digest.hexdigest()


def uhf_response_integrity_fingerprint(response: UHFResponse) -> str:
    """Return a digest covering every UHF response field except itself."""
    return dataclass_fingerprint(
        response,
        excluded=frozenset({"integrity_fingerprint"}),
    )


def uhf_adjoint_integrity_fingerprint(adjoint: UHFAdjoint) -> str:
    """Return a digest covering every UHF adjoint field except itself."""
    return dataclass_fingerprint(
        adjoint,
        excluded=frozenset({"integrity_fingerprint"}),
    )


_validated_float64_array = partial(
    validated_float64_array,
    error_type=UHFResponseError,
)


def _validated_response_array(value, expected_shape, name: str) -> np.ndarray:
    if type(value) is not np.ndarray:
        raise UHFResponseError(f"the supplied UHF response {name} is not an ndarray")
    array = _validated_float64_array(value, expected_shape, name)
    if array.flags.writeable:
        raise UHFResponseError(
            f"the supplied UHF response {name} must be immutable"
        )
    return array


_cycle_limit = integer_control
_response_real_control = real_control
