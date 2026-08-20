"""Isolated PySCF 2.14 adapter for molecular UHF nuclear response."""

from dataclasses import dataclass, fields, replace
import hashlib
from numbers import Real
import operator
from typing import Any

import numpy as np
import pyscf
from pyscf import gto
from pyscf.gto import mole as gto_mole
from pyscf.hessian import uhf as uhf_hessian
from pyscf.scf import hf as scf_hf
from pyscf.scf import ucphf, uhf as scf_uhf

from deepks.descriptor import is_ghost_atom

from .capabilities import DeePHFCapabilityError


SUPPORTED_PYSCF_SERIES = (2, 14)


class UHFResponseError(RuntimeError):
    """Raised when the UHF response equations fail the strict contract."""


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
    operator_stability_tolerance: float
    operator_condition_tolerance: float
    operator_symmetry_tolerance: float
    operator_dimension_limit: int
    operator_minimum_eigenvalue: float
    operator_maximum_eigenvalue: float
    operator_condition_number: float
    operator_symmetry_residual: float
    alpha_metric_residual: float
    beta_metric_residual: float
    alpha_idempotency_residual: float
    beta_idempotency_residual: float
    alpha_particle_number_residual: float
    beta_particle_number_residual: float
    density_reconstruction_residual: float
    alpha_translation_residual: float
    beta_translation_residual: float
    translation_residual: float
    refinement_cycles: int
    residual_history: tuple[float, ...]


@dataclass(frozen=True)
class UHFResponse:
    """Complete spin-resolved first-order UHF state for nuclear coordinates."""

    reference_identity: int
    state_fingerprint: str
    integrity_fingerprint: str
    alpha_mo_response: np.ndarray
    beta_mo_response: np.ndarray
    alpha_mo_response_occupied_virtual: np.ndarray
    beta_mo_response_occupied_virtual: np.ndarray
    alpha_mo_response_metric: np.ndarray
    beta_mo_response_metric: np.ndarray
    alpha_coefficient_response: np.ndarray
    beta_coefficient_response: np.ndarray
    alpha_coefficient_response_occupied_virtual: np.ndarray
    beta_coefficient_response_occupied_virtual: np.ndarray
    alpha_coefficient_response_metric: np.ndarray
    beta_coefficient_response_metric: np.ndarray
    alpha_density_response: np.ndarray
    beta_density_response: np.ndarray
    total_density_response: np.ndarray
    alpha_density_response_occupied_virtual: np.ndarray
    beta_density_response_occupied_virtual: np.ndarray
    total_density_response_occupied_virtual: np.ndarray
    alpha_density_response_metric: np.ndarray
    beta_density_response_metric: np.ndarray
    total_density_response_metric: np.ndarray
    overlap_derivative: np.ndarray
    alpha_hamiltonian_derivative: np.ndarray
    beta_hamiltonian_derivative: np.ndarray
    alpha_orbital_response_residual: np.ndarray
    beta_orbital_response_residual: np.ndarray
    diagnostics: UHFResponseDiagnostics


def _version_series(version: str) -> tuple[int, int]:
    components = version.split(".")
    try:
        return int(components[0]), int(components[1])
    except (IndexError, ValueError) as error:
        raise DeePHFCapabilityError(
            f"cannot interpret the PySCF version {version!r}"
        ) from error


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
    """Validate the native real-orbital integer-occupation UHF contract."""
    if type(reference) is not scf_uhf.UHF:
        raise DeePHFCapabilityError(
            "UHF DeePHF requires an undecorated native pyscf.scf.uhf.UHF reference"
        )
    if not reference.converged:
        raise DeePHFCapabilityError("the UHF reference must be converged")
    molecule = reference.mol
    if type(molecule) is not gto_mole.Mole:
        raise DeePHFCapabilityError(
            "the UHF reference must use a native molecular pyscf.gto.Mole"
        )
    if molecule.symmetry:
        raise DeePHFCapabilityError(
            "the UHF reference must not use symmetry-constrained occupations"
        )
    if molecule.cart:
        raise DeePHFCapabilityError(
            "the initial UHF force contract requires spherical AO functions"
        )
    if getattr(molecule, "_pseudo", None):
        raise DeePHFCapabilityError(
            "the initial UHF force contract does not support pseudopotentials"
        )
    if getattr(molecule, "_ecp", None) or molecule.has_ecp():
        raise DeePHFCapabilityError(
            "the initial UHF force contract requires an all-electron reference"
        )
    if float(getattr(molecule, "omega", 0.0)) != 0.0:
        raise DeePHFCapabilityError(
            "the initial UHF force contract requires the full Coulomb interaction"
        )
    if getattr(molecule, "nucmod", None):
        raise DeePHFCapabilityError(
            "the initial UHF force contract requires point nuclei"
        )
    ghost_indices = [
        atom_index
        for atom_index in range(molecule.natm)
        if is_ghost_atom(molecule, atom_index)
    ]
    if ghost_indices:
        raise DeePHFCapabilityError(
            "the initial UHF force contract requires real atoms; ghost indices: "
            f"{ghost_indices}"
        )
    decorated_attributes = {
        "density fitting": "with_df",
        "solvent": "with_solvent",
        "X2C": "with_x2c",
        "QM/MM": "mm_mol",
        "dispersion": "disp",
        "penalty": "penalties",
    }
    active_decorations = [
        name
        for name, attribute in decorated_attributes.items()
        if getattr(reference, attribute, None)
    ]
    if active_decorations:
        raise DeePHFCapabilityError(
            "the UHF reference has unsupported decorations: "
            + ", ".join(active_decorations)
        )
    custom_hooks = sorted(
        name
        for name, value in reference.__dict__.items()
        if name != "mol" and callable(value)
    )
    if custom_hooks:
        raise DeePHFCapabilityError(
            "the UHF reference has unsupported instance hooks: "
            + ", ".join(custom_hooks)
        )
    molecule_hooks = sorted(
        name for name, value in molecule.__dict__.items() if callable(value)
    )
    if molecule_hooks:
        raise DeePHFCapabilityError(
            "the UHF molecule has unsupported instance hooks: "
            + ", ".join(molecule_hooks)
        )
    if reference.mo_coeff is None or reference.mo_energy is None:
        raise DeePHFCapabilityError("the UHF reference orbital state is incomplete")
    if reference.mo_occ is None:
        raise DeePHFCapabilityError("the UHF reference occupations are missing")
    coefficient = np.asarray(reference.mo_coeff)
    energy = np.asarray(reference.mo_energy)
    occupation = np.asarray(reference.mo_occ)
    orbital_values = (coefficient, energy, occupation)
    if any(np.iscomplexobj(value) for value in orbital_values):
        raise DeePHFCapabilityError("the UHF orbitals must be real")
    if any(value.dtype != np.dtype(np.float64) for value in orbital_values):
        raise DeePHFCapabilityError(
            "the UHF orbital state must use numpy.float64"
        )
    if not all(np.isfinite(value).all() for value in orbital_values):
        raise DeePHFCapabilityError("the UHF orbital state must be finite")
    expected_coefficient_shape = (2, molecule.nao, molecule.nao)
    if coefficient.shape != expected_coefficient_shape:
        raise DeePHFCapabilityError(
            "the UHF response requires two complete square MO coefficient matrices"
        )
    expected_orbital_shape = (2, molecule.nao)
    if energy.shape != expected_orbital_shape:
        raise DeePHFCapabilityError("the UHF orbital energy shape is invalid")
    if occupation.shape != expected_orbital_shape:
        raise DeePHFCapabilityError("the UHF occupation shape is invalid")
    if not np.all(np.isin(occupation, (0.0, 1.0))):
        raise DeePHFCapabilityError(
            "the UHF occupations must be integer spin-orbital occupations"
        )
    expected_electrons = tuple(int(value) for value in molecule.nelec)
    actual_electrons = tuple(int(value) for value in occupation.sum(axis=1))
    if actual_electrons != expected_electrons:
        raise DeePHFCapabilityError(
            "the UHF occupations do not match the molecular alpha and beta electron counts"
        )
    if sum(actual_electrons) != molecule.nelectron:
        raise DeePHFCapabilityError(
            "the UHF occupations do not match the molecular electron count"
        )
    if actual_electrons[0] - actual_electrons[1] != molecule.spin:
        raise DeePHFCapabilityError(
            "the UHF occupations do not match the molecular spin"
        )
    for spin_index, spin_name in enumerate(("alpha", "beta")):
        occupied_count = expected_electrons[spin_index]
        expected_occupation = np.zeros_like(occupation[spin_index])
        expected_occupation[:occupied_count] = 1.0
        if not np.array_equal(occupation[spin_index], expected_occupation):
            raise DeePHFCapabilityError(
                "the initial UHF force contract requires the Aufbau ground-state "
                f"root in the {spin_name} channel"
            )
        if occupied_count == 0 or occupied_count == molecule.nao:
            raise DeePHFCapabilityError(
                "UHF response requires occupied and virtual orbitals in each spin channel"
            )
        if np.any(np.diff(energy[spin_index]) < -1.0e-10):
            raise DeePHFCapabilityError(
                f"the UHF {spin_name} canonical orbital energies are not ordered"
            )
        root_gap = float(
            energy[spin_index, occupied_count]
            - energy[spin_index, occupied_count - 1]
        )
        if root_gap <= 0.0:
            raise DeePHFCapabilityError(
                f"the UHF {spin_name} occupied and virtual root spaces overlap"
            )
    if not np.isfinite(reference.e_tot):
        raise DeePHFCapabilityError("the UHF reference energy must be finite")
    try:
        overlap = np.asarray(reference.get_ovlp())
        hcore = np.asarray(reference.get_hcore())
        density = np.asarray(reference.make_rdm1())
        effective_potential = np.asarray(
            reference.get_veff(molecule, density)
        )
        direct_effective_potential = _direct_effective_potential(
            molecule,
            density,
        )
    except UHFResponseError as error:
        raise DeePHFCapabilityError(
            f"the UHF reference matrices could not be evaluated: {error}"
        ) from error
    except Exception as error:
        raise DeePHFCapabilityError(
            f"the UHF reference matrices could not be evaluated: {error}"
        ) from error
    ao_values = (
        overlap,
        hcore,
        density,
        effective_potential,
        direct_effective_potential,
    )
    if any(np.iscomplexobj(value) for value in ao_values):
        raise DeePHFCapabilityError("the UHF AO matrices must be real")
    if any(value.dtype != np.dtype(np.float64) for value in ao_values):
        raise DeePHFCapabilityError("the UHF AO matrices must use numpy.float64")
    if not all(np.isfinite(value).all() for value in ao_values):
        raise DeePHFCapabilityError("the UHF AO matrices must be finite")
    expected_ao_shape = (molecule.nao, molecule.nao)
    if overlap.shape != expected_ao_shape or hcore.shape != expected_ao_shape:
        raise DeePHFCapabilityError("the UHF spin-independent AO matrix shape is invalid")
    expected_spin_ao_shape = (2, molecule.nao, molecule.nao)
    if any(
        value.shape != expected_spin_ao_shape
        for value in (density, effective_potential, direct_effective_potential)
    ):
        raise DeePHFCapabilityError("the UHF spin-resolved AO matrix shape is invalid")
    interaction_error = float(
        np.max(
            np.abs(effective_potential - direct_effective_potential),
            initial=0.0,
        )
    )
    if interaction_error > 1.0e-10:
        raise DeePHFCapabilityError(
            "the UHF two-electron interaction does not match the native molecular "
            f"integrals: residual {interaction_error:.3e}"
        )
    overlap_eigenvalues = np.linalg.eigvalsh(overlap)
    if overlap_eigenvalues[0] <= 1.0e-10:
        raise DeePHFCapabilityError(
            "the UHF AO overlap is singular or ill conditioned"
        )
    fock = hcore[None, :, :] + direct_effective_potential
    for spin_index, spin_name in enumerate(("alpha", "beta")):
        spin_coefficient = coefficient[spin_index]
        spin_density = density[spin_index]
        orthonormality_error = float(
            np.max(
                np.abs(
                    spin_coefficient.T @ overlap @ spin_coefficient
                    - np.eye(molecule.nao)
                ),
                initial=0.0,
            )
        )
        if orthonormality_error > 1.0e-8:
            raise DeePHFCapabilityError(
                f"the UHF {spin_name} orbitals violate AO-metric orthonormality: "
                f"{orthonormality_error:.3e}"
            )
        density_symmetry_error = float(
            np.max(np.abs(spin_density - spin_density.T), initial=0.0)
        )
        if density_symmetry_error > 1.0e-10:
            raise DeePHFCapabilityError(
                f"the UHF {spin_name} density violates symmetry"
            )
        electron_count = float(np.einsum("ij,ji->", spin_density, overlap))
        if not np.isclose(
            electron_count,
            expected_electrons[spin_index],
            rtol=0.0,
            atol=1.0e-8,
        ):
            raise DeePHFCapabilityError(
                f"the UHF {spin_name} AO density has an inconsistent electron count: "
                f"{electron_count:.12g}"
            )
        idempotency_error = float(
            np.max(
                np.abs(spin_density @ overlap @ spin_density - spin_density),
                initial=0.0,
            )
        )
        if idempotency_error > 1.0e-8:
            raise DeePHFCapabilityError(
                f"the UHF {spin_name} AO density violates metric idempotency: "
                f"{idempotency_error:.3e}"
            )
        canonical_residual = (
            fock[spin_index] @ spin_coefficient
            - overlap
            @ (spin_coefficient * energy[spin_index])
        )
        maximum_canonical_residual = float(
            np.max(np.abs(canonical_residual), initial=0.0)
        )
        if maximum_canonical_residual > 1.0e-7:
            raise DeePHFCapabilityError(
                f"the stored UHF {spin_name} orbitals and energies do not satisfy "
                "the canonical SCF equations: residual "
                f"{maximum_canonical_residual:.3e}"
            )
    recomputed_energy = (
        np.einsum("sij,ji->", density, hcore)
        + 0.5 * np.einsum("sij,sji->", density, direct_effective_potential)
        + molecule.energy_nuc()
    )
    if not np.isclose(
        recomputed_energy,
        reference.e_tot,
        rtol=0.0,
        atol=1.0e-8,
    ):
        raise DeePHFCapabilityError(
            "the stored UHF total energy is inconsistent with its AO state: "
            f"{reference.e_tot:.12g} != {recomputed_energy:.12g}"
        )
    coordinates = np.asarray(molecule.atom_coords(unit="Bohr"))
    if coordinates.dtype != np.dtype(np.float64) or not np.isfinite(coordinates).all():
        raise DeePHFCapabilityError("the molecular geometry must be finite float64")
    return reference


def _update_fingerprint_value(digest, value: Any) -> None:
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(repr(array.shape).encode("ascii"))
        digest.update(array.tobytes())
        return
    if isinstance(value, np.generic):
        _update_fingerprint_value(digest, value.item())
        return
    if isinstance(value, dict):
        for key in sorted(value, key=lambda item: (type(item).__name__, repr(item))):
            _update_fingerprint_value(digest, key)
            _update_fingerprint_value(digest, value[key])
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            _update_fingerprint_value(digest, item)
        return
    if value is None or isinstance(value, (bool, int, float, str)):
        digest.update(type(value).__name__.encode("ascii"))
        digest.update(repr(value).encode("utf-8"))
        return
    digest.update(type(value).__name__.encode("utf-8"))
    digest.update(repr(value).encode("utf-8"))


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
    digest = hashlib.sha256()
    for response_field in fields(response):
        if response_field.name == "integrity_fingerprint":
            continue
        value = getattr(response, response_field.name)
        digest.update(response_field.name.encode("utf-8"))
        if isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            digest.update(array.dtype.str.encode("ascii"))
            digest.update(repr(array.shape).encode("ascii"))
            digest.update(array.tobytes())
        else:
            digest.update(repr(value).encode("utf-8"))
    return digest.hexdigest()


def _immutable_array(value: np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(value)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _validated_float64_array(value, expected_shape, name: str) -> np.ndarray:
    try:
        array = np.asarray(value)
    except Exception as error:
        raise UHFResponseError(f"{name} is not a numerical array: {error}") from error
    if array.shape != expected_shape:
        raise UHFResponseError(
            f"unexpected {name} shape {array.shape}; expected {expected_shape}"
        )
    if array.dtype != np.dtype(np.float64) or np.iscomplexobj(array):
        raise UHFResponseError(f"{name} must be a real float64 array")
    if not np.isfinite(array).all():
        raise UHFResponseError(f"{name} must be finite")
    return array


def _validated_response_array(value, expected_shape, name: str) -> np.ndarray:
    if type(value) is not np.ndarray:
        raise UHFResponseError(f"the supplied UHF response {name} is not an ndarray")
    array = _validated_float64_array(value, expected_shape, name)
    if array.flags.writeable:
        raise UHFResponseError(
            f"the supplied UHF response {name} must be immutable"
        )
    return array


def _cycle_limit(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"response {name} must be an integer")
    try:
        return operator.index(value)
    except TypeError as error:
        raise ValueError(f"response {name} must be an integer") from error


def _response_real_control(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"response {name} must be a real numeric scalar")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"response {name} must be finite")
    return result


class UHFResponseAdapter:
    """Solve and independently audit molecular UHF nuclear UC-PHF response."""

    def __init__(
        self,
        reference,
        *,
        cphf_tolerance: float = 1.0e-11,
        residual_tolerance: float = 1.0e-9,
        invariant_tolerance: float = 1.0e-9,
        orbital_gap_tolerance: float = 1.0e-7,
        max_cycle: int = 100,
        max_refinement_cycles: int = 3,
        level_shift: float = 0.0,
        operator_stability_tolerance: float = 1.0e-6,
        operator_condition_tolerance: float = 1.0e8,
        operator_symmetry_tolerance: float = 1.0e-10,
        operator_dimension_limit: int = 512,
    ):
        validate_pyscf_version()
        self.reference = validate_uhf_reference(reference)
        self.cphf_tolerance = _response_real_control(
            cphf_tolerance,
            "cphf_tolerance",
        )
        self.residual_tolerance = _response_real_control(
            residual_tolerance,
            "residual_tolerance",
        )
        self.invariant_tolerance = _response_real_control(
            invariant_tolerance,
            "invariant_tolerance",
        )
        self.orbital_gap_tolerance = _response_real_control(
            orbital_gap_tolerance,
            "orbital_gap_tolerance",
        )
        self.max_cycle = _cycle_limit(max_cycle, "max_cycle")
        self.max_refinement_cycles = _cycle_limit(
            max_refinement_cycles,
            "max_refinement_cycles",
        )
        self.level_shift = _response_real_control(level_shift, "level_shift")
        self.operator_stability_tolerance = _response_real_control(
            operator_stability_tolerance,
            "operator_stability_tolerance",
        )
        self.operator_condition_tolerance = _response_real_control(
            operator_condition_tolerance,
            "operator_condition_tolerance",
        )
        self.operator_symmetry_tolerance = _response_real_control(
            operator_symmetry_tolerance,
            "operator_symmetry_tolerance",
        )
        self.operator_dimension_limit = _cycle_limit(
            operator_dimension_limit,
            "operator_dimension_limit",
        )
        tolerance_values = (
            self.cphf_tolerance,
            self.residual_tolerance,
            self.invariant_tolerance,
            self.orbital_gap_tolerance,
            self.operator_stability_tolerance,
            self.operator_condition_tolerance,
            self.operator_symmetry_tolerance,
        )
        if any(value <= 0 for value in tolerance_values[:5]):
            raise ValueError("response tolerances must be positive")
        if self.operator_condition_tolerance <= 1:
            raise ValueError("operator_condition_tolerance must exceed one")
        if self.operator_symmetry_tolerance <= 0:
            raise ValueError("operator_symmetry_tolerance must be positive")
        if self.max_cycle <= 0 or self.max_refinement_cycles < 0:
            raise ValueError("response cycle limits are invalid")
        if self.operator_dimension_limit <= 0:
            raise ValueError("operator_dimension_limit must be positive")

    @property
    def molecule(self):
        return self.reference.mol

    def _state(self):
        coefficient = np.asarray(self.reference.mo_coeff)
        energy = np.asarray(self.reference.mo_energy)
        occupation = np.asarray(self.reference.mo_occ)
        occupied = occupation > 0
        virtual = occupation == 0
        minimum_gaps = []
        for spin_index, spin_name in enumerate(("alpha", "beta")):
            gaps = (
                energy[spin_index, virtual[spin_index], None]
                - energy[spin_index, occupied[spin_index]]
            )
            minimum_gap = float(np.min(gaps))
            if (
                not np.isfinite(minimum_gap)
                or minimum_gap <= self.orbital_gap_tolerance
            ):
                raise DeePHFCapabilityError(
                    f"UHF {spin_name} occupied-virtual gap is outside the strict "
                    f"response domain: {minimum_gap:.3e} <= "
                    f"{self.orbital_gap_tolerance:.3e}"
                )
            minimum_gaps.append(minimum_gap)
        return coefficient, energy, occupation, occupied, virtual, tuple(minimum_gaps)

    def _overlap_derivative(self) -> np.ndarray:
        molecule = self.molecule
        try:
            integral = -molecule.intor("int1e_ipovlp", comp=3)
        except Exception as error:
            raise UHFResponseError(
                f"PySCF overlap-derivative construction failed: {error}"
            ) from error
        integral = _validated_float64_array(
            integral,
            (3, molecule.nao, molecule.nao),
            "overlap-derivative integral",
        )
        result = np.zeros((molecule.natm, 3, molecule.nao, molecule.nao))
        for atom_index, atom_slice in enumerate(molecule.aoslice_by_atom()):
            ao_start, ao_stop = atom_slice[2:]
            result[atom_index, :, ao_start:ao_stop] += integral[
                :, ao_start:ao_stop
            ]
            result[atom_index, :, :, ao_start:ao_stop] += integral[
                :, ao_start:ao_stop
            ].transpose(0, 2, 1)
        return result

    def _hamiltonian_derivative(
        self,
        coefficient: np.ndarray,
        occupation: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        try:
            derivatives = uhf_hessian.Hessian(self.reference).make_h1(
                coefficient,
                occupation,
                atmlst=range(self.molecule.natm),
            )
        except Exception as error:
            raise UHFResponseError(
                f"PySCF UHF Hamiltonian derivative construction failed: {error}"
            ) from error
        if derivatives is None or len(derivatives) != 2:
            raise UHFResponseError("PySCF UHF Hamiltonian derivative is incomplete")
        expected = (self.molecule.natm, 3, self.molecule.nao, self.molecule.nao)
        return tuple(
            _validated_float64_array(
                spin_derivative,
                expected,
                f"{spin_name} Hamiltonian derivative",
            )
            for spin_derivative, spin_name in zip(
                derivatives,
                ("alpha", "beta"),
                strict=True,
            )
        )

    @staticmethod
    def _density_from_mo_response(
        mo_response: np.ndarray,
        coefficient: np.ndarray,
        occupied: np.ndarray,
    ) -> np.ndarray:
        occupied_coefficient = coefficient[:, occupied]
        coefficient_response = np.einsum(
            "mp,...pi->...mi",
            coefficient,
            mo_response,
        )
        one_sided = np.einsum(
            "...pi,qi->...pq",
            coefficient_response,
            occupied_coefficient,
        )
        return one_sided + one_sided.swapaxes(-1, -2)

    def _induced_potential(
        self,
        alpha_density: np.ndarray,
        beta_density: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        perturbation_shape = alpha_density.shape[:-2]
        flat_alpha = np.asarray(alpha_density).reshape(
            -1,
            self.molecule.nao,
            self.molecule.nao,
        )
        flat_beta = np.asarray(beta_density).reshape(flat_alpha.shape)
        density = np.stack((flat_alpha, flat_beta), axis=0)
        try:
            potential = _direct_effective_potential(self.molecule, density)
        except Exception as error:
            if isinstance(error, UHFResponseError):
                raise
            raise UHFResponseError(
                f"PySCF induced UHF potential construction failed: {error}"
            ) from error
        potential = _validated_float64_array(
            potential,
            (2, *flat_alpha.shape),
            "induced UHF potential",
        )
        expected = (*perturbation_shape, self.molecule.nao, self.molecule.nao)
        return potential[0].reshape(expected), potential[1].reshape(expected)

    def _induced_mo_potential(
        self,
        alpha_response: np.ndarray,
        beta_response: np.ndarray,
        coefficient: np.ndarray,
        occupied: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        alpha_density = self._density_from_mo_response(
            alpha_response,
            coefficient[0],
            occupied[0],
        )
        beta_density = self._density_from_mo_response(
            beta_response,
            coefficient[1],
            occupied[1],
        )
        alpha_potential, beta_potential = self._induced_potential(
            alpha_density,
            beta_density,
        )
        alpha_mo = np.einsum(
            "mp,...mn,ni->...pi",
            coefficient[0],
            alpha_potential,
            coefficient[0][:, occupied[0]],
        )
        beta_mo = np.einsum(
            "mp,...mn,ni->...pi",
            coefficient[1],
            beta_potential,
            coefficient[1][:, occupied[1]],
        )
        return alpha_mo, beta_mo

    @staticmethod
    def _dimensions(occupied, virtual) -> tuple[int, int, int, int, int, int]:
        alpha_nocc = int(np.count_nonzero(occupied[0]))
        beta_nocc = int(np.count_nonzero(occupied[1]))
        alpha_nvir = int(np.count_nonzero(virtual[0]))
        beta_nvir = int(np.count_nonzero(virtual[1]))
        alpha_dimension = alpha_nocc * alpha_nvir
        beta_dimension = beta_nocc * beta_nvir
        return (
            alpha_nocc,
            beta_nocc,
            alpha_nvir,
            beta_nvir,
            alpha_dimension,
            beta_dimension,
        )

    def _split_occupied_virtual(self, vectors, occupied, virtual):
        (
            alpha_nocc,
            beta_nocc,
            alpha_nvir,
            beta_nvir,
            alpha_dimension,
            beta_dimension,
        ) = self._dimensions(occupied, virtual)
        vectors = np.asarray(vectors)
        expected_dimension = alpha_dimension + beta_dimension
        if vectors.shape[-1] != expected_dimension:
            raise UHFResponseError(
                "the coupled UHF occupied-virtual response has an invalid shape"
            )
        alpha = vectors[..., :alpha_dimension].reshape(
            *vectors.shape[:-1],
            alpha_nvir,
            alpha_nocc,
        )
        beta = vectors[..., alpha_dimension:].reshape(
            *vectors.shape[:-1],
            beta_nvir,
            beta_nocc,
        )
        return alpha, beta

    def _apply_occupied_virtual_operator(
        self,
        vectors: np.ndarray,
        coefficient: np.ndarray,
        energy: np.ndarray,
        occupied: np.ndarray,
        virtual: np.ndarray,
    ) -> np.ndarray:
        alpha, beta = self._split_occupied_virtual(vectors, occupied, virtual)
        alpha_full = np.zeros(
            (*alpha.shape[:-2], coefficient.shape[2], alpha.shape[-1]),
            dtype=np.float64,
        )
        beta_full = np.zeros(
            (*beta.shape[:-2], coefficient.shape[2], beta.shape[-1]),
            dtype=np.float64,
        )
        alpha_full[..., virtual[0], :] = alpha
        beta_full[..., virtual[1], :] = beta
        induced_alpha, induced_beta = self._induced_mo_potential(
            alpha_full,
            beta_full,
            coefficient,
            occupied,
        )
        alpha_image = (
            (
                energy[0, virtual[0], None]
                - energy[0, occupied[0]]
            )
            * alpha
            + induced_alpha[..., virtual[0], :]
        )
        beta_image = (
            (
                energy[1, virtual[1], None]
                - energy[1, occupied[1]]
            )
            * beta
            + induced_beta[..., virtual[1], :]
        )
        return np.concatenate(
            (
                alpha_image.reshape(*alpha.shape[:-2], -1),
                beta_image.reshape(*beta.shape[:-2], -1),
            ),
            axis=-1,
        )

    def _response_operator_diagnostics(
        self,
        coefficient: np.ndarray,
        energy: np.ndarray,
        occupied: np.ndarray,
        virtual: np.ndarray,
    ) -> tuple[int, int, int, float, float, float, float]:
        dimensions = self._dimensions(occupied, virtual)
        alpha_dimension, beta_dimension = dimensions[-2:]
        dimension = alpha_dimension + beta_dimension
        if dimension > self.operator_dimension_limit:
            raise DeePHFCapabilityError(
                "UHF coupled occupied-virtual response dimension exceeds the "
                f"condition-audit limit: {dimension} > {self.operator_dimension_limit}"
            )
        identity = np.eye(dimension, dtype=np.float64)
        matrix = np.empty((dimension, dimension), dtype=np.float64)
        batch_size = min(64, dimension)
        for start in range(0, dimension, batch_size):
            stop = min(start + batch_size, dimension)
            images = self._apply_occupied_virtual_operator(
                identity[start:stop],
                coefficient,
                energy,
                occupied,
                virtual,
            )
            matrix[:, start:stop] = images.T
        if not np.isfinite(matrix).all():
            raise UHFResponseError(
                "the coupled UHF occupied-virtual response operator is nonfinite"
            )
        symmetry_residual = float(
            np.max(np.abs(matrix - matrix.T), initial=0.0)
        )
        if symmetry_residual > self.operator_symmetry_tolerance:
            raise UHFResponseError(
                "the coupled UHF occupied-virtual response operator violates symmetry: "
                f"{symmetry_residual:.3e} > {self.operator_symmetry_tolerance:.3e}"
            )
        try:
            eigenvalues = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
        except np.linalg.LinAlgError as error:
            raise UHFResponseError(
                f"the coupled UHF response-operator eigensolve failed: {error}"
            ) from error
        minimum_eigenvalue = float(eigenvalues[0])
        maximum_eigenvalue = float(eigenvalues[-1])
        if minimum_eigenvalue <= self.operator_stability_tolerance:
            raise DeePHFCapabilityError(
                "the coupled UHF occupied-virtual response operator is unstable or "
                f"singular: minimum eigenvalue {minimum_eigenvalue:.3e} <= "
                f"{self.operator_stability_tolerance:.3e}"
            )
        condition_number = maximum_eigenvalue / minimum_eigenvalue
        if (
            not np.isfinite(condition_number)
            or condition_number > self.operator_condition_tolerance
        ):
            raise DeePHFCapabilityError(
                "the coupled UHF occupied-virtual response operator is ill conditioned: "
                f"{condition_number:.3e} > {self.operator_condition_tolerance:.3e}"
            )
        return (
            dimension,
            alpha_dimension,
            beta_dimension,
            minimum_eigenvalue,
            maximum_eigenvalue,
            float(condition_number),
            symmetry_residual,
        )

    def _orbital_residual(
        self,
        responses,
        hamiltonian_mo,
        overlap_mo,
        coefficient,
        energy,
        occupied,
        virtual,
    ) -> tuple[np.ndarray, np.ndarray]:
        induced = self._induced_mo_potential(
            responses[0],
            responses[1],
            coefficient,
            occupied,
        )
        residuals = []
        for spin_index in range(2):
            spin_residual = (
                hamiltonian_mo[spin_index]
                + induced[spin_index]
                - overlap_mo[spin_index] * energy[spin_index, occupied[spin_index]]
                + (
                    energy[spin_index, :, None]
                    - energy[spin_index, occupied[spin_index]]
                )
                * responses[spin_index]
            )
            residuals.append(spin_residual[..., virtual[spin_index], :])
        return tuple(residuals)

    def _solve_orbitals(
        self,
        hamiltonian_mo,
        overlap_mo,
        coefficient,
        energy,
        occupation,
        occupied,
        virtual,
    ):
        perturbation_shape = hamiltonian_mo[0].shape[:-2]
        nset = int(np.prod(perturbation_shape))
        nmo = coefficient.shape[2]
        alpha_nocc, beta_nocc, alpha_nvir, beta_nvir, _, _ = self._dimensions(
            occupied,
            virtual,
        )
        flat_hamiltonian = (
            hamiltonian_mo[0].reshape(nset, nmo, alpha_nocc),
            hamiltonian_mo[1].reshape(nset, nmo, beta_nocc),
        )
        flat_overlap = (
            overlap_mo[0].reshape(nset, nmo, alpha_nocc),
            overlap_mo[1].reshape(nset, nmo, beta_nocc),
        )

        def induced_full(response):
            response = np.asarray(response).reshape(
                -1,
                nmo * alpha_nocc + nmo * beta_nocc,
            )
            alpha = response[:, : nmo * alpha_nocc].reshape(-1, nmo, alpha_nocc)
            beta = response[:, nmo * alpha_nocc :].reshape(-1, nmo, beta_nocc)
            induced = self._induced_mo_potential(
                alpha,
                beta,
                coefficient,
                occupied,
            )
            return np.concatenate(
                (induced[0].reshape(len(response), -1), induced[1].reshape(len(response), -1)),
                axis=1,
            )

        try:
            response, _ = ucphf.solve(
                induced_full,
                energy,
                occupation,
                flat_hamiltonian,
                flat_overlap,
                max_cycle=self.max_cycle,
                tol=self.cphf_tolerance,
                level_shift=self.level_shift,
                verbose=self.reference.verbose,
            )
        except Exception as error:
            raise UHFResponseError(
                f"PySCF UHF coupled CPHF solve failed: {error}"
            ) from error
        alpha_response = _validated_float64_array(
            response[0],
            flat_hamiltonian[0].shape,
            "PySCF alpha UHF CPHF response",
        ).reshape(*perturbation_shape, nmo, alpha_nocc)
        beta_response = _validated_float64_array(
            response[1],
            flat_hamiltonian[1].shape,
            "PySCF beta UHF CPHF response",
        ).reshape(*perturbation_shape, nmo, beta_nocc)
        residual = self._orbital_residual(
            (alpha_response, beta_response),
            hamiltonian_mo,
            overlap_mo,
            coefficient,
            energy,
            occupied,
            virtual,
        )

        def maximum_residual(values):
            return max(float(np.max(np.abs(value), initial=0.0)) for value in values)

        residual_history = [maximum_residual(residual)]
        while (
            residual_history[-1] > self.residual_tolerance
            and len(residual_history) - 1 < self.max_refinement_cycles
        ):
            flat_residual = (
                residual[0].reshape(nset, alpha_nvir, alpha_nocc),
                residual[1].reshape(nset, beta_nvir, beta_nocc),
            )
            combined = np.concatenate(
                (
                    flat_residual[0].reshape(nset, -1),
                    flat_residual[1].reshape(nset, -1),
                ),
                axis=1,
            )
            scales = np.linalg.norm(combined, axis=1)
            active = scales > np.finfo(float).eps
            alpha_correction = np.zeros_like(flat_residual[0])
            beta_correction = np.zeros_like(flat_residual[1])

            def induced_virtual(response):
                response = np.asarray(response).reshape(
                    -1,
                    alpha_nvir * alpha_nocc + beta_nvir * beta_nocc,
                )
                alpha_virtual = response[:, : alpha_nvir * alpha_nocc].reshape(
                    -1,
                    alpha_nvir,
                    alpha_nocc,
                )
                beta_virtual = response[:, alpha_nvir * alpha_nocc :].reshape(
                    -1,
                    beta_nvir,
                    beta_nocc,
                )
                alpha_full = np.zeros((len(response), nmo, alpha_nocc))
                beta_full = np.zeros((len(response), nmo, beta_nocc))
                alpha_full[:, virtual[0]] = alpha_virtual
                beta_full[:, virtual[1]] = beta_virtual
                induced = self._induced_mo_potential(
                    alpha_full,
                    beta_full,
                    coefficient,
                    occupied,
                )
                return np.concatenate(
                    (
                        induced[0][:, virtual[0]].reshape(len(response), -1),
                        induced[1][:, virtual[1]].reshape(len(response), -1),
                    ),
                    axis=1,
                )

            if np.any(active):
                normalized_rhs = (
                    flat_residual[0][active] / scales[active, None, None],
                    flat_residual[1][active] / scales[active, None, None],
                )
                try:
                    normalized_correction, _ = ucphf.solve(
                        induced_virtual,
                        energy,
                        occupation,
                        normalized_rhs,
                        s1=None,
                        max_cycle=self.max_cycle,
                        tol=self.cphf_tolerance,
                        level_shift=self.level_shift,
                        verbose=self.reference.verbose,
                    )
                except Exception as error:
                    raise UHFResponseError(
                        f"PySCF UHF coupled CPHF residual refinement failed: {error}"
                    ) from error
                alpha_normalized = _validated_float64_array(
                    normalized_correction[0],
                    normalized_rhs[0].shape,
                    "PySCF alpha UHF CPHF refinement response",
                )
                beta_normalized = _validated_float64_array(
                    normalized_correction[1],
                    normalized_rhs[1].shape,
                    "PySCF beta UHF CPHF refinement response",
                )
                alpha_correction[active] = alpha_normalized * scales[active, None, None]
                beta_correction[active] = beta_normalized * scales[active, None, None]
                alpha_response[..., virtual[0], :] += alpha_correction.reshape(
                    *perturbation_shape,
                    alpha_nvir,
                    alpha_nocc,
                )
                beta_response[..., virtual[1], :] += beta_correction.reshape(
                    *perturbation_shape,
                    beta_nvir,
                    beta_nocc,
                )
            residual = self._orbital_residual(
                (alpha_response, beta_response),
                hamiltonian_mo,
                overlap_mo,
                coefficient,
                energy,
                occupied,
                virtual,
            )
            residual_history.append(maximum_residual(residual))
        return (
            (alpha_response, beta_response),
            residual,
            tuple(residual_history),
        )

    @staticmethod
    def _invariants(
        density_response,
        density_ground,
        overlap,
        overlap_derivative,
    ):
        idempotency = (
            np.einsum("...ij,jk,kl->...il", density_response, overlap, density_ground)
            + np.einsum("ij,...jk,kl->...il", density_ground, overlap_derivative, density_ground)
            + np.einsum("ij,jk,...kl->...il", density_ground, overlap, density_response)
            - density_response
        )
        particle_number = (
            np.einsum("...ij,ji->...", density_response, overlap)
            + np.einsum("ij,...ji->...", density_ground, overlap_derivative)
        )
        return (
            float(np.max(np.abs(idempotency), initial=0.0)),
            float(np.max(np.abs(particle_number), initial=0.0)),
        )

    @staticmethod
    def _density_reconstruction_residual(
        responses,
        response_parts,
        coefficient_responses,
        coefficient_parts,
        density_responses,
        density_parts,
        total_density,
        total_density_occupied_virtual,
        total_density_metric,
        coefficient,
        occupied,
    ) -> float:
        residuals = [
            total_density - density_responses[0] - density_responses[1],
            total_density_occupied_virtual - density_parts[0][0] - density_parts[1][0],
            total_density_metric - density_parts[0][1] - density_parts[1][1],
            total_density - total_density_occupied_virtual - total_density_metric,
        ]
        for spin_index in range(2):
            occupied_coefficient = coefficient[spin_index][:, occupied[spin_index]]
            one_sided = np.einsum(
                "...pi,qi->...pq",
                coefficient_responses[spin_index],
                occupied_coefficient,
            )
            density_from_coefficient = one_sided + one_sided.swapaxes(-1, -2)
            residuals.extend(
                (
                    responses[spin_index]
                    - response_parts[spin_index][0]
                    - response_parts[spin_index][1],
                    coefficient_responses[spin_index]
                    - coefficient_parts[spin_index][0]
                    - coefficient_parts[spin_index][1],
                    density_responses[spin_index]
                    - density_parts[spin_index][0]
                    - density_parts[spin_index][1],
                    density_responses[spin_index] - density_from_coefficient,
                )
            )
        return max(
            float(np.max(np.abs(value), initial=0.0)) for value in residuals
        )

    def solve(self) -> UHFResponse:
        """Return the audited complete spin-resolved AO density response."""
        validate_uhf_reference(self.reference)
        coefficient, energy, occupation, occupied, virtual, minimum_gaps = self._state()
        (
            response_dimension,
            alpha_dimension,
            beta_dimension,
            operator_minimum_eigenvalue,
            operator_maximum_eigenvalue,
            operator_condition_number,
            operator_symmetry_residual,
        ) = self._response_operator_diagnostics(
            coefficient,
            energy,
            occupied,
            virtual,
        )
        overlap = np.asarray(self.reference.get_ovlp())
        overlap_derivative = self._overlap_derivative()
        hamiltonian_derivative = self._hamiltonian_derivative(coefficient, occupation)
        hamiltonian_mo = tuple(
            np.einsum(
                "mp,...mn,ni->...pi",
                coefficient[spin_index],
                hamiltonian_derivative[spin_index],
                coefficient[spin_index][:, occupied[spin_index]],
            )
            for spin_index in range(2)
        )
        overlap_mo = tuple(
            np.einsum(
                "mp,...mn,ni->...pi",
                coefficient[spin_index],
                overlap_derivative,
                coefficient[spin_index][:, occupied[spin_index]],
            )
            for spin_index in range(2)
        )
        responses, residuals, residual_history = self._solve_orbitals(
            hamiltonian_mo,
            overlap_mo,
            coefficient,
            energy,
            occupation,
            occupied,
            virtual,
        )
        metric_responses = []
        occupied_virtual_responses = []
        coefficient_responses = []
        coefficient_metric = []
        coefficient_occupied_virtual = []
        density_responses = []
        density_metric = []
        density_occupied_virtual = []
        metric_residuals = []
        for spin_index in range(2):
            metric_response = np.zeros_like(responses[spin_index])
            metric_response[..., occupied[spin_index], :] = responses[spin_index][
                ..., occupied[spin_index], :
            ]
            occupied_virtual_response = np.zeros_like(responses[spin_index])
            occupied_virtual_response[..., virtual[spin_index], :] = responses[
                spin_index
            ][..., virtual[spin_index], :]
            metric_responses.append(metric_response)
            occupied_virtual_responses.append(occupied_virtual_response)
            coefficient_responses.append(
                np.einsum("mp,...pi->...mi", coefficient[spin_index], responses[spin_index])
            )
            coefficient_metric.append(
                np.einsum("mp,...pi->...mi", coefficient[spin_index], metric_response)
            )
            coefficient_occupied_virtual.append(
                np.einsum(
                    "mp,...pi->...mi",
                    coefficient[spin_index],
                    occupied_virtual_response,
                )
            )
            density_responses.append(
                self._density_from_mo_response(
                    responses[spin_index],
                    coefficient[spin_index],
                    occupied[spin_index],
                )
            )
            density_metric.append(
                self._density_from_mo_response(
                    metric_response,
                    coefficient[spin_index],
                    occupied[spin_index],
                )
            )
            density_occupied_virtual.append(
                self._density_from_mo_response(
                    occupied_virtual_response,
                    coefficient[spin_index],
                    occupied[spin_index],
                )
            )
            overlap_occupied = overlap_mo[spin_index][
                ..., occupied[spin_index], :
            ]
            metric_residuals.append(
                float(
                    np.max(
                        np.abs(
                            responses[spin_index][..., occupied[spin_index], :]
                            + responses[spin_index][..., occupied[spin_index], :].swapaxes(-1, -2)
                            + overlap_occupied
                        ),
                        initial=0.0,
                    )
                )
            )
        total_density = density_responses[0] + density_responses[1]
        total_density_metric = density_metric[0] + density_metric[1]
        total_density_occupied_virtual = (
            density_occupied_virtual[0] + density_occupied_virtual[1]
        )
        reconstruction_residual = self._density_reconstruction_residual(
            responses,
            tuple(zip(occupied_virtual_responses, metric_responses, strict=True)),
            coefficient_responses,
            tuple(zip(coefficient_occupied_virtual, coefficient_metric, strict=True)),
            density_responses,
            tuple(zip(density_occupied_virtual, density_metric, strict=True)),
            total_density,
            total_density_occupied_virtual,
            total_density_metric,
            coefficient,
            occupied,
        )
        alpha_translation_residual = float(
            np.max(np.abs(np.sum(density_responses[0], axis=0)), initial=0.0)
        )
        beta_translation_residual = float(
            np.max(np.abs(np.sum(density_responses[1], axis=0)), initial=0.0)
        )
        translation_residual = float(
            np.max(np.abs(np.sum(total_density, axis=0)), initial=0.0)
        )
        density_ground = np.asarray(self.reference.make_rdm1())
        invariants = [
            self._invariants(
                density_responses[spin_index],
                density_ground[spin_index],
                overlap,
                overlap_derivative,
            )
            for spin_index in range(2)
        ]
        alpha_maximum_residual = float(
            np.max(np.abs(residuals[0]), initial=0.0)
        )
        beta_maximum_residual = float(
            np.max(np.abs(residuals[1]), initial=0.0)
        )
        squared_residual_sum = sum(float(np.sum(np.square(value))) for value in residuals)
        residual_size = sum(value.size for value in residuals)
        diagnostics = UHFResponseDiagnostics(
            minimum_alpha_orbital_gap=minimum_gaps[0],
            minimum_beta_orbital_gap=minimum_gaps[1],
            pyscf_version=pyscf.__version__,
            cphf_tolerance=self.cphf_tolerance,
            maximum_residual=max(alpha_maximum_residual, beta_maximum_residual),
            alpha_maximum_residual=alpha_maximum_residual,
            beta_maximum_residual=beta_maximum_residual,
            residual_rms=float(np.sqrt(squared_residual_sum / residual_size)),
            residual_tolerance=self.residual_tolerance,
            invariant_tolerance=self.invariant_tolerance,
            orbital_gap_tolerance=self.orbital_gap_tolerance,
            max_cycle=self.max_cycle,
            max_refinement_cycles=self.max_refinement_cycles,
            level_shift=self.level_shift,
            response_dimension=response_dimension,
            alpha_response_dimension=alpha_dimension,
            beta_response_dimension=beta_dimension,
            operator_stability_tolerance=self.operator_stability_tolerance,
            operator_condition_tolerance=self.operator_condition_tolerance,
            operator_symmetry_tolerance=self.operator_symmetry_tolerance,
            operator_dimension_limit=self.operator_dimension_limit,
            operator_minimum_eigenvalue=operator_minimum_eigenvalue,
            operator_maximum_eigenvalue=operator_maximum_eigenvalue,
            operator_condition_number=operator_condition_number,
            operator_symmetry_residual=operator_symmetry_residual,
            alpha_metric_residual=metric_residuals[0],
            beta_metric_residual=metric_residuals[1],
            alpha_idempotency_residual=invariants[0][0],
            beta_idempotency_residual=invariants[1][0],
            alpha_particle_number_residual=invariants[0][1],
            beta_particle_number_residual=invariants[1][1],
            density_reconstruction_residual=reconstruction_residual,
            alpha_translation_residual=alpha_translation_residual,
            beta_translation_residual=beta_translation_residual,
            translation_residual=translation_residual,
            refinement_cycles=len(residual_history) - 1,
            residual_history=residual_history,
        )
        arrays = {
            "alpha MO response": responses[0],
            "beta MO response": responses[1],
            "alpha density response": density_responses[0],
            "beta density response": density_responses[1],
            "total density response": total_density,
            "overlap derivative": overlap_derivative,
            "alpha Hamiltonian derivative": hamiltonian_derivative[0],
            "beta Hamiltonian derivative": hamiltonian_derivative[1],
            "alpha residual": residuals[0],
            "beta residual": residuals[1],
        }
        nonfinite = [name for name, value in arrays.items() if not np.isfinite(value).all()]
        if nonfinite:
            raise UHFResponseError(
                "nonfinite UHF response quantities: " + ", ".join(nonfinite)
            )
        diagnostic_values = tuple(
            getattr(diagnostics, field.name)
            for field in fields(diagnostics)
            if field.name not in {"pyscf_version", "residual_history"}
        ) + diagnostics.residual_history
        if not np.isfinite(diagnostic_values).all():
            raise UHFResponseError("nonfinite UHF response diagnostics")
        if diagnostics.maximum_residual > self.residual_tolerance:
            history = " -> ".join(f"{value:.3e}" for value in residual_history)
            raise UHFResponseError(
                "UHF coupled response residual exceeds tolerance: "
                f"{diagnostics.maximum_residual:.3e} > {self.residual_tolerance:.3e}; "
                f"refinement history: {history}"
            )
        invariant_values = (
            *metric_residuals,
            invariants[0][0],
            invariants[1][0],
            invariants[0][1],
            invariants[1][1],
            reconstruction_residual,
            alpha_translation_residual,
            beta_translation_residual,
            translation_residual,
        )
        if max(invariant_values) > self.invariant_tolerance:
            raise UHFResponseError(
                "UHF response invariant exceeds tolerance "
                f"{self.invariant_tolerance:.3e}: maximum={max(invariant_values):.3e}"
            )
        response = UHFResponse(
            reference_identity=id(self.reference),
            state_fingerprint=uhf_reference_fingerprint(self.reference),
            integrity_fingerprint="",
            alpha_mo_response=_immutable_array(responses[0]),
            beta_mo_response=_immutable_array(responses[1]),
            alpha_mo_response_occupied_virtual=_immutable_array(occupied_virtual_responses[0]),
            beta_mo_response_occupied_virtual=_immutable_array(occupied_virtual_responses[1]),
            alpha_mo_response_metric=_immutable_array(metric_responses[0]),
            beta_mo_response_metric=_immutable_array(metric_responses[1]),
            alpha_coefficient_response=_immutable_array(coefficient_responses[0]),
            beta_coefficient_response=_immutable_array(coefficient_responses[1]),
            alpha_coefficient_response_occupied_virtual=_immutable_array(coefficient_occupied_virtual[0]),
            beta_coefficient_response_occupied_virtual=_immutable_array(coefficient_occupied_virtual[1]),
            alpha_coefficient_response_metric=_immutable_array(coefficient_metric[0]),
            beta_coefficient_response_metric=_immutable_array(coefficient_metric[1]),
            alpha_density_response=_immutable_array(density_responses[0]),
            beta_density_response=_immutable_array(density_responses[1]),
            total_density_response=_immutable_array(total_density),
            alpha_density_response_occupied_virtual=_immutable_array(density_occupied_virtual[0]),
            beta_density_response_occupied_virtual=_immutable_array(density_occupied_virtual[1]),
            total_density_response_occupied_virtual=_immutable_array(total_density_occupied_virtual),
            alpha_density_response_metric=_immutable_array(density_metric[0]),
            beta_density_response_metric=_immutable_array(density_metric[1]),
            total_density_response_metric=_immutable_array(total_density_metric),
            overlap_derivative=_immutable_array(overlap_derivative),
            alpha_hamiltonian_derivative=_immutable_array(hamiltonian_derivative[0]),
            beta_hamiltonian_derivative=_immutable_array(hamiltonian_derivative[1]),
            alpha_orbital_response_residual=_immutable_array(residuals[0]),
            beta_orbital_response_residual=_immutable_array(residuals[1]),
            diagnostics=diagnostics,
        )
        response = replace(
            response,
            integrity_fingerprint=uhf_response_integrity_fingerprint(response),
        )
        self.audit_response_equations(response)
        return response

    def _validate_supplied_structure(
        self,
        response: UHFResponse,
        occupied: np.ndarray,
        virtual: np.ndarray,
    ) -> None:
        coordinate_shape = (self.molecule.natm, 3)
        nao = self.molecule.nao
        alpha_nocc = int(np.count_nonzero(occupied[0]))
        beta_nocc = int(np.count_nonzero(occupied[1]))
        alpha_nvir = int(np.count_nonzero(virtual[0]))
        beta_nvir = int(np.count_nonzero(virtual[1]))
        alpha_response_shape = (*coordinate_shape, nao, alpha_nocc)
        beta_response_shape = (*coordinate_shape, nao, beta_nocc)
        density_shape = (*coordinate_shape, nao, nao)
        expected_shapes = {
            "alpha_mo_response": alpha_response_shape,
            "beta_mo_response": beta_response_shape,
            "alpha_mo_response_occupied_virtual": alpha_response_shape,
            "beta_mo_response_occupied_virtual": beta_response_shape,
            "alpha_mo_response_metric": alpha_response_shape,
            "beta_mo_response_metric": beta_response_shape,
            "alpha_coefficient_response": alpha_response_shape,
            "beta_coefficient_response": beta_response_shape,
            "alpha_coefficient_response_occupied_virtual": alpha_response_shape,
            "beta_coefficient_response_occupied_virtual": beta_response_shape,
            "alpha_coefficient_response_metric": alpha_response_shape,
            "beta_coefficient_response_metric": beta_response_shape,
            "alpha_density_response": density_shape,
            "beta_density_response": density_shape,
            "total_density_response": density_shape,
            "alpha_density_response_occupied_virtual": density_shape,
            "beta_density_response_occupied_virtual": density_shape,
            "total_density_response_occupied_virtual": density_shape,
            "alpha_density_response_metric": density_shape,
            "beta_density_response_metric": density_shape,
            "total_density_response_metric": density_shape,
            "overlap_derivative": density_shape,
            "alpha_hamiltonian_derivative": density_shape,
            "beta_hamiltonian_derivative": density_shape,
            "alpha_orbital_response_residual": (
                *coordinate_shape,
                alpha_nvir,
                alpha_nocc,
            ),
            "beta_orbital_response_residual": (
                *coordinate_shape,
                beta_nvir,
                beta_nocc,
            ),
        }
        for name, expected_shape in expected_shapes.items():
            _validated_response_array(
                getattr(response, name),
                expected_shape,
                name.replace("_", " "),
            )
        if type(response.reference_identity) is not int:
            raise UHFResponseError(
                "the supplied UHF response reference identity must be an integer"
            )
        for name in ("state_fingerprint", "integrity_fingerprint"):
            value = getattr(response, name)
            if type(value) is not str or not value:
                raise UHFResponseError(
                    f"the supplied UHF response {name.replace('_', ' ')} is invalid"
                )
        diagnostics = response.diagnostics
        if type(diagnostics.pyscf_version) is not str:
            raise UHFResponseError(
                "the supplied UHF response PySCF version is invalid"
            )
        integer_fields = {
            "max_cycle",
            "max_refinement_cycles",
            "response_dimension",
            "alpha_response_dimension",
            "beta_response_dimension",
            "operator_dimension_limit",
            "refinement_cycles",
        }
        for diagnostic_field in fields(diagnostics):
            name = diagnostic_field.name
            if name in {"pyscf_version", "residual_history"}:
                continue
            value = getattr(diagnostics, name)
            if name in integer_fields:
                if type(value) is not int:
                    raise UHFResponseError(
                        f"the supplied UHF response {name.replace('_', ' ')} must be an integer"
                    )
            elif (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, Real)
                or not np.isfinite(float(value))
            ):
                raise UHFResponseError(
                    f"the supplied UHF response {name.replace('_', ' ')} must be finite and real"
                )
        history = diagnostics.residual_history
        if type(history) is not tuple or not history:
            raise UHFResponseError(
                "the supplied UHF response residual history must be a nonempty tuple"
            )
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, Real)
            or not np.isfinite(float(value))
            or float(value) < 0
            for value in history
        ):
            raise UHFResponseError(
                "the supplied UHF response residual history is invalid"
            )
        expected_controls = {
            "cphf_tolerance": self.cphf_tolerance,
            "residual_tolerance": self.residual_tolerance,
            "invariant_tolerance": self.invariant_tolerance,
            "orbital_gap_tolerance": self.orbital_gap_tolerance,
            "max_cycle": self.max_cycle,
            "max_refinement_cycles": self.max_refinement_cycles,
            "level_shift": self.level_shift,
            "operator_stability_tolerance": self.operator_stability_tolerance,
            "operator_condition_tolerance": self.operator_condition_tolerance,
            "operator_symmetry_tolerance": self.operator_symmetry_tolerance,
            "operator_dimension_limit": self.operator_dimension_limit,
        }
        for name, expected in expected_controls.items():
            if getattr(diagnostics, name) != expected:
                raise UHFResponseError(
                    f"the supplied UHF response {name.replace('_', ' ')} does not match the adapter"
                )
        if diagnostics.refinement_cycles != len(history) - 1:
            raise UHFResponseError(
                "the supplied UHF response refinement history is inconsistent"
            )
        if not 0 <= diagnostics.refinement_cycles <= self.max_refinement_cycles:
            raise UHFResponseError(
                "the supplied UHF response refinement cycle count is invalid"
            )
        if diagnostics.response_dimension != (
            diagnostics.alpha_response_dimension
            + diagnostics.beta_response_dimension
        ):
            raise UHFResponseError(
                "the supplied UHF response dimensions are inconsistent"
            )
        expected_dimensions = (
            alpha_nocc * alpha_nvir,
            beta_nocc * beta_nvir,
        )
        if (
            diagnostics.alpha_response_dimension,
            diagnostics.beta_response_dimension,
        ) != expected_dimensions:
            raise UHFResponseError(
                "the supplied UHF response spin dimensions are inconsistent"
            )
        nonnegative_fields = {
            "maximum_residual",
            "alpha_maximum_residual",
            "beta_maximum_residual",
            "residual_rms",
            "operator_symmetry_residual",
            "alpha_metric_residual",
            "beta_metric_residual",
            "alpha_idempotency_residual",
            "beta_idempotency_residual",
            "alpha_particle_number_residual",
            "beta_particle_number_residual",
            "density_reconstruction_residual",
            "alpha_translation_residual",
            "beta_translation_residual",
            "translation_residual",
        }
        if any(float(getattr(diagnostics, name)) < 0 for name in nonnegative_fields):
            raise UHFResponseError(
                "the supplied UHF response contains a negative residual diagnostic"
            )
        if diagnostics.minimum_alpha_orbital_gap <= self.orbital_gap_tolerance:
            raise UHFResponseError(
                "the supplied UHF response alpha gap is outside the adapter domain"
            )
        if diagnostics.minimum_beta_orbital_gap <= self.orbital_gap_tolerance:
            raise UHFResponseError(
                "the supplied UHF response beta gap is outside the adapter domain"
            )
        if diagnostics.operator_minimum_eigenvalue <= self.operator_stability_tolerance:
            raise UHFResponseError(
                "the supplied UHF response operator is outside the stability domain"
            )
        if diagnostics.operator_maximum_eigenvalue < diagnostics.operator_minimum_eigenvalue:
            raise UHFResponseError(
                "the supplied UHF response operator spectral bounds are invalid"
            )
        if not 1 <= diagnostics.operator_condition_number <= self.operator_condition_tolerance:
            raise UHFResponseError(
                "the supplied UHF response operator condition number is invalid"
            )
        if diagnostics.operator_symmetry_residual > self.operator_symmetry_tolerance:
            raise UHFResponseError(
                "the supplied UHF response operator symmetry residual is excessive"
            )
        expected_maximum = max(
            diagnostics.alpha_maximum_residual,
            diagnostics.beta_maximum_residual,
        )
        if not np.isclose(
            diagnostics.maximum_residual,
            expected_maximum,
            rtol=1.0e-12,
            atol=1.0e-15,
        ):
            raise UHFResponseError(
                "the supplied UHF response spin residual diagnostics are inconsistent"
            )
        if not np.isclose(
            history[-1],
            diagnostics.maximum_residual,
            rtol=1.0e-12,
            atol=1.0e-15,
        ):
            raise UHFResponseError(
                "the supplied UHF response final residual history is inconsistent"
            )

    def audit_response_equations(self, response: UHFResponse) -> None:
        """Rebuild coupled equations and invariants for a supplied response."""
        validate_uhf_reference(self.reference)
        if type(response) is not UHFResponse:
            raise UHFResponseError("the supplied UHF response has an invalid type")
        if type(response.diagnostics) is not UHFResponseDiagnostics:
            raise UHFResponseError(
                "the supplied UHF response diagnostics have an invalid type"
            )
        coefficient, energy, occupation, occupied, virtual, minimum_gaps = self._state()
        self._validate_supplied_structure(response, occupied, virtual)
        if response.reference_identity != id(self.reference):
            raise UHFResponseError("the supplied UHF response belongs to another reference")
        if response.state_fingerprint != uhf_reference_fingerprint(self.reference):
            raise UHFResponseError("the supplied UHF response does not match the current state")
        if response.integrity_fingerprint != uhf_response_integrity_fingerprint(response):
            raise UHFResponseError("the supplied UHF response failed its integrity check")
        if response.diagnostics.pyscf_version != pyscf.__version__:
            raise UHFResponseError(
                "the supplied UHF response PySCF version does not match the runtime"
            )
        operator_diagnostics = self._response_operator_diagnostics(
            coefficient,
            energy,
            occupied,
            virtual,
        )
        overlap = np.asarray(self.reference.get_ovlp())
        overlap_derivative = self._overlap_derivative()
        hamiltonian_derivative = self._hamiltonian_derivative(coefficient, occupation)
        expected_derivatives = (
            (response.overlap_derivative, overlap_derivative, "overlap derivative"),
            (
                response.alpha_hamiltonian_derivative,
                hamiltonian_derivative[0],
                "alpha Hamiltonian derivative",
            ),
            (
                response.beta_hamiltonian_derivative,
                hamiltonian_derivative[1],
                "beta Hamiltonian derivative",
            ),
        )
        for stored, expected, name in expected_derivatives:
            if not np.allclose(stored, expected, rtol=0.0, atol=1.0e-12):
                raise UHFResponseError(
                    f"the supplied UHF response {name} does not match the reference"
                )
        responses = (response.alpha_mo_response, response.beta_mo_response)
        response_parts = (
            (
                response.alpha_mo_response_occupied_virtual,
                response.alpha_mo_response_metric,
            ),
            (
                response.beta_mo_response_occupied_virtual,
                response.beta_mo_response_metric,
            ),
        )
        density_responses = (
            response.alpha_density_response,
            response.beta_density_response,
        )
        density_parts = (
            (
                response.alpha_density_response_occupied_virtual,
                response.alpha_density_response_metric,
            ),
            (
                response.beta_density_response_occupied_virtual,
                response.beta_density_response_metric,
            ),
        )
        coefficient_responses = (
            response.alpha_coefficient_response,
            response.beta_coefficient_response,
        )
        coefficient_parts = (
            (
                response.alpha_coefficient_response_occupied_virtual,
                response.alpha_coefficient_response_metric,
            ),
            (
                response.beta_coefficient_response_occupied_virtual,
                response.beta_coefficient_response_metric,
            ),
        )
        metric_residuals = []
        invariant_values = []
        ground_density = np.asarray(self.reference.make_rdm1())
        for spin_index, spin_name in enumerate(("alpha", "beta")):
            expected_mo_occupied_virtual = np.zeros_like(responses[spin_index])
            expected_mo_occupied_virtual[..., virtual[spin_index], :] = responses[
                spin_index
            ][..., virtual[spin_index], :]
            expected_mo_metric = np.zeros_like(responses[spin_index])
            expected_mo_metric[..., occupied[spin_index], :] = responses[spin_index][
                ..., occupied[spin_index], :
            ]
            expected_coefficient = np.einsum(
                "mp,...pi->...mi",
                coefficient[spin_index],
                responses[spin_index],
            )
            expected_coefficient_occupied_virtual = np.einsum(
                "mp,...pi->...mi",
                coefficient[spin_index],
                expected_mo_occupied_virtual,
            )
            expected_coefficient_metric = np.einsum(
                "mp,...pi->...mi",
                coefficient[spin_index],
                expected_mo_metric,
            )
            expected_density = self._density_from_mo_response(
                responses[spin_index],
                coefficient[spin_index],
                occupied[spin_index],
            )
            expected_density_occupied_virtual = self._density_from_mo_response(
                expected_mo_occupied_virtual,
                coefficient[spin_index],
                occupied[spin_index],
            )
            expected_density_metric = self._density_from_mo_response(
                expected_mo_metric,
                coefficient[spin_index],
                occupied[spin_index],
            )
            comparisons = (
                (response_parts[spin_index][0], expected_mo_occupied_virtual, "MO OV"),
                (response_parts[spin_index][1], expected_mo_metric, "MO metric"),
                (coefficient_responses[spin_index], expected_coefficient, "coefficient"),
                (
                    coefficient_parts[spin_index][0],
                    expected_coefficient_occupied_virtual,
                    "coefficient OV",
                ),
                (
                    coefficient_parts[spin_index][1],
                    expected_coefficient_metric,
                    "coefficient metric",
                ),
                (density_responses[spin_index], expected_density, "density"),
                (
                    density_parts[spin_index][0],
                    expected_density_occupied_virtual,
                    "density OV",
                ),
                (
                    density_parts[spin_index][1],
                    expected_density_metric,
                    "density metric",
                ),
            )
            for stored, expected, name in comparisons:
                if not np.allclose(stored, expected, rtol=0.0, atol=1.0e-12):
                    raise UHFResponseError(
                        f"the supplied UHF {spin_name} {name} response is inconsistent"
                    )
            overlap_mo = np.einsum(
                "mp,...mn,ni->...pi",
                coefficient[spin_index],
                overlap_derivative,
                coefficient[spin_index][:, occupied[spin_index]],
            )
            overlap_occupied = overlap_mo[..., occupied[spin_index], :]
            metric_residuals.append(
                float(
                    np.max(
                        np.abs(
                            responses[spin_index][..., occupied[spin_index], :]
                            + responses[spin_index][..., occupied[spin_index], :].swapaxes(-1, -2)
                            + overlap_occupied
                        ),
                        initial=0.0,
                    )
                )
            )
            invariant_values.append(
                self._invariants(
                    density_responses[spin_index],
                    ground_density[spin_index],
                    overlap,
                    overlap_derivative,
                )
            )
        expected_total = density_responses[0] + density_responses[1]
        expected_total_occupied_virtual = density_parts[0][0] + density_parts[1][0]
        expected_total_metric = density_parts[0][1] + density_parts[1][1]
        total_comparisons = (
            (response.total_density_response, expected_total, "total density"),
            (
                response.total_density_response_occupied_virtual,
                expected_total_occupied_virtual,
                "total density OV",
            ),
            (
                response.total_density_response_metric,
                expected_total_metric,
                "total density metric",
            ),
        )
        for stored, expected, name in total_comparisons:
            if not np.allclose(stored, expected, rtol=0.0, atol=1.0e-12):
                raise UHFResponseError(
                    f"the supplied UHF response {name} is inconsistent"
                )
        hamiltonian_mo = tuple(
            np.einsum(
                "mp,...mn,ni->...pi",
                coefficient[spin_index],
                hamiltonian_derivative[spin_index],
                coefficient[spin_index][:, occupied[spin_index]],
            )
            for spin_index in range(2)
        )
        overlap_mo = tuple(
            np.einsum(
                "mp,...mn,ni->...pi",
                coefficient[spin_index],
                overlap_derivative,
                coefficient[spin_index][:, occupied[spin_index]],
            )
            for spin_index in range(2)
        )
        residuals = self._orbital_residual(
            responses,
            hamiltonian_mo,
            overlap_mo,
            coefficient,
            energy,
            occupied,
            virtual,
        )
        stored_residuals = (
            response.alpha_orbital_response_residual,
            response.beta_orbital_response_residual,
        )
        for stored, expected, spin_name in zip(
            stored_residuals,
            residuals,
            ("alpha", "beta"),
            strict=True,
        ):
            if not np.allclose(stored, expected, rtol=0.0, atol=1.0e-12):
                raise UHFResponseError(
                    f"the supplied UHF {spin_name} residual is not reproducible"
                )
        alpha_maximum = float(np.max(np.abs(residuals[0]), initial=0.0))
        beta_maximum = float(np.max(np.abs(residuals[1]), initial=0.0))
        squared_sum = sum(float(np.sum(np.square(value))) for value in residuals)
        residual_size = sum(value.size for value in residuals)
        reconstruction = self._density_reconstruction_residual(
            responses,
            response_parts,
            coefficient_responses,
            coefficient_parts,
            density_responses,
            density_parts,
            response.total_density_response,
            response.total_density_response_occupied_virtual,
            response.total_density_response_metric,
            coefficient,
            occupied,
        )
        alpha_translation = float(
            np.max(np.abs(np.sum(density_responses[0], axis=0)), initial=0.0)
        )
        beta_translation = float(
            np.max(np.abs(np.sum(density_responses[1], axis=0)), initial=0.0)
        )
        translation = float(np.max(np.abs(np.sum(expected_total, axis=0)), initial=0.0))
        measured = {
            "minimum_alpha_orbital_gap": minimum_gaps[0],
            "minimum_beta_orbital_gap": minimum_gaps[1],
            "response_dimension": operator_diagnostics[0],
            "alpha_response_dimension": operator_diagnostics[1],
            "beta_response_dimension": operator_diagnostics[2],
            "operator_minimum_eigenvalue": operator_diagnostics[3],
            "operator_maximum_eigenvalue": operator_diagnostics[4],
            "operator_condition_number": operator_diagnostics[5],
            "operator_symmetry_residual": operator_diagnostics[6],
            "maximum_residual": max(alpha_maximum, beta_maximum),
            "alpha_maximum_residual": alpha_maximum,
            "beta_maximum_residual": beta_maximum,
            "residual_rms": float(np.sqrt(squared_sum / residual_size)),
            "alpha_metric_residual": metric_residuals[0],
            "beta_metric_residual": metric_residuals[1],
            "alpha_idempotency_residual": invariant_values[0][0],
            "beta_idempotency_residual": invariant_values[1][0],
            "alpha_particle_number_residual": invariant_values[0][1],
            "beta_particle_number_residual": invariant_values[1][1],
            "density_reconstruction_residual": reconstruction,
            "alpha_translation_residual": alpha_translation,
            "beta_translation_residual": beta_translation,
            "translation_residual": translation,
        }
        for name, value in measured.items():
            recorded = getattr(response.diagnostics, name)
            if isinstance(value, int):
                consistent = recorded == value
            else:
                consistent = np.isclose(recorded, value, rtol=1.0e-10, atol=1.0e-12)
            if not consistent:
                raise UHFResponseError(
                    f"the supplied UHF response {name} diagnostic is inconsistent"
                )
        if measured["maximum_residual"] > self.residual_tolerance:
            raise UHFResponseError(
                "the supplied UHF response residual exceeds its tolerance"
            )
        invariant_maximum = max(
            measured[name]
            for name in (
                "alpha_metric_residual",
                "beta_metric_residual",
                "alpha_idempotency_residual",
                "beta_idempotency_residual",
                "alpha_particle_number_residual",
                "beta_particle_number_residual",
                "density_reconstruction_residual",
                "alpha_translation_residual",
                "beta_translation_residual",
                "translation_residual",
            )
        )
        if invariant_maximum > self.invariant_tolerance:
            raise UHFResponseError(
                "the supplied UHF response invariant exceeds its tolerance"
            )
