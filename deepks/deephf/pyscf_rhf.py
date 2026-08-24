"""Isolated PySCF 2.14 adapter for molecular RHF nuclear response."""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field, fields, replace
import hashlib
from numbers import Real
import operator
from types import MappingProxyType
from typing import Any
import weakref

import numpy as np
import pyscf
from pyscf import gto
from pyscf.gto import mole as gto_mole
from pyscf.hessian import rhf as rhf_hessian
from pyscf.scf import cphf, hf as scf_hf

from deepks.descriptor import is_ghost_atom

from .adjoint import (
    AdjointError,
    scalar_operator_fingerprint,
    solve_scalar_adjoint,
)
from .capabilities import (
    DeePHFCapabilityError,
    reference_is_transaction_validated,
    transaction_reference_fingerprint,
)


SUPPORTED_PYSCF_SERIES = (2, 14)


class RHFResponseError(RuntimeError):
    """Raised when the RHF response equations fail the strict contract."""


class RHFAdjointError(AdjointError):
    """Raised when the RHF scalar adjoint fails the strict contract."""


class RHFScannerReferenceError(RuntimeError):
    """Raised when a fresh scanner reference violates its strict contract."""


def validate_reference(reference):
    """Validate the molecular real-orbital integer-occupation RHF contract."""
    if type(reference) is not scf_hf.RHF:
        raise DeePHFCapabilityError(
            "DeePHF requires an undecorated native pyscf.scf.hf.RHF reference"
        )
    if reference_is_transaction_validated(reference):
        return reference
    if not reference.converged:
        raise DeePHFCapabilityError("the RHF reference must be converged")
    mol = reference.mol
    if type(mol) is not gto_mole.Mole:
        raise DeePHFCapabilityError(
            "the RHF reference must use a native molecular pyscf.gto.Mole"
        )
    if mol.spin != 0:
        raise DeePHFCapabilityError("the RHF reference must have spin zero")
    if mol.symmetry:
        raise DeePHFCapabilityError(
            "the RHF reference must not use symmetry-constrained occupations"
        )
    if mol.cart:
        raise DeePHFCapabilityError(
            "the initial DeePHF contract requires spherical AO functions"
        )
    if getattr(mol, "_ecp", None) or mol.has_ecp():
        raise DeePHFCapabilityError(
            "the initial DeePHF contract requires an all-electron reference"
        )
    if getattr(mol, "_pseudo", None):
        raise DeePHFCapabilityError(
            "the initial DeePHF contract does not support pseudopotentials"
        )
    if float(getattr(mol, "omega", 0.0)) != 0.0:
        raise DeePHFCapabilityError(
            "the initial DeePHF contract requires the full Coulomb interaction"
        )
    if getattr(mol, "nucmod", None):
        raise DeePHFCapabilityError(
            "the initial DeePHF contract requires point nuclei"
        )
    ghost_indices = [
        atom_index
        for atom_index in range(mol.natm)
        if is_ghost_atom(mol, atom_index)
    ]
    if ghost_indices:
        raise DeePHFCapabilityError(
            f"the initial DeePHF contract requires real atoms; ghost indices: {ghost_indices}"
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
            "the RHF reference has unsupported decorations: "
            + ", ".join(active_decorations)
        )
    custom_hooks = sorted(
        name
        for name, value in reference.__dict__.items()
        if name != "mol" and callable(value)
    )
    if custom_hooks:
        raise DeePHFCapabilityError(
            "the RHF reference has unsupported instance hooks: "
            + ", ".join(custom_hooks)
        )
    molecule_hooks = sorted(
        name for name, value in mol.__dict__.items() if callable(value)
    )
    if molecule_hooks:
        raise DeePHFCapabilityError(
            "the RHF molecule has unsupported instance hooks: "
            + ", ".join(molecule_hooks)
        )
    if reference.mo_coeff is None or reference.mo_energy is None:
        raise DeePHFCapabilityError("the RHF reference orbital state is incomplete")
    if reference.mo_occ is None:
        raise DeePHFCapabilityError("the RHF reference occupations are missing")
    mo_coeff = np.asarray(reference.mo_coeff)
    mo_energy = np.asarray(reference.mo_energy)
    occupations = np.asarray(reference.mo_occ)
    if any(
        np.iscomplexobj(value)
        for value in (mo_coeff, mo_energy, occupations)
    ):
        raise DeePHFCapabilityError("the RHF orbitals must be real")
    if any(
        value.dtype != np.dtype(np.float64)
        for value in (mo_coeff, mo_energy, occupations)
    ):
        raise DeePHFCapabilityError(
            "the RHF orbital state must use numpy.float64"
        )
    if not all(
        np.isfinite(value).all()
        for value in (mo_coeff, mo_energy, occupations)
    ):
        raise DeePHFCapabilityError("the RHF orbital state must be finite")
    if mo_coeff.shape != (mol.nao, mol.nao):
        raise DeePHFCapabilityError(
            "the RHF response requires a complete square MO coefficient matrix"
        )
    if mo_energy.shape != (mo_coeff.shape[1],):
        raise DeePHFCapabilityError("the RHF orbital energy shape is invalid")
    if occupations.shape != mo_energy.shape:
        raise DeePHFCapabilityError("the RHF occupation shape is invalid")
    if not np.all(np.isin(occupations, (0.0, 2.0))):
        raise DeePHFCapabilityError(
            "the RHF occupations must be integer closed-shell occupations"
        )
    if not np.isclose(occupations.sum(), mol.nelectron, rtol=0.0, atol=1.0e-12):
        raise DeePHFCapabilityError(
            "the RHF occupations do not match the molecular electron count"
        )
    occupied_count = mol.nelectron // 2
    expected_occupations = np.zeros_like(occupations)
    expected_occupations[:occupied_count] = 2.0
    if not np.array_equal(occupations, expected_occupations):
        raise DeePHFCapabilityError(
            "the initial RHF force contract requires the Aufbau ground-state root"
        )
    if not np.isfinite(reference.e_tot):
        raise DeePHFCapabilityError("the RHF reference energy must be finite")
    try:
        overlap = np.asarray(reference.get_ovlp())
        hcore = np.asarray(reference.get_hcore())
        density = np.asarray(reference.make_rdm1())
        effective_potential = np.asarray(reference.get_veff(mol, density))
        direct_coulomb, direct_exchange = scf_hf.get_jk(
            mol,
            density,
            hermi=1,
        )
        direct_effective_potential = np.asarray(
            direct_coulomb - 0.5 * direct_exchange
        )
    except Exception as error:
        raise DeePHFCapabilityError(
            f"the RHF reference matrices could not be evaluated: {error}"
        ) from error
    if any(
        np.iscomplexobj(value)
        for value in (
            overlap,
            hcore,
            density,
            effective_potential,
            direct_effective_potential,
        )
    ):
        raise DeePHFCapabilityError("the RHF AO matrices must be real")
    if any(
        value.dtype != np.dtype(np.float64)
        for value in (
            overlap,
            hcore,
            density,
            effective_potential,
            direct_effective_potential,
        )
    ):
        raise DeePHFCapabilityError("the RHF AO matrices must use numpy.float64")
    if not all(
        np.isfinite(value).all()
        for value in (
            overlap,
            hcore,
            density,
            effective_potential,
            direct_effective_potential,
        )
    ):
        raise DeePHFCapabilityError("the RHF AO matrices must be finite")
    expected_ao_shape = (mol.nao, mol.nao)
    if any(
        value.shape != expected_ao_shape
        for value in (
            overlap,
            hcore,
            density,
            effective_potential,
            direct_effective_potential,
        )
    ):
        raise DeePHFCapabilityError("the RHF AO matrix shape is invalid")
    interaction_error = np.max(
        np.abs(effective_potential - direct_effective_potential),
        initial=0.0,
    )
    if interaction_error > 1.0e-10:
        raise DeePHFCapabilityError(
            "the RHF two-electron interaction does not match the native "
            f"molecular integrals: residual {interaction_error:.3e}"
        )
    overlap_eigenvalues = np.linalg.eigvalsh(overlap)
    if overlap_eigenvalues[0] <= 1.0e-10:
        raise DeePHFCapabilityError(
            "the RHF AO overlap is singular or ill conditioned"
        )
    orthonormality_error = np.max(
        np.abs(mo_coeff.T @ overlap @ mo_coeff - np.eye(mo_coeff.shape[1]))
    )
    if orthonormality_error > 1.0e-8:
        raise DeePHFCapabilityError(
            "the RHF orbitals violate AO-metric orthonormality: "
            f"{orthonormality_error:.3e}"
        )
    electron_count = np.einsum("ij,ji->", density, overlap)
    if not np.isclose(electron_count, mol.nelectron, rtol=0.0, atol=1.0e-8):
        raise DeePHFCapabilityError(
            "the RHF AO density has an inconsistent electron count: "
            f"{electron_count:.12g}"
        )
    fock = hcore + direct_effective_potential
    canonical_residual = fock @ mo_coeff - overlap @ (
        mo_coeff * mo_energy
    )
    maximum_canonical_residual = np.max(
        np.abs(canonical_residual),
        initial=0.0,
    )
    if maximum_canonical_residual > 1.0e-7:
        raise DeePHFCapabilityError(
            "the stored RHF orbitals and energies do not satisfy the canonical "
            f"SCF equations: residual {maximum_canonical_residual:.3e}"
        )
    recomputed_energy = (
        0.5 * np.einsum("ij,ji->", density, hcore + fock)
        + mol.energy_nuc()
    )
    if not np.isclose(
        recomputed_energy,
        reference.e_tot,
        rtol=0.0,
        atol=1.0e-8,
    ):
        raise DeePHFCapabilityError(
            "the stored RHF total energy is inconsistent with its AO state: "
            f"{reference.e_tot:.12g} != {recomputed_energy:.12g}"
        )
    if not np.isfinite(mol.atom_coords(unit="Bohr")).all():
        raise DeePHFCapabilityError("the molecular geometry must be finite")
    return reference


@dataclass(frozen=True)
class RHFReferenceSnapshot:
    """PySCF-independent provenance view of one molecular RHF reference."""

    reference_class: str
    converged: bool
    state_fingerprint: str
    charge: int
    spin: int
    electron_count: int
    occupations: tuple[float, ...]
    basis: Mapping[str, Any]
    ecp: Mapping[str, Any]
    geometry_bohr: tuple[tuple[float, float, float], ...]
    atom_charges: tuple[int, ...]
    ao_count: int
    ao_labels: tuple[str, ...]
    scf_controls: Mapping[str, Any]


@dataclass(frozen=True)
class RHFRootSnapshot:
    """Immutable occupied-subspace anchor for strict RHF root tracking."""

    system_fingerprint: str
    state_fingerprint: str
    integrity_fingerprint: str
    parent_state_fingerprint: str | None
    minimum_occupied_overlap: float
    occupied_coefficients: np.ndarray
    occupations: np.ndarray
    _molecule: Any = field(repr=False, compare=False)

    @property
    def molecule(self):
        """Return an isolated copy of the root-defining molecule."""
        return deepcopy(self._molecule)


@dataclass(frozen=True)
class RHFResponseDiagnostics:
    """Independent diagnostics for one full or coordinate-block CPHF solve."""

    minimum_orbital_gap: float
    pyscf_version: str
    cphf_tolerance: float
    maximum_residual: float
    residual_rms: float
    residual_tolerance: float
    invariant_tolerance: float
    orbital_gap_tolerance: float
    max_cycle: int
    max_refinement_cycles: int
    level_shift: float
    response_dimension: int
    operator_is_self_adjoint: bool
    metric_residual: float
    idempotency_residual: float
    particle_number_residual: float
    refinement_cycles: int
    residual_history: tuple[float, ...]


@dataclass(frozen=True)
class RHFResponse:
    """Own one canonical MO response; derived properties allocate without caching."""

    reference_identity: int
    state_fingerprint: str
    integrity_fingerprint: str
    atom_indices: tuple[int, ...]
    mo_response: np.ndarray
    _mo_coefficients: np.ndarray
    _mo_occupations: np.ndarray
    overlap_derivative: np.ndarray
    hamiltonian_derivative: np.ndarray
    orbital_response_residual: np.ndarray
    diagnostics: RHFResponseDiagnostics

    def _mo_partition(self, occupied_virtual: bool) -> np.ndarray:
        occupied = self._mo_occupations > 0
        selected = ~occupied if occupied_virtual else occupied
        result = np.zeros_like(self.mo_response)
        result[..., selected, :] = self.mo_response[..., selected, :]
        return _immutable_array(result)

    def _coefficient_response(self, mo_response: np.ndarray) -> np.ndarray:
        return _immutable_array(
            np.einsum("mp,...pi->...mi", self._mo_coefficients, mo_response)
        )

    def _density_response(self, mo_response: np.ndarray) -> np.ndarray:
        occupied = self._mo_occupations > 0
        occupied_coefficients = self._mo_coefficients[:, occupied]
        coefficient_response = np.einsum(
            "mp,...pi->...mi", self._mo_coefficients, mo_response
        )
        density = np.einsum(
            "...pi,qi,i->...pq",
            coefficient_response,
            occupied_coefficients,
            self._mo_occupations[occupied],
        )
        return _immutable_array(density + density.swapaxes(-1, -2))

    @property
    def mo_response_occupied_virtual(self) -> np.ndarray:
        return self._mo_partition(True)

    @property
    def mo_response_metric(self) -> np.ndarray:
        return self._mo_partition(False)

    @property
    def coefficient_response(self) -> np.ndarray:
        return self._coefficient_response(self.mo_response)

    @property
    def coefficient_response_occupied_virtual(self) -> np.ndarray:
        return self._coefficient_response(self.mo_response_occupied_virtual)

    @property
    def coefficient_response_metric(self) -> np.ndarray:
        return self._coefficient_response(self.mo_response_metric)

    @property
    def density_response(self) -> np.ndarray:
        return self._density_response(self.mo_response)

    @property
    def density_response_occupied_virtual(self) -> np.ndarray:
        return self._density_response(self.mo_response_occupied_virtual)

    @property
    def density_response_metric(self) -> np.ndarray:
        return self._density_response(self.mo_response_metric)


@dataclass(frozen=True)
class RHFBlockedResponseSummary:
    """Compact provenance for a direct response consumed block by block."""

    reference_identity: int
    state_fingerprint: str
    integrity_fingerprint: str
    coordinate_block_size: int
    block_count: int
    diagnostics: RHFResponseDiagnostics


def blocked_response_summary_integrity_fingerprint(
    summary: RHFBlockedResponseSummary,
) -> str:
    """Return a digest covering one compact blocked-response summary."""
    digest = hashlib.sha256()
    for field_definition in fields(summary):
        if field_definition.name == "integrity_fingerprint":
            continue
        digest.update(field_definition.name.encode("utf-8"))
        digest.update(repr(getattr(summary, field_definition.name)).encode("utf-8"))
    return digest.hexdigest()


@dataclass(frozen=True)
class RHFAdjointDiagnostics:
    """Independent diagnostics for one correction-specific RHF adjoint."""

    minimum_orbital_gap: float
    pyscf_version: str
    residual_tolerance: float
    orbital_gap_tolerance: float
    response_dimension: int
    operator_is_self_adjoint: bool
    objective_symmetry_tolerance: float
    objective_symmetry_residual: float
    adjoint_density_symmetry_residual: float
    adjoint_potential_symmetry_residual: float
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
class RHFAdjoint:
    """Immutable RHF Z-vector and its nuclear response contractions."""

    reference_identity: int
    state_fingerprint: str
    integrity_fingerprint: str
    operator_fingerprint: str
    atom_indices: tuple[int, ...]
    objective_ao_potential: np.ndarray
    objective_orbital_gradient: np.ndarray
    zvector: np.ndarray
    residual: np.ndarray
    adjoint_ao_density: np.ndarray
    adjoint_ao_potential: np.ndarray
    correction_gradient_metric: np.ndarray
    correction_gradient_adjoint_nuclear: np.ndarray
    correction_gradient_adjoint_metric: np.ndarray
    correction_gradient_occupied_virtual: np.ndarray
    correction_gradient_response: np.ndarray
    diagnostics: RHFAdjointDiagnostics


def _version_series(version: str) -> tuple[int, int]:
    components = version.split(".")
    try:
        return int(components[0]), int(components[1])
    except (IndexError, ValueError) as error:
        raise DeePHFCapabilityError(
            f"cannot interpret the PySCF version {version!r}"
        ) from error


def validate_pyscf_version() -> None:
    """Require the PySCF series characterized by the RHF response adapters."""
    series = _version_series(pyscf.__version__)
    if series != SUPPORTED_PYSCF_SERIES:
        raise DeePHFCapabilityError(
            "the RHF response adapter supports PySCF 2.14; "
            f"found {pyscf.__version__}"
        )


def molecule_science_fingerprint(molecule) -> str:
    """Fingerprint stable molecular geometry and AO data, excluding libcint scratch."""
    if type(molecule) is not gto.mole.Mole:
        raise DeePHFCapabilityError(
            "RHF science-state fingerprints require a native pyscf.gto.Mole"
        )
    environment = np.asarray(molecule._env).copy()
    environment[: gto_mole.PTR_ENV_START] = 0.0
    digest = hashlib.sha256()
    digest.update(pyscf.__version__.encode("utf-8"))
    values = (
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
        getattr(molecule, "symmetry_subgroup", None),
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


def reference_fingerprint(reference) -> str:
    """Return a scratch-independent fingerprint of the scientific RHF state."""
    trusted = transaction_reference_fingerprint(reference)
    if trusted is not None:
        return trusted
    digest = hashlib.sha256()
    values = (
        f"{type(reference).__module__}.{type(reference).__qualname__}",
        bool(reference.converged),
        molecule_science_fingerprint(reference.mol),
        float(reference.e_tot),
        reference.mo_coeff,
        reference.mo_energy,
        reference.mo_occ,
        reference.make_rdm1(),
    )
    for value in values:
        _update_fingerprint_value(digest, value)
    return digest.hexdigest()


def _immutable_metadata(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _immutable_metadata(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _immutable_metadata(item) for key, item in value.items()}
        )
    if isinstance(value, (tuple, list)):
        return tuple(_immutable_metadata(item) for item in value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise DeePHFCapabilityError(
        "cannot snapshot PySCF reference metadata of type "
        f"{type(value).__name__}"
    )


def reference_provenance_snapshot(reference) -> RHFReferenceSnapshot:
    """Snapshot RHF reference metadata behind the PySCF 2.14 boundary."""
    validate_pyscf_version()
    reference = validate_reference(reference)
    molecule = reference.mol
    controls = {
        name: _immutable_metadata(getattr(reference, name, None))
        for name in (
            "conv_tol",
            "conv_tol_grad",
            "conv_tol_cpscf",
            "max_cycle",
            "level_shift",
            "diis_space",
            "direct_scf",
            "conv_check",
        )
    }
    geometry = np.asarray(molecule.atom_coords(unit="Bohr"), dtype=np.float64)
    atom_charges = np.asarray(molecule.atom_charges())
    occupations = np.asarray(reference.mo_occ, dtype=np.float64)
    return RHFReferenceSnapshot(
        reference_class=(
            f"{type(reference).__module__}.{type(reference).__qualname__}"
        ),
        converged=bool(reference.converged),
        state_fingerprint=reference_fingerprint(reference),
        charge=int(molecule.charge),
        spin=int(molecule.spin),
        electron_count=int(molecule.nelectron),
        occupations=tuple(float(value) for value in occupations),
        basis=_immutable_metadata(molecule._basis),
        ecp=_immutable_metadata(molecule._ecp),
        geometry_bohr=tuple(
            tuple(float(component) for component in coordinates)
            for coordinates in geometry
        ),
        atom_charges=tuple(int(value) for value in atom_charges),
        ao_count=int(molecule.nao),
        ao_labels=tuple(molecule.ao_labels()),
        scf_controls=MappingProxyType(controls),
    )


def _immutable_array(value: np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(value)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


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
    if isinstance(value, Mapping):
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
    raise RHFScannerReferenceError(
        "scanner metadata cannot fingerprint values of type "
        f"{type(value).__name__}"
    )


def _molecule_static_fingerprint(molecule) -> str:
    """Fingerprint every supported molecular property except coordinates."""
    if type(molecule) is not gto.mole.Mole:
        raise TypeError("scanner molecules must be exact native pyscf.gto.Mole objects")
    environment = np.asarray(molecule._env).copy()
    environment[: gto_mole.PTR_ENV_START] = 0.0
    atoms = np.asarray(molecule._atm)
    for atom in atoms:
        coordinate_pointer = int(atom[gto_mole.PTR_COORD])
        environment[coordinate_pointer : coordinate_pointer + 3] = 0.0
    digest = hashlib.sha256()
    static_values = (
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
        getattr(molecule, "symmetry_subgroup", None),
        float(getattr(molecule, "omega", 0.0)),
        getattr(molecule, "nucmod", None),
        tuple(molecule.atom_symbol(index) for index in range(molecule.natm)),
        np.asarray(molecule.atom_charges()),
        tuple(molecule.ao_labels()),
        np.asarray(molecule._atm),
        np.asarray(molecule._bas),
        environment,
        molecule._basis,
        molecule._ecp,
        getattr(molecule, "_pseudo", None),
    )
    for value in static_values:
        _update_fingerprint_value(digest, value)
    return digest.hexdigest()


def _reject_molecule_instance_callables(molecule) -> None:
    """Reject executable attributes attached directly to a scanner input."""
    instance_state = object.__getattribute__(molecule, "__dict__")
    callable_fields = sorted(
        name if type(name) is str else f"<{type(name).__name__}>"
        for name, value in instance_state.items()
        if callable(value)
    )
    if callable_fields:
        raise RHFScannerReferenceError(
            "scanner molecules cannot contain callable instance hooks; "
            f"active fields: {', '.join(callable_fields)}"
        )


def _root_integrity_fingerprint(root: RHFRootSnapshot) -> str:
    digest = hashlib.sha256()
    for value in (
        root.system_fingerprint,
        root.state_fingerprint,
        root.parent_state_fingerprint,
        root.minimum_occupied_overlap,
        root.occupied_coefficients,
        root.occupations,
        _molecule_static_fingerprint(root._molecule),
        np.asarray(root._molecule.atom_coords(unit="Bohr")),
    ):
        _update_fingerprint_value(digest, value)
    return digest.hexdigest()


_SCANNER_INITIAL_GUESSES = frozenset(
    {"minao", "atom", "huckel", "mod_huckel", "hcore", "1e", "sap"}
)


def _scanner_real_control(
    value,
    name: str,
    *,
    positive: bool = False,
    minimum: float | None = None,
    maximum_exclusive: float | None = None,
    allow_none: bool = False,
):
    if value is None and allow_none:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise RHFScannerReferenceError(
            f"scanner SCF control {name} must be a real number"
        )
    result = float(value)
    if not np.isfinite(result):
        raise RHFScannerReferenceError(
            f"scanner SCF control {name} must be finite"
        )
    if positive and result <= 0:
        raise RHFScannerReferenceError(
            f"scanner SCF control {name} must be positive"
        )
    if minimum is not None and result < minimum:
        raise RHFScannerReferenceError(
            f"scanner SCF control {name} must be at least {minimum}"
        )
    if maximum_exclusive is not None and result >= maximum_exclusive:
        raise RHFScannerReferenceError(
            f"scanner SCF control {name} must be below {maximum_exclusive}"
        )
    return result


def _scanner_integer_control(value, name: str, *, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise RHFScannerReferenceError(
            f"scanner SCF control {name} must be an integer"
        )
    try:
        result = operator.index(value)
    except TypeError as error:
        raise RHFScannerReferenceError(
            f"scanner SCF control {name} must be an integer"
        ) from error
    if result < minimum:
        raise RHFScannerReferenceError(
            f"scanner SCF control {name} must be at least {minimum}"
        )
    return int(result)


def _scanner_boolean_control(value, name: str) -> bool:
    if type(value) is not bool:
        raise RHFScannerReferenceError(
            f"scanner SCF control {name} must be boolean"
        )
    return value


def _scanner_scf_controls(reference) -> Mapping[str, Any]:
    if getattr(reference, "callback", None) is not None:
        raise RHFScannerReferenceError(
            "scanner references must not use an SCF callback"
        )
    if getattr(reference, "diis_file", None) is not None:
        raise RHFScannerReferenceError(
            "scanner references must not use an external DIIS file"
        )
    if getattr(reference, "DIIS", None) is not scf_hf.RHF.DIIS:
        raise RHFScannerReferenceError(
            "scanner references must not use a custom DIIS implementation"
        )
    init_guess = getattr(reference, "init_guess", None)
    if type(init_guess) is not str:
        raise RHFScannerReferenceError(
            "scanner SCF control init_guess must be a string"
        )
    init_guess = init_guess.lower()
    if init_guess.startswith("chk"):
        raise RHFScannerReferenceError(
            "scanner SCF control init_guess must not use a checkpoint"
        )
    if init_guess not in _SCANNER_INITIAL_GUESSES:
        raise RHFScannerReferenceError(
            f"scanner SCF control init_guess is unsupported: {init_guess!r}"
        )
    sap_basis = getattr(reference, "sap_basis", None)
    if type(sap_basis) is not str or not sap_basis:
        raise RHFScannerReferenceError(
            "scanner SCF control sap_basis must be a nonempty string"
        )
    controls = {
        "conv_tol": _scanner_real_control(
            getattr(reference, "conv_tol", None),
            "conv_tol",
            positive=True,
        ),
        "conv_tol_grad": _scanner_real_control(
            getattr(reference, "conv_tol_grad", None),
            "conv_tol_grad",
            positive=True,
            allow_none=True,
        ),
        "conv_tol_cpscf": _scanner_real_control(
            getattr(reference, "conv_tol_cpscf", None),
            "conv_tol_cpscf",
            positive=True,
        ),
        "max_cycle": _scanner_integer_control(
            getattr(reference, "max_cycle", None),
            "max_cycle",
            minimum=1,
        ),
        "conv_check": _scanner_boolean_control(
            getattr(reference, "conv_check", None),
            "conv_check",
        ),
        "init_guess": init_guess,
        "level_shift": _scanner_real_control(
            getattr(reference, "level_shift", None),
            "level_shift",
            minimum=0.0,
        ),
        "damp": _scanner_real_control(
            getattr(reference, "damp", None),
            "damp",
            minimum=0.0,
            maximum_exclusive=1.0,
        ),
        "diis": _scanner_boolean_control(
            getattr(reference, "diis", None),
            "diis",
        ),
        "diis_space": _scanner_integer_control(
            getattr(reference, "diis_space", None),
            "diis_space",
            minimum=1,
        ),
        "diis_start_cycle": _scanner_integer_control(
            getattr(reference, "diis_start_cycle", None),
            "diis_start_cycle",
            minimum=0,
        ),
        "diis_damp": _scanner_real_control(
            getattr(reference, "diis_damp", None),
            "diis_damp",
            minimum=0.0,
            maximum_exclusive=1.0,
        ),
        "diis_space_rollback": _scanner_integer_control(
            getattr(reference, "diis_space_rollback", None),
            "diis_space_rollback",
            minimum=0,
        ),
        "direct_scf": _scanner_boolean_control(
            getattr(reference, "direct_scf", None),
            "direct_scf",
        ),
        "direct_scf_tol": _scanner_real_control(
            getattr(reference, "direct_scf_tol", None),
            "direct_scf_tol",
            positive=True,
        ),
        "sap_basis": sap_basis,
        "max_memory": _scanner_real_control(
            getattr(reference, "max_memory", None),
            "max_memory",
            positive=True,
        ),
        "verbose": _scanner_integer_control(
            getattr(reference, "verbose", None),
            "verbose",
            minimum=0,
        ),
    }
    return MappingProxyType(controls)


class RHFScannerReferenceFactory:
    """Build independent native RHF references and track one continuous root."""

    def __init__(self, reference, *, root_overlap_tolerance: float = 0.5):
        validate_pyscf_version()
        reference = validate_reference(reference)
        if isinstance(root_overlap_tolerance, (bool, np.bool_)):
            raise TypeError(
                "scanner root_overlap_tolerance must be a real number"
            )
        try:
            root_overlap_tolerance = float(root_overlap_tolerance)
        except (TypeError, ValueError) as error:
            raise TypeError(
                "scanner root_overlap_tolerance must be a real number"
            ) from error
        if (
            not np.isfinite(root_overlap_tolerance)
            or root_overlap_tolerance <= 0.0
            or root_overlap_tolerance > 1.0
        ):
            raise ValueError(
                "scanner root_overlap_tolerance must be finite and in (0, 1]"
            )
        try:
            template = deepcopy(reference.mol)
        except Exception as error:
            raise RHFScannerReferenceError(
                f"the scanner molecule template could not be copied: {error}"
            ) from error
        if type(template) is not gto.mole.Mole:
            raise RHFScannerReferenceError(
                "the scanner molecule copy is not an exact native pyscf.gto.Mole"
            )
        self.root_overlap_tolerance = root_overlap_tolerance
        self._template = template
        self._system_fingerprint = _molecule_static_fingerprint(template)
        self._atom_count = int(template.natm)
        self._ao_count = int(template.nao)
        self._occupations = _immutable_array(np.asarray(reference.mo_occ))
        self._scf_controls = _scanner_scf_controls(reference)
        self._issued_roots = {}
        self._initial_root = self._root_snapshot(
            reference,
            parent_state_fingerprint=None,
            minimum_occupied_overlap=1.0,
        )

    @property
    def initial_root(self) -> RHFRootSnapshot:
        return self._initial_root

    @property
    def system_fingerprint(self) -> str:
        return self._system_fingerprint

    @property
    def scf_controls(self) -> Mapping[str, Any]:
        return self._scf_controls

    def _root_snapshot(
        self,
        reference,
        *,
        parent_state_fingerprint: str | None,
        minimum_occupied_overlap: float,
    ) -> RHFRootSnapshot:
        occupations = np.asarray(reference.mo_occ)
        occupied = occupations > 0
        occupied_coefficients = np.asarray(reference.mo_coeff)[:, occupied]
        root = RHFRootSnapshot(
            system_fingerprint=self._system_fingerprint,
            state_fingerprint=reference_fingerprint(reference),
            integrity_fingerprint="",
            parent_state_fingerprint=parent_state_fingerprint,
            minimum_occupied_overlap=float(minimum_occupied_overlap),
            occupied_coefficients=_immutable_array(occupied_coefficients),
            occupations=_immutable_array(occupations),
            _molecule=deepcopy(reference.mol),
        )
        root = replace(
            root,
            integrity_fingerprint=_root_integrity_fingerprint(root),
        )
        self._register_root(root)
        return root

    def _register_root(self, root: RHFRootSnapshot) -> None:
        """Record one factory-issued root without retaining it indefinitely."""
        identity = id(root)
        factory_reference = weakref.ref(self)

        def discard(reference, *, identity=identity, factory_reference=factory_reference):
            factory = factory_reference()
            if factory is None:
                return
            issued = factory._issued_roots.get(identity)
            if issued is not None and issued[0] is reference:
                factory._issued_roots.pop(identity, None)

        root_reference = weakref.ref(root, discard)
        self._issued_roots[identity] = (
            root_reference,
            root.integrity_fingerprint,
            root.state_fingerprint,
            root.parent_state_fingerprint,
            root.minimum_occupied_overlap,
        )

    def _validate_root(self, root) -> RHFRootSnapshot:
        if type(root) is not RHFRootSnapshot:
            raise TypeError("scanner previous_root must be an RHFRootSnapshot")
        issued = self._issued_roots.get(id(root))
        if issued is None or issued[0]() is not root:
            raise RHFScannerReferenceError(
                "scanner previous_root was not issued by this reference factory"
            )
        if (
            root.integrity_fingerprint != issued[1]
            or root.state_fingerprint != issued[2]
            or root.parent_state_fingerprint != issued[3]
            or root.minimum_occupied_overlap != issued[4]
        ):
            raise RHFScannerReferenceError(
                "scanner previous_root changed after it was issued"
            )
        if root.system_fingerprint != self._system_fingerprint:
            raise RHFScannerReferenceError(
                "scanner previous_root belongs to another molecular system"
            )
        if root.integrity_fingerprint != _root_integrity_fingerprint(root):
            raise RHFScannerReferenceError(
                "scanner previous_root failed its integrity check"
            )
        if type(root._molecule) is not gto.mole.Mole:
            raise RHFScannerReferenceError(
                "scanner previous_root has an invalid molecule type"
            )
        if _molecule_static_fingerprint(root._molecule) != self._system_fingerprint:
            raise RHFScannerReferenceError(
                "scanner previous_root molecule has incompatible static metadata"
            )
        expected_occupied = int(np.count_nonzero(self._occupations > 0))
        array_fields = (
            (
                root.occupied_coefficients,
                (self._ao_count, expected_occupied),
                "occupied coefficients",
            ),
            (root.occupations, (self._ao_count,), "occupations"),
        )
        for value, shape, name in array_fields:
            if (
                not isinstance(value, np.ndarray)
                or value.shape != shape
                or value.dtype != np.dtype(np.float64)
                or np.iscomplexobj(value)
                or not np.isfinite(value).all()
                or value.flags.writeable
            ):
                raise RHFScannerReferenceError(
                    f"scanner previous_root {name} are invalid"
                )
        if not np.array_equal(root.occupations, self._occupations):
            raise RHFScannerReferenceError(
                "scanner previous_root occupations changed"
            )
        coordinates = np.asarray(root._molecule.atom_coords(unit="Bohr"))
        if (
            coordinates.shape != (self._atom_count, 3)
            or coordinates.dtype != np.dtype(np.float64)
            or not np.isfinite(coordinates).all()
        ):
            raise RHFScannerReferenceError(
                "scanner previous_root geometry is invalid"
            )
        if (
            not np.isfinite(root.minimum_occupied_overlap)
            or root.minimum_occupied_overlap < 0.0
            or root.minimum_occupied_overlap > 1.0 + 1.0e-10
        ):
            raise RHFScannerReferenceError(
                "scanner previous_root overlap diagnostic is invalid"
            )
        fingerprints = (
            root.system_fingerprint,
            root.state_fingerprint,
            root.integrity_fingerprint,
        )
        if root.parent_state_fingerprint is not None:
            fingerprints += (root.parent_state_fingerprint,)
        if any(
            type(value) is not str
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in fingerprints
        ):
            raise RHFScannerReferenceError(
                "scanner previous_root fingerprints are invalid"
            )
        return root

    def _coordinates(self, mol_or_coordinates) -> np.ndarray:
        if type(mol_or_coordinates) is gto.mole.Mole:
            _reject_molecule_instance_callables(mol_or_coordinates)
            fingerprints_before = (
                _molecule_static_fingerprint(mol_or_coordinates),
                molecule_science_fingerprint(mol_or_coordinates),
            )
            if fingerprints_before[0] != self._system_fingerprint:
                raise RHFScannerReferenceError(
                    "scanner molecule static metadata does not match the template"
                )
            value = gto_mole.Mole.atom_coords(
                mol_or_coordinates,
                unit="Bohr",
            )
            _reject_molecule_instance_callables(mol_or_coordinates)
            fingerprints_after = (
                _molecule_static_fingerprint(mol_or_coordinates),
                molecule_science_fingerprint(mol_or_coordinates),
            )
            if fingerprints_after != fingerprints_before:
                raise RHFScannerReferenceError(
                    "scanner molecule state changed while coordinates were read"
                )
        elif isinstance(mol_or_coordinates, gto.mole.Mole):
            raise TypeError(
                "scanner molecules must be exact native pyscf.gto.Mole objects"
            )
        else:
            try:
                value = np.asarray(mol_or_coordinates)
            except Exception as error:
                raise TypeError(
                    f"scanner coordinates are not a numerical array: {error}"
                ) from error
            if (
                value.dtype.hasobject
                or np.iscomplexobj(value)
                or not (
                    np.issubdtype(value.dtype, np.integer)
                    or np.issubdtype(value.dtype, np.floating)
                )
            ):
                raise TypeError(
                    "scanner coordinates must contain real integer or floating values"
                )
        if value.shape != (self._atom_count, 3):
            raise ValueError(
                "scanner coordinates have shape "
                f"{value.shape}; expected {(self._atom_count, 3)}"
            )
        coordinates = np.asarray(value, dtype=np.float64)
        if not np.isfinite(coordinates).all():
            raise ValueError("scanner coordinates must be finite")
        return np.ascontiguousarray(coordinates).copy()

    def _fresh_molecule(self, coordinates: np.ndarray):
        try:
            molecule = deepcopy(self._template)
            molecule.set_geom_(coordinates, unit="Bohr", inplace=True)
        except Exception as error:
            raise RHFScannerReferenceError(
                f"the fresh scanner molecule could not be built: {error}"
            ) from error
        if (
            type(molecule) is not gto.mole.Mole
            or _molecule_static_fingerprint(molecule) != self._system_fingerprint
        ):
            raise RHFScannerReferenceError(
                "the fresh scanner molecule changed its static metadata"
            )
        return molecule

    def _fresh_reference(self, molecule):
        reference = scf_hf.RHF(molecule)
        if type(reference) is not scf_hf.RHF:
            raise RHFScannerReferenceError(
                "the scanner did not construct an exact native RHF reference"
            )
        for name, value in self._scf_controls.items():
            setattr(reference, name, value)
        reference.chkfile = None
        reference.callback = None
        reference.diis_file = None
        try:
            reference.kernel(dm0=None)
        except Exception as error:
            raise RHFScannerReferenceError(
                f"fresh scanner RHF evaluation failed: {error}"
            ) from error
        if not reference.converged:
            raise DeePHFCapabilityError(
                "the fresh scanner RHF reference did not converge"
            )
        return validate_reference(reference)

    def _occupied_overlap(
        self,
        previous_root: RHFRootSnapshot,
        candidate_reference,
    ) -> float:
        candidate_occupations = np.asarray(candidate_reference.mo_occ)
        if not np.array_equal(candidate_occupations, previous_root.occupations):
            raise RHFScannerReferenceError(
                "the fresh scanner RHF occupations changed from the root anchor"
            )
        candidate_occupied = np.asarray(candidate_reference.mo_coeff)[
            :, candidate_occupations > 0
        ]
        try:
            cross_overlap = gto.intor_cross(
                "int1e_ovlp",
                previous_root._molecule,
                candidate_reference.mol,
            )
        except Exception as error:
            raise RHFScannerReferenceError(
                f"scanner cross-AO overlap construction failed: {error}"
            ) from error
        cross_overlap = _validated_float64_array(
            cross_overlap,
            (self._ao_count, self._ao_count),
            "scanner cross-AO overlap",
        )
        occupied_overlap = (
            previous_root.occupied_coefficients.T
            @ cross_overlap
            @ candidate_occupied
        )
        if not np.isfinite(occupied_overlap).all():
            raise RHFScannerReferenceError(
                "scanner occupied-subspace overlap is nonfinite"
            )
        try:
            singular_values = np.linalg.svd(
                occupied_overlap,
                compute_uv=False,
            )
        except np.linalg.LinAlgError as error:
            raise RHFScannerReferenceError(
                f"scanner occupied-subspace overlap SVD failed: {error}"
            ) from error
        minimum_overlap = float(np.min(singular_values))
        if (
            not np.isfinite(minimum_overlap)
            or minimum_overlap < self.root_overlap_tolerance
        ):
            raise RHFScannerReferenceError(
                "fresh scanner RHF occupied subspace is discontinuous: "
                f"minimum overlap {minimum_overlap:.6f} < "
                f"{self.root_overlap_tolerance:.6f}"
            )
        return minimum_overlap

    def build(
        self,
        mol_or_coordinates,
        previous_root: RHFRootSnapshot,
    ) -> tuple[Any, RHFRootSnapshot]:
        """Build one fresh RHF reference without changing the root anchor."""
        previous_root = self._validate_root(previous_root)
        coordinates = self._coordinates(mol_or_coordinates)
        molecule = self._fresh_molecule(coordinates)
        reference = self._fresh_reference(molecule)
        minimum_overlap = self._occupied_overlap(previous_root, reference)
        candidate_root = self._root_snapshot(
            reference,
            parent_state_fingerprint=previous_root.state_fingerprint,
            minimum_occupied_overlap=minimum_overlap,
        )
        return reference, candidate_root


def response_integrity_fingerprint(response: RHFResponse) -> str:
    """Return a digest covering every response field except the digest itself."""
    digest = hashlib.sha256()
    for field in fields(response):
        if field.name == "integrity_fingerprint":
            continue
        value = getattr(response, field.name)
        digest.update(field.name.encode("utf-8"))
        if isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            digest.update(array.dtype.str.encode("ascii"))
            digest.update(repr(array.shape).encode("ascii"))
            digest.update(array.tobytes())
        else:
            digest.update(repr(value).encode("utf-8"))
    return digest.hexdigest()


def adjoint_integrity_fingerprint(adjoint: RHFAdjoint) -> str:
    """Return a digest covering every RHF adjoint field except its digest."""
    digest = hashlib.sha256()
    for field in fields(adjoint):
        if field.name == "integrity_fingerprint":
            continue
        value = getattr(adjoint, field.name)
        digest.update(field.name.encode("utf-8"))
        if isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            digest.update(array.dtype.str.encode("ascii"))
            digest.update(repr(array.shape).encode("ascii"))
            digest.update(array.tobytes())
        else:
            digest.update(repr(value).encode("utf-8"))
    return digest.hexdigest()


def _cycle_limit(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"response {name} must be an integer")
    try:
        return operator.index(value)
    except TypeError as error:
        raise ValueError(f"response {name} must be an integer") from error


def _adjoint_real_control(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"adjoint {name} must be a real numeric scalar")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"adjoint {name} must be finite")
    return result


def _validated_float64_array(value, expected_shape, name: str) -> np.ndarray:
    try:
        array = np.asarray(value)
    except Exception as error:
        raise RHFResponseError(f"{name} is not a numerical array: {error}") from error
    if array.shape != expected_shape:
        raise RHFResponseError(
            f"unexpected {name} shape {array.shape}; expected {expected_shape}"
        )
    if array.dtype != np.dtype(np.float64) or np.iscomplexobj(array):
        raise RHFResponseError(f"{name} must be a real float64 array")
    if not np.isfinite(array).all():
        raise RHFResponseError(f"{name} must be finite")
    return array


class _RHFLinearResponseCore:
    """Shared PySCF 2.14 molecular RHF linear-response operations."""

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
        self.reference = validate_reference(reference)
        self.cphf_tolerance = float(cphf_tolerance)
        self.residual_tolerance = float(residual_tolerance)
        self.invariant_tolerance = float(invariant_tolerance)
        self.orbital_gap_tolerance = float(orbital_gap_tolerance)
        self.max_cycle = _cycle_limit(max_cycle, "max_cycle")
        self.max_refinement_cycles = _cycle_limit(
            max_refinement_cycles,
            "max_refinement_cycles",
        )
        self.level_shift = float(level_shift)
        self.operator_stability_tolerance = float(
            operator_stability_tolerance
        )
        self.operator_condition_tolerance = float(
            operator_condition_tolerance
        )
        self.operator_symmetry_tolerance = float(
            operator_symmetry_tolerance
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
        if not np.isfinite(tolerance_values).all():
            raise ValueError("response tolerances must be finite")
        if not np.isfinite(self.level_shift):
            raise ValueError("response level_shift must be finite")
        if self.cphf_tolerance <= 0 or self.residual_tolerance <= 0:
            raise ValueError("response tolerances must be positive")
        if self.invariant_tolerance <= 0 or self.orbital_gap_tolerance <= 0:
            raise ValueError("response tolerances must be positive")
        if (
            self.operator_stability_tolerance <= 0
            or self.operator_condition_tolerance <= 1
            or self.operator_symmetry_tolerance <= 0
        ):
            raise ValueError("response operator tolerances are invalid")
        if self.max_cycle <= 0 or self.max_refinement_cycles < 0:
            raise ValueError("response cycle limits are invalid")
        if self.operator_dimension_limit <= 0:
            raise ValueError("response operator_dimension_limit must be positive")

    @property
    def molecule(self):
        return self.reference.mol

    def _response_atom_indices(self, atom_indices) -> tuple[int, ...]:
        if atom_indices is None:
            return tuple(range(self.molecule.natm))
        try:
            indices = tuple(atom_indices)
        except TypeError as error:
            raise TypeError("response atom_indices must be iterable") from error
        if not indices:
            raise ValueError("response atom_indices must not be empty")
        validated_indices = []
        for atom_index in indices:
            if isinstance(atom_index, (bool, np.bool_)):
                raise TypeError("response atom indices must be integers")
            try:
                validated = operator.index(atom_index)
            except TypeError as error:
                raise TypeError("response atom indices must be integers") from error
            if validated < 0 or validated >= self.molecule.natm:
                raise IndexError("response atom index is outside the molecule")
            if validated in validated_indices:
                raise ValueError(
                    "response atom_indices must not contain duplicates"
                )
            validated_indices.append(validated)
        return tuple(validated_indices)

    def _state(self):
        coefficient = np.asarray(self.reference.mo_coeff)
        energy = np.asarray(self.reference.mo_energy)
        occupation = np.asarray(self.reference.mo_occ)
        occupied = occupation > 0
        virtual = occupation == 0
        if not np.any(occupied) or not np.any(virtual):
            raise DeePHFCapabilityError(
                "RHF response requires occupied and virtual orbitals"
            )
        gaps = energy[virtual, None] - energy[occupied]
        minimum_gap = float(np.min(gaps))
        if not np.isfinite(minimum_gap) or minimum_gap <= self.orbital_gap_tolerance:
            raise DeePHFCapabilityError(
                "RHF occupied-virtual gap is outside the strict response domain: "
                f"{minimum_gap:.3e} <= {self.orbital_gap_tolerance:.3e}"
            )
        return coefficient, energy, occupation, occupied, virtual, minimum_gap

    def _overlap_derivative(self, atom_indices=None) -> np.ndarray:
        molecule = self.molecule
        atom_indices = self._response_atom_indices(atom_indices)
        nao = molecule.nao
        try:
            integral = -molecule.intor("int1e_ipovlp", comp=3)
        except Exception as error:
            raise RHFResponseError(
                f"PySCF overlap-derivative construction failed: {error}"
            ) from error
        integral = _validated_float64_array(
            integral,
            (3, nao, nao),
            "overlap-derivative integral",
        )
        result = np.zeros((len(atom_indices), 3, nao, nao))
        atom_slices = molecule.aoslice_by_atom()
        for result_index, atom_index in enumerate(atom_indices):
            atom_slice = atom_slices[atom_index]
            ao_start, ao_stop = atom_slice[2:]
            result[result_index, :, ao_start:ao_stop] += integral[
                :, ao_start:ao_stop
            ]
            result[result_index, :, :, ao_start:ao_stop] += integral[
                :, ao_start:ao_stop
            ].transpose(0, 2, 1)
        return result

    def _hamiltonian_derivative(
        self,
        coefficient: np.ndarray,
        occupation: np.ndarray,
        atom_indices=None,
    ) -> np.ndarray:
        atom_indices = self._response_atom_indices(atom_indices)
        try:
            hessian = rhf_hessian.Hessian(self.reference)
            derivatives = hessian.make_h1(
                coefficient,
                occupation,
                atmlst=atom_indices,
            )
            derivatives = [derivatives[index] for index in atom_indices]
        except Exception as error:
            raise RHFResponseError(
                f"PySCF RHF Hamiltonian derivative construction failed: {error}"
            ) from error
        if derivatives is None:
            raise RHFResponseError(
                "PySCF RHF Hamiltonian derivative is incomplete"
            )
        expected = (len(atom_indices), 3, self.molecule.nao, self.molecule.nao)
        return _validated_float64_array(
            derivatives,
            expected,
            "Hamiltonian derivative",
        )

    def _density_from_mo_response(
        self,
        mo_response: np.ndarray,
        coefficient: np.ndarray,
        occupation: np.ndarray,
        occupied: np.ndarray,
    ) -> np.ndarray:
        occupied_coefficients = coefficient[:, occupied]
        coefficient_response = np.einsum(
            "mp,...pi->...mi",
            coefficient,
            mo_response,
        )
        density_response = np.einsum(
            "...pi,qi,i->...pq",
            coefficient_response,
            occupied_coefficients,
            occupation[occupied],
        )
        return density_response + density_response.swapaxes(-1, -2)

    def _induced_potential(self, density_response: np.ndarray) -> np.ndarray:
        flat_density = np.asarray(density_response).reshape(
            -1,
            self.molecule.nao,
            self.molecule.nao,
        )
        try:
            coulomb, exchange = scf_hf.get_jk(
                self.molecule,
                flat_density,
                hermi=1,
            )
            coulomb = _validated_float64_array(
                coulomb,
                flat_density.shape,
                "induced Coulomb response",
            )
            exchange = _validated_float64_array(
                exchange,
                flat_density.shape,
                "induced exchange response",
            )
        except RHFResponseError:
            raise
        except Exception as error:
            raise RHFResponseError(
                f"PySCF induced-potential construction failed: {error}"
            ) from error
        return (coulomb - 0.5 * exchange).reshape(density_response.shape)

    def _response_operator_matrix_and_diagnostics(
        self,
        coefficient: np.ndarray,
        energy: np.ndarray,
        occupation: np.ndarray,
        occupied: np.ndarray,
        virtual: np.ndarray,
    ) -> tuple[np.ndarray, int, float, float, float, float]:
        """Build and audit a small unshifted occupied-virtual operator."""
        nocc = int(np.count_nonzero(occupied))
        nvir = int(np.count_nonzero(virtual))
        dimension = nocc * nvir
        if dimension > self.operator_dimension_limit:
            raise DeePHFCapabilityError(
                "the explicit RHF operator validation dimension exceeds its "
                f"debug limit: {dimension} > {self.operator_dimension_limit}"
            )
        matrix = np.empty((dimension, dimension), dtype=np.float64)
        batch_size = min(64, dimension)
        for start in range(0, dimension, batch_size):
            stop = min(start + batch_size, dimension)
            flat_roots = np.zeros((stop - start, dimension), dtype=np.float64)
            flat_roots[np.arange(stop - start), np.arange(start, stop)] = 1.0
            roots = flat_roots.reshape(-1, nvir, nocc)
            images = self._apply_occupied_virtual_operator(
                roots,
                coefficient,
                energy,
                occupation,
                occupied,
                virtual,
            )
            matrix[:, start:stop] = images.reshape(stop - start, dimension).T
        if not np.isfinite(matrix).all():
            raise RHFResponseError(
                "the RHF occupied-virtual response operator is nonfinite"
            )
        symmetry_residual = float(
            np.max(np.abs(matrix - matrix.T), initial=0.0)
        )
        if symmetry_residual > self.operator_symmetry_tolerance:
            raise RHFResponseError(
                "the RHF occupied-virtual response operator violates symmetry: "
                f"{symmetry_residual:.3e} > {self.operator_symmetry_tolerance:.3e}"
            )
        try:
            eigenvalues = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
        except np.linalg.LinAlgError as error:
            raise RHFResponseError(
                f"the RHF response-operator eigensolve failed: {error}"
            ) from error
        minimum_eigenvalue = float(eigenvalues[0])
        maximum_eigenvalue = float(eigenvalues[-1])
        if minimum_eigenvalue <= self.operator_stability_tolerance:
            raise DeePHFCapabilityError(
                "the RHF occupied-virtual response operator is unstable or singular: "
                f"minimum eigenvalue {minimum_eigenvalue:.3e} <= "
                f"{self.operator_stability_tolerance:.3e}"
            )
        condition_number = maximum_eigenvalue / minimum_eigenvalue
        if (
            not np.isfinite(condition_number)
            or condition_number > self.operator_condition_tolerance
        ):
            raise DeePHFCapabilityError(
                "the RHF occupied-virtual response operator is ill conditioned: "
                f"{condition_number:.3e} > {self.operator_condition_tolerance:.3e}"
            )
        return (
            matrix,
            dimension,
            minimum_eigenvalue,
            maximum_eigenvalue,
            float(condition_number),
            symmetry_residual,
        )

    def validate_response_operator_exact(self) -> tuple[int, float, float, float, float]:
        """Run an explicit dense stability audit for a bounded debug problem."""
        coefficient, energy, occupation, occupied, virtual, _gap = self._state()
        return self._response_operator_matrix_and_diagnostics(
            coefficient,
            energy,
            occupation,
            occupied,
            virtual,
        )[1:]

    def _induced_mo_potential(
        self,
        mo_response: np.ndarray,
        coefficient: np.ndarray,
        occupation: np.ndarray,
        occupied: np.ndarray,
    ) -> np.ndarray:
        density_response = self._density_from_mo_response(
            mo_response,
            coefficient,
            occupation,
            occupied,
        )
        induced = self._induced_potential(density_response)
        return np.einsum(
            "mp,...mn,ni->...pi",
            coefficient,
            induced,
            coefficient[:, occupied],
        )

    def _apply_occupied_virtual_operator(
        self,
        occupied_virtual_response: np.ndarray,
        coefficient: np.ndarray,
        energy: np.ndarray,
        occupation: np.ndarray,
        occupied: np.ndarray,
        virtual: np.ndarray,
    ) -> np.ndarray:
        """Apply the physical unshifted RHF operator to virtual-occupied amplitudes."""
        nocc = int(np.count_nonzero(occupied))
        nvir = int(np.count_nonzero(virtual))
        response = np.asarray(occupied_virtual_response)
        if response.shape[-2:] != (nvir, nocc):
            raise RHFResponseError(
                "the RHF occupied-virtual response has an invalid shape"
            )
        full_response = np.zeros(
            (*response.shape[:-2], coefficient.shape[1], nocc),
            dtype=np.float64,
        )
        full_response[..., virtual, :] = response
        induced = self._induced_mo_potential(
            full_response,
            coefficient,
            occupation,
            occupied,
        )[..., virtual, :]
        orbital_gap = energy[virtual, None] - energy[occupied]
        return orbital_gap * response + induced


class RHFResponseAdapter(_RHFLinearResponseCore):
    """Solve and audit molecular RHF nuclear CPHF through PySCF 2.14."""

    def coordinate_blocks(
        self,
        block_size: int,
        atom_indices=None,
        result_mode="response",
    ):
        """Yield audited responses while retaining at most one atom block."""
        block_size = _cycle_limit(block_size, "coordinate_block_size")
        if block_size <= 0:
            raise ValueError("coordinate_block_size must be positive")
        selected_atoms = self._response_atom_indices(atom_indices)
        for start in range(0, len(selected_atoms), block_size):
            block_atoms = selected_atoms[start : start + block_size]
            yield block_atoms, self._solve(block_atoms, result_mode)

    def _orbital_residual(
        self,
        mo_response: np.ndarray,
        hamiltonian_mo: np.ndarray,
        overlap_mo: np.ndarray,
        coefficient: np.ndarray,
        energy: np.ndarray,
        occupation: np.ndarray,
        occupied: np.ndarray,
        virtual: np.ndarray,
    ) -> np.ndarray:
        induced_mo = self._induced_mo_potential(
            mo_response,
            coefficient,
            occupation,
            occupied,
        )
        residual = (
            hamiltonian_mo
            + induced_mo
            - overlap_mo * energy[occupied]
            + (energy[:, None] - energy[occupied]) * mo_response
        )
        return residual[..., virtual, :]

    def audit_response_equations(self, response: RHFResponse) -> None:
        """Rebuild derivative inputs, equations, and invariants for a supplied response."""
        validate_reference(self.reference)
        if response.diagnostics.operator_is_self_adjoint is not True:
            raise RHFResponseError("the supplied RHF response operator contract is invalid")
        if response.diagnostics.pyscf_version != pyscf.__version__:
            raise RHFResponseError(
                "the supplied RHF response PySCF version does not match the runtime"
            )
        (
            coefficient,
            energy,
            occupation,
            occupied,
            virtual,
            minimum_gap,
        ) = self._state()
        response_dimension = int(np.count_nonzero(occupied)) * int(
            np.count_nonzero(virtual)
        )
        atom_indices = self._response_atom_indices(response.atom_indices)
        expected_overlap_derivative = self._overlap_derivative(atom_indices)
        expected_hamiltonian_derivative = self._hamiltonian_derivative(
            coefficient,
            occupation,
            atom_indices,
        )
        derivative_fields = (
            (
                response.overlap_derivative,
                expected_overlap_derivative,
                "overlap derivative",
            ),
            (
                response.hamiltonian_derivative,
                expected_hamiltonian_derivative,
                "Hamiltonian derivative",
            ),
        )
        for stored, expected, name in derivative_fields:
            if not np.allclose(stored, expected, rtol=0.0, atol=1.0e-12):
                raise RHFResponseError(
                    f"the supplied RHF response {name} does not match the reference"
                )
        occupied_coefficients = coefficient[:, occupied]
        hamiltonian_mo = np.einsum(
            "mp,...mn,ni->...pi",
            coefficient,
            expected_hamiltonian_derivative,
            occupied_coefficients,
        )
        overlap_mo = np.einsum(
            "mp,...mn,ni->...pi",
            coefficient,
            expected_overlap_derivative,
            occupied_coefficients,
        )
        residual = self._orbital_residual(
            response.mo_response,
            hamiltonian_mo,
            overlap_mo,
            coefficient,
            energy,
            occupation,
            occupied,
            virtual,
        )
        if not np.allclose(
            response.orbital_response_residual,
            residual,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise RHFResponseError(
                "the supplied RHF response orbital residual is not independently reproducible"
            )
        overlap = np.asarray(self.reference.get_ovlp())
        density_ground = np.asarray(self.reference.make_rdm1())
        density_response = response.density_response
        overlap_occupied = overlap_mo[..., occupied, :]
        metric_residual = float(
            np.max(
                np.abs(
                    response.mo_response[..., occupied, :]
                    + response.mo_response[..., occupied, :].swapaxes(-1, -2)
                    + overlap_occupied
                ),
                initial=0.0,
            )
        )
        idempotency = (
            np.einsum(
                "...ij,jk,kl->...il",
                density_response,
                overlap,
                density_ground,
            )
            + np.einsum(
                "ij,...jk,kl->...il",
                density_ground,
                expected_overlap_derivative,
                density_ground,
            )
            + np.einsum(
                "ij,jk,...kl->...il",
                density_ground,
                overlap,
                density_response,
            )
            - 2.0 * density_response
        )
        particle_number = (
            np.einsum("...ij,ji->...", density_response, overlap)
            + np.einsum(
                "ij,...ji->...",
                density_ground,
                expected_overlap_derivative,
            )
        )
        measured = {
            "minimum_orbital_gap": minimum_gap,
            "response_dimension": response_dimension,
            "maximum_residual": float(np.max(np.abs(residual), initial=0.0)),
            "residual_rms": float(np.sqrt(np.mean(np.square(residual)))),
            "metric_residual": metric_residual,
            "idempotency_residual": float(
                np.max(np.abs(idempotency), initial=0.0)
            ),
            "particle_number_residual": float(
                np.max(np.abs(particle_number), initial=0.0)
            ),
        }
        for name, value in measured.items():
            recorded = getattr(response.diagnostics, name)
            if isinstance(value, int):
                consistent = recorded == value
            else:
                consistent = np.isclose(
                    recorded,
                    value,
                    rtol=1.0e-10,
                    atol=1.0e-12,
                )
            if not consistent:
                raise RHFResponseError(
                    f"the supplied RHF response {name} diagnostic is inconsistent"
                )
        if measured["maximum_residual"] > self.residual_tolerance:
            raise RHFResponseError(
                "the supplied RHF response residual exceeds its tolerance"
            )
        invariant_values = (
            measured["metric_residual"],
            measured["idempotency_residual"],
            measured["particle_number_residual"],
        )
        if max(invariant_values) > self.invariant_tolerance:
            raise RHFResponseError(
                "the supplied RHF response invariant exceeds its tolerance"
            )

    def _solve_orbitals(
        self,
        hamiltonian_mo: np.ndarray,
        overlap_mo: np.ndarray,
        coefficient: np.ndarray,
        energy: np.ndarray,
        occupation: np.ndarray,
        occupied: np.ndarray,
        virtual: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, tuple[float, ...]]:
        perturbation_shape = hamiltonian_mo.shape[:-2]
        nmo = coefficient.shape[1]
        nocc = int(np.count_nonzero(occupied))
        flattened_hamiltonian = hamiltonian_mo.reshape(-1, nmo, nocc)
        flattened_overlap = overlap_mo.reshape(-1, nmo, nocc)

        def induced_full(response):
            response = np.asarray(response).reshape(-1, nmo, nocc)
            return self._induced_mo_potential(
                response,
                coefficient,
                occupation,
                occupied,
            )

        try:
            response, _ = cphf.solve(
                induced_full,
                energy,
                occupation,
                flattened_hamiltonian,
                flattened_overlap,
                max_cycle=self.max_cycle,
                tol=self.cphf_tolerance,
                level_shift=self.level_shift,
                verbose=self.reference.verbose,
            )
        except Exception as error:
            raise RHFResponseError(f"PySCF RHF CPHF solve failed: {error}") from error
        response = _validated_float64_array(
            response,
            flattened_hamiltonian.shape,
            "PySCF RHF CPHF response",
        ).reshape(*perturbation_shape, nmo, nocc)
        residual = self._orbital_residual(
            response,
            hamiltonian_mo,
            overlap_mo,
            coefficient,
            energy,
            occupation,
            occupied,
            virtual,
        )
        residual_history = [float(np.max(np.abs(residual), initial=0.0))]
        while (
            residual_history[-1] > self.residual_tolerance
            and len(residual_history) - 1 < self.max_refinement_cycles
        ):
            flat_residual = residual.reshape(-1, int(np.count_nonzero(virtual)), nocc)
            root_scales = np.linalg.norm(flat_residual.reshape(len(flat_residual), -1), axis=1)
            active = root_scales > np.finfo(float).eps
            correction = np.zeros_like(flat_residual)

            def induced_virtual(virtual_response):
                virtual_response = np.asarray(virtual_response).reshape(
                    -1,
                    int(np.count_nonzero(virtual)),
                    nocc,
                )
                full_response = np.zeros(
                    (len(virtual_response), nmo, nocc),
                    dtype=virtual_response.dtype,
                )
                full_response[:, virtual] = virtual_response
                return induced_full(full_response)[:, virtual]

            if np.any(active):
                try:
                    normalized_correction, _ = cphf.solve(
                        induced_virtual,
                        energy,
                        occupation,
                        flat_residual[active] / root_scales[active, None, None],
                        s1=None,
                        max_cycle=self.max_cycle,
                        tol=self.cphf_tolerance,
                        level_shift=self.level_shift,
                        verbose=self.reference.verbose,
                    )
                except Exception as error:
                    raise RHFResponseError(
                        f"PySCF RHF CPHF residual refinement failed: {error}"
                    ) from error
                normalized_correction = _validated_float64_array(
                    normalized_correction,
                    flat_residual[active].shape,
                    "PySCF RHF CPHF refinement response",
                )
                correction[active] = (
                    normalized_correction * root_scales[active, None, None]
                )
                response[..., virtual, :] += correction.reshape(
                    *perturbation_shape,
                    int(np.count_nonzero(virtual)),
                    nocc,
                )
            residual = self._orbital_residual(
                response,
                hamiltonian_mo,
                overlap_mo,
                coefficient,
                energy,
                occupation,
                occupied,
                virtual,
            )
            residual_history.append(
                float(np.max(np.abs(residual), initial=0.0))
            )
        return response, residual, tuple(residual_history)

    def solve(self, atom_indices=None) -> RHFResponse:
        """Return an audited first-order response for selected atoms."""
        return self._solve(atom_indices, "response")

    def _solve_with_density_partitions(self, atom_indices=None):
        """Return a response and its transient AO density work arrays."""
        return self._solve(atom_indices, "partitions")

    def _solve_for_gradient(self, atom_indices=None):
        """Return compact diagnostics and transient AO density work arrays."""
        return self._solve(atom_indices, "gradient")

    def _solve(self, atom_indices, result_mode):
        validate_reference(self.reference)
        atom_indices = self._response_atom_indices(atom_indices)
        (
            coefficient,
            energy,
            occupation,
            occupied,
            virtual,
            minimum_gap,
        ) = self._state()
        response_dimension = int(np.count_nonzero(occupied)) * int(
            np.count_nonzero(virtual)
        )
        overlap = np.asarray(self.reference.get_ovlp())
        overlap_derivative = self._overlap_derivative(atom_indices)
        hamiltonian_derivative = self._hamiltonian_derivative(
            coefficient,
            occupation,
            atom_indices,
        )
        occupied_coefficients = coefficient[:, occupied]
        hamiltonian_mo = np.einsum(
            "mp,...mn,ni->...pi",
            coefficient,
            hamiltonian_derivative,
            occupied_coefficients,
        )
        overlap_mo = np.einsum(
            "mp,...mn,ni->...pi",
            coefficient,
            overlap_derivative,
            occupied_coefficients,
        )
        mo_response, residual, residual_history = self._solve_orbitals(
            hamiltonian_mo,
            overlap_mo,
            coefficient,
            energy,
            occupation,
            occupied,
            virtual,
        )
        metric_response = np.zeros_like(mo_response)
        metric_response[..., occupied, :] = mo_response[..., occupied, :]
        occupied_virtual_response = np.zeros_like(mo_response)
        occupied_virtual_response[..., virtual, :] = mo_response[..., virtual, :]
        density_metric = self._density_from_mo_response(
            metric_response,
            coefficient,
            occupation,
            occupied,
        )
        density_occupied_virtual = self._density_from_mo_response(
            occupied_virtual_response,
            coefficient,
            occupation,
            occupied,
        )
        density_response = density_metric + density_occupied_virtual
        density_ground = self.reference.make_rdm1()
        overlap_occupied = overlap_mo[..., occupied, :]
        metric_residual = np.max(
            np.abs(
                mo_response[..., occupied, :]
                + mo_response[..., occupied, :].swapaxes(-1, -2)
                + overlap_occupied
            ),
            initial=0.0,
        )
        idempotency = (
            np.einsum("...ij,jk,kl->...il", density_response, overlap, density_ground)
            + np.einsum(
                "ij,...jk,kl->...il",
                density_ground,
                overlap_derivative,
                density_ground,
            )
            + np.einsum("ij,jk,...kl->...il", density_ground, overlap, density_response)
            - 2.0 * density_response
        )
        particle_number = (
            np.einsum("...ij,ji->...", density_response, overlap)
            + np.einsum("ij,...ji->...", density_ground, overlap_derivative)
        )
        diagnostics = RHFResponseDiagnostics(
            minimum_orbital_gap=minimum_gap,
            pyscf_version=pyscf.__version__,
            cphf_tolerance=self.cphf_tolerance,
            maximum_residual=float(np.max(np.abs(residual), initial=0.0)),
            residual_rms=float(np.sqrt(np.mean(np.square(residual)))),
            residual_tolerance=self.residual_tolerance,
            invariant_tolerance=self.invariant_tolerance,
            orbital_gap_tolerance=self.orbital_gap_tolerance,
            max_cycle=self.max_cycle,
            max_refinement_cycles=self.max_refinement_cycles,
            level_shift=self.level_shift,
            response_dimension=response_dimension,
            operator_is_self_adjoint=True,
            metric_residual=float(metric_residual),
            idempotency_residual=float(np.max(np.abs(idempotency), initial=0.0)),
            particle_number_residual=float(
                np.max(np.abs(particle_number), initial=0.0)
            ),
            refinement_cycles=len(residual_history) - 1,
            residual_history=residual_history,
        )
        arrays = {
            "mo_response": mo_response,
            "mo_response_occupied_virtual": occupied_virtual_response,
            "mo_response_metric": metric_response,
            "density_response": density_response,
            "density_response_occupied_virtual": density_occupied_virtual,
            "density_response_metric": density_metric,
            "overlap_derivative": overlap_derivative,
            "hamiltonian_derivative": hamiltonian_derivative,
            "orbital_response_residual": residual,
        }
        nonfinite = [name for name, value in arrays.items() if not np.isfinite(value).all()]
        if nonfinite:
            raise RHFResponseError(
                f"nonfinite RHF response quantities: {', '.join(nonfinite)}"
            )
        diagnostic_values = (
            diagnostics.minimum_orbital_gap,
            diagnostics.maximum_residual,
            diagnostics.residual_rms,
            diagnostics.metric_residual,
            diagnostics.idempotency_residual,
            diagnostics.particle_number_residual,
            *diagnostics.residual_history,
        )
        if not np.isfinite(diagnostic_values).all():
            raise RHFResponseError("nonfinite RHF response diagnostics")
        if diagnostics.maximum_residual > self.residual_tolerance:
            history = " -> ".join(
                f"{value:.3e}" for value in diagnostics.residual_history
            )
            raise RHFResponseError(
                "RHF response residual exceeds tolerance: "
                f"{diagnostics.maximum_residual:.3e} > {self.residual_tolerance:.3e}; "
                f"refinement history: {history}"
            )
        invariant_failures = {
            "metric": diagnostics.metric_residual,
            "idempotency": diagnostics.idempotency_residual,
            "particle number": diagnostics.particle_number_residual,
        }
        invariant_failures = {
            name: value
            for name, value in invariant_failures.items()
            if value > self.invariant_tolerance
        }
        if invariant_failures:
            details = ", ".join(
                f"{name}={value:.3e}" for name, value in invariant_failures.items()
            )
            raise RHFResponseError(
                "RHF response invariant exceeds tolerance "
                f"{self.invariant_tolerance:.3e}: {details}"
            )
        density_partitions = (
            density_response,
            density_metric,
            density_occupied_virtual,
        )
        if result_mode == "gradient":
            return diagnostics, density_partitions
        response = RHFResponse(
            reference_identity=id(self.reference),
            state_fingerprint=reference_fingerprint(self.reference),
            integrity_fingerprint="",
            atom_indices=atom_indices,
            mo_response=_immutable_array(mo_response),
            _mo_coefficients=_immutable_array(coefficient),
            _mo_occupations=_immutable_array(occupation),
            overlap_derivative=_immutable_array(overlap_derivative),
            hamiltonian_derivative=_immutable_array(
                hamiltonian_derivative
            ),
            orbital_response_residual=_immutable_array(residual),
            diagnostics=diagnostics,
        )
        response = replace(
            response,
            integrity_fingerprint=response_integrity_fingerprint(response),
        )
        return (response, density_partitions) if result_mode == "partitions" else response


class _RHFScalarAdjointProblem:
    """Bind the RHF occupied-virtual action to the adjoint protocol."""

    is_self_adjoint = True

    def __init__(
        self,
        adapter: "RHFAdjointAdapter",
        coefficient: np.ndarray,
        energy: np.ndarray,
        occupation: np.ndarray,
        occupied: np.ndarray,
        virtual: np.ndarray,
    ):
        self._adapter = adapter
        self._coefficient = coefficient
        self._energy = energy
        self._occupation = occupation
        self._occupied = occupied
        self._virtual = virtual
        self._nocc = int(np.count_nonzero(occupied))
        self._nvir = int(np.count_nonzero(virtual))

    @property
    def dimension(self) -> int:
        return self._nocc * self._nvir

    @property
    def operator_fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"pyscf-2.14-rhf-occupied-virtual-operator-v1")
        for value in (
            self._coefficient,
            self._energy,
            self._occupation,
            self._occupied,
            self._virtual,
        ):
            array = np.ascontiguousarray(value)
            digest.update(array.dtype.str.encode("ascii"))
            digest.update(repr(array.shape).encode("ascii"))
            digest.update(array.tobytes())
        return digest.hexdigest()

    def apply(self, vector: np.ndarray) -> np.ndarray:
        amplitudes = np.asarray(vector).reshape(self._nvir, self._nocc)
        image = self._adapter._apply_occupied_virtual_operator(
            amplitudes,
            self._coefficient,
            self._energy,
            self._occupation,
            self._occupied,
            self._virtual,
        )
        return np.asarray(image).reshape(self.dimension)

    def apply_transpose(self, vector: np.ndarray) -> np.ndarray:
        # The real closed-shell orbital Hessian is symmetric in this space.
        return self.apply(vector)

    def precondition(self, vector: np.ndarray) -> np.ndarray:
        gaps = (
            self._energy[self._virtual, None]
            - self._energy[self._occupied]
        )
        return np.asarray(vector).reshape(self.dimension) / gaps.reshape(
            self.dimension
        )


class RHFAdjointAdapter(_RHFLinearResponseCore):
    """Solve one correction-specific RHF Z-vector through PySCF 2.14."""

    def __init__(
        self,
        reference,
        *,
        residual_tolerance: float = 1.0e-9,
        orbital_gap_tolerance: float = 1.0e-7,
        operator_stability_tolerance: float = 1.0e-6,
        operator_condition_tolerance: float = 1.0e8,
        operator_symmetry_tolerance: float = 1.0e-10,
        operator_dimension_limit: int = 512,
        objective_symmetry_tolerance: float = 1.0e-10,
        max_cycle: int = 100,
        krylov_restart: int = 50,
    ):
        validate_pyscf_version()
        self.reference = validate_reference(reference)
        self.residual_tolerance = _adjoint_real_control(
            residual_tolerance,
            "residual_tolerance",
        )
        self.orbital_gap_tolerance = _adjoint_real_control(
            orbital_gap_tolerance,
            "orbital_gap_tolerance",
        )
        self.operator_stability_tolerance = _adjoint_real_control(
            operator_stability_tolerance,
            "operator_stability_tolerance",
        )
        self.operator_condition_tolerance = _adjoint_real_control(
            operator_condition_tolerance,
            "operator_condition_tolerance",
        )
        self.operator_symmetry_tolerance = _adjoint_real_control(
            operator_symmetry_tolerance,
            "operator_symmetry_tolerance",
        )
        self.operator_dimension_limit = _cycle_limit(
            operator_dimension_limit,
            "operator_dimension_limit",
        )
        self.objective_symmetry_tolerance = _adjoint_real_control(
            objective_symmetry_tolerance,
            "objective_symmetry_tolerance",
        )
        self.max_cycle = _cycle_limit(max_cycle, "max_cycle")
        self.krylov_restart = _cycle_limit(
            krylov_restart,
            "krylov_restart",
        )
        if (
            self.residual_tolerance <= 0
            or self.orbital_gap_tolerance <= 0
            or self.operator_stability_tolerance <= 0
            or self.operator_condition_tolerance <= 1
            or self.operator_symmetry_tolerance <= 0
            or self.objective_symmetry_tolerance <= 0
        ):
            raise ValueError("adjoint tolerances are invalid")
        if self.operator_dimension_limit <= 0:
            raise ValueError("adjoint operator_dimension_limit must be positive")
        if self.max_cycle <= 0 or self.krylov_restart <= 0:
            raise ValueError("adjoint Krylov cycle limits must be positive")

    def _validated_objective_potential(self, value) -> np.ndarray:
        potential = _validated_float64_array(
            value,
            (self.molecule.nao, self.molecule.nao),
            "correction AO objective potential",
        )
        symmetry_residual = float(
            np.max(np.abs(potential - potential.T), initial=0.0)
        )
        if symmetry_residual > self.objective_symmetry_tolerance:
            raise RHFAdjointError(
                "the correction AO objective potential violates symmetry: "
                f"{symmetry_residual:.3e} > "
                f"{self.objective_symmetry_tolerance:.3e}"
            )
        return potential

    @staticmethod
    def _audited_array(value, expected_shape, name: str) -> np.ndarray:
        if type(value) is not np.ndarray:
            raise RHFAdjointError(
                f"the supplied RHF adjoint field {name} has an invalid type"
            )
        if value.shape != expected_shape:
            raise RHFAdjointError(
                f"the supplied RHF adjoint field {name} has shape {value.shape}; "
                f"expected {expected_shape}"
            )
        if value.dtype != np.dtype(np.float64) or np.iscomplexobj(value):
            raise RHFAdjointError(
                f"the supplied RHF adjoint field {name} must use real numpy.float64"
            )
        if not np.isfinite(value).all():
            raise RHFAdjointError(
                f"the supplied RHF adjoint field {name} must be finite"
            )
        if value.flags.writeable:
            raise RHFAdjointError(
                f"the supplied RHF adjoint field {name} must be immutable"
            )
        return value

    @staticmethod
    def _require_close(stored, expected, name: str) -> None:
        if not np.allclose(stored, expected, rtol=1.0e-11, atol=1.0e-12):
            raise RHFAdjointError(
                f"the supplied RHF adjoint {name} is inconsistent"
            )

    def audit_adjoint(
        self,
        adjoint: RHFAdjoint,
        expected_objective_ao_potential: np.ndarray,
    ) -> None:
        """Independently audit one consumed RHF adjoint without another solve."""
        validate_reference(self.reference)
        if type(adjoint) is not RHFAdjoint:
            raise RHFAdjointError("the supplied RHF adjoint has an invalid type")
        diagnostics = adjoint.diagnostics
        if type(diagnostics) is not RHFAdjointDiagnostics:
            raise RHFAdjointError(
                "the supplied RHF adjoint diagnostics have an invalid type"
            )
        if adjoint.reference_identity != id(self.reference):
            raise RHFAdjointError(
                "the supplied RHF adjoint belongs to another reference"
            )
        if adjoint.state_fingerprint != reference_fingerprint(self.reference):
            raise RHFAdjointError(
                "the supplied RHF adjoint does not match the current RHF state"
            )
        if adjoint.integrity_fingerprint != adjoint_integrity_fingerprint(adjoint):
            raise RHFAdjointError(
                "the supplied RHF adjoint failed its integrity check"
            )
        if (
            type(adjoint.reference_identity) is not int
            or type(adjoint.state_fingerprint) is not str
            or type(adjoint.integrity_fingerprint) is not str
            or type(adjoint.operator_fingerprint) is not str
        ):
            raise RHFAdjointError(
                "the supplied RHF adjoint provenance fields have invalid types"
            )
        if diagnostics.pyscf_version != pyscf.__version__:
            raise RHFAdjointError(
                "the supplied RHF adjoint PySCF version does not match the runtime"
            )
        if diagnostics.solver != "scipy.sparse.linalg.gmres(A.T, b)":
            raise RHFAdjointError(
                "the supplied RHF adjoint solver convention is invalid"
            )
        if type(diagnostics.solve_count) is not int or diagnostics.solve_count != 1:
            raise RHFAdjointError(
                "the supplied RHF adjoint must contain exactly one scalar solve"
            )
        if type(diagnostics.response_dimension) is not int:
            raise RHFAdjointError(
                "the supplied RHF adjoint response dimension has an invalid type"
            )
        if diagnostics.operator_is_self_adjoint is not True:
            raise RHFAdjointError("the supplied RHF adjoint operator contract is invalid")
        if (
            type(diagnostics.max_cycle) is not int
            or type(diagnostics.krylov_restart) is not int
            or type(diagnostics.iteration_count) is not int
        ):
            raise RHFAdjointError(
                "the supplied RHF adjoint Krylov diagnostics have invalid types"
            )
        diagnostic_reals = (
            diagnostics.minimum_orbital_gap,
            diagnostics.residual_tolerance,
            diagnostics.orbital_gap_tolerance,
            diagnostics.objective_symmetry_tolerance,
            diagnostics.objective_symmetry_residual,
            diagnostics.adjoint_density_symmetry_residual,
            diagnostics.adjoint_potential_symmetry_residual,
            diagnostics.objective_gradient_norm,
            diagnostics.solution_norm,
            diagnostics.maximum_residual,
            diagnostics.residual_rms,
        )
        if any(
            isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            for value in diagnostic_reals
        ) or not np.isfinite(diagnostic_reals).all():
            raise RHFAdjointError(
                "the supplied RHF adjoint diagnostics must be finite real scalars"
            )
        if (
            diagnostics.residual_tolerance <= 0
            or diagnostics.orbital_gap_tolerance <= 0
            or diagnostics.objective_symmetry_tolerance <= 0
            or diagnostics.response_dimension <= 0
            or diagnostics.max_cycle <= 0
            or diagnostics.krylov_restart <= 0
            or diagnostics.iteration_count < 0
        ):
            raise RHFAdjointError(
                "the supplied RHF adjoint controls are invalid"
            )
        accepted_controls = {
            "residual_tolerance": self.residual_tolerance,
            "orbital_gap_tolerance": self.orbital_gap_tolerance,
            "objective_symmetry_tolerance": (
                self.objective_symmetry_tolerance
            ),
            "max_cycle": self.max_cycle,
            "krylov_restart": self.krylov_restart,
        }
        for name, expected in accepted_controls.items():
            if getattr(diagnostics, name) != expected:
                raise RHFAdjointError(
                    f"the supplied RHF adjoint {name} control is inconsistent"
                )
        (
            coefficient,
            energy,
            occupation,
            occupied,
            virtual,
            minimum_gap,
        ) = self._state()
        nocc = int(np.count_nonzero(occupied))
        nvir = int(np.count_nonzero(virtual))
        dimension = nocc * nvir
        atom_indices = self._response_atom_indices(adjoint.atom_indices)
        if atom_indices != adjoint.atom_indices:
            raise RHFAdjointError("the supplied RHF adjoint atom selection is invalid")
        natm = len(atom_indices)
        nao = int(self.molecule.nao)
        arrays = {
            "objective_ao_potential": (nao, nao),
            "objective_orbital_gradient": (nvir, nocc),
            "zvector": (nvir, nocc),
            "residual": (nvir, nocc),
            "adjoint_ao_density": (nao, nao),
            "adjoint_ao_potential": (nao, nao),
            "correction_gradient_metric": (natm, 3),
            "correction_gradient_adjoint_nuclear": (natm, 3),
            "correction_gradient_adjoint_metric": (natm, 3),
            "correction_gradient_occupied_virtual": (natm, 3),
            "correction_gradient_response": (natm, 3),
        }
        for name, shape in arrays.items():
            self._audited_array(getattr(adjoint, name), shape, name)
        expected_objective_ao_potential = _validated_float64_array(
            expected_objective_ao_potential,
            (nao, nao),
            "expected correction AO objective potential",
        )
        self._require_close(
            adjoint.objective_ao_potential,
            expected_objective_ao_potential,
            "objective AO potential",
        )
        objective_mo = (
            coefficient.T @ expected_objective_ao_potential @ coefficient
        )
        expected_objective_gradient = (
            objective_mo[virtual][:, occupied]
            + objective_mo.T[virtual][:, occupied]
        ) * occupation[occupied]
        self._require_close(
            adjoint.objective_orbital_gradient,
            expected_objective_gradient,
            "bilateral occupied-virtual objective gradient",
        )
        response_dimension = dimension
        problem = _RHFScalarAdjointProblem(
            self,
            coefficient,
            energy,
            occupation,
            occupied,
            virtual,
        )
        expected_operator_fingerprint = scalar_operator_fingerprint(
            problem,
            solver="gmres",
        )
        if adjoint.operator_fingerprint != expected_operator_fingerprint:
            raise RHFAdjointError(
                "the supplied RHF adjoint response operator is inconsistent"
            )
        zvector = adjoint.zvector
        objective_vector = expected_objective_gradient.reshape(dimension)
        residual = (
            self._apply_occupied_virtual_operator(
                zvector,
                coefficient,
                energy,
                occupation,
                occupied,
                virtual,
            ).reshape(dimension)
            - objective_vector
        ).reshape(nvir, nocc)
        self._require_close(
            adjoint.residual,
            residual,
            "independent residual",
        )
        occupied_coefficients = coefficient[:, occupied]
        virtual_coefficients = coefficient[:, virtual]
        rotated_occupied = virtual_coefficients @ zvector
        expected_adjoint_density = (
            rotated_occupied
            @ (occupied_coefficients * occupation[occupied]).T
        )
        expected_adjoint_density = (
            expected_adjoint_density + expected_adjoint_density.T
        )
        self._require_close(
            adjoint.adjoint_ao_density,
            expected_adjoint_density,
            "AO density",
        )
        expected_adjoint_potential = self._induced_potential(
            expected_adjoint_density
        )
        self._require_close(
            adjoint.adjoint_ao_potential,
            expected_adjoint_potential,
            "AO potential",
        )
        overlap_derivative = self._overlap_derivative(atom_indices)
        hamiltonian_derivative = self._hamiltonian_derivative(
            coefficient,
            occupation,
            atom_indices,
        )
        overlap_mo = np.einsum(
            "mp,...mn,ni->...pi",
            coefficient,
            overlap_derivative,
            occupied_coefficients,
        )
        hamiltonian_mo = np.einsum(
            "mp,...mn,ni->...pi",
            coefficient,
            hamiltonian_derivative,
            occupied_coefficients,
        )
        bare_nuclear_rhs = (
            hamiltonian_mo[..., virtual, :]
            - overlap_mo[..., virtual, :] * energy[occupied]
        )
        expected_adjoint_nuclear = -np.einsum(
            "ai,...ai->...",
            zvector,
            bare_nuclear_rhs,
        )
        objective_occupied = objective_mo[occupied][:, occupied]
        objective_occupied = 0.5 * (
            objective_occupied + objective_occupied.T
        )
        adjoint_potential_mo = (
            coefficient.T @ expected_adjoint_potential @ coefficient
        )
        adjoint_potential_occupied = adjoint_potential_mo[occupied][
            :, occupied
        ]
        adjoint_potential_occupied = 0.5 * (
            adjoint_potential_occupied + adjoint_potential_occupied.T
        )
        overlap_occupied = overlap_mo[..., occupied, :]
        expected_metric = np.einsum(
            "...ij,ij->...",
            overlap_occupied,
            -2.0 * objective_occupied,
        )
        expected_adjoint_metric = np.einsum(
            "...ij,ij->...",
            overlap_occupied,
            0.5 * adjoint_potential_occupied,
        )
        expected_occupied_virtual = (
            expected_adjoint_nuclear + expected_adjoint_metric
        )
        expected_response = expected_metric + expected_occupied_virtual
        expected_gradients = {
            "correction_gradient_metric": expected_metric,
            "correction_gradient_adjoint_nuclear": expected_adjoint_nuclear,
            "correction_gradient_adjoint_metric": expected_adjoint_metric,
            "correction_gradient_occupied_virtual": expected_occupied_virtual,
            "correction_gradient_response": expected_response,
        }
        for name, expected in expected_gradients.items():
            self._require_close(
                getattr(adjoint, name),
                expected,
                name,
            )
        self._require_close(
            adjoint.correction_gradient_occupied_virtual,
            adjoint.correction_gradient_adjoint_nuclear
            + adjoint.correction_gradient_adjoint_metric,
            "occupied-virtual gradient partition",
        )
        self._require_close(
            adjoint.correction_gradient_response,
            adjoint.correction_gradient_metric
            + adjoint.correction_gradient_occupied_virtual,
            "response gradient partition",
        )

        def residual_statistics(value):
            return (
                float(np.max(np.abs(value), initial=0.0)),
                float(np.sqrt(np.mean(np.square(value)))),
            )

        maximum_residual, residual_rms = residual_statistics(residual)
        measured = {
            "minimum_orbital_gap": minimum_gap,
            "response_dimension": response_dimension,
            "objective_symmetry_residual": float(
                np.max(
                    np.abs(
                        expected_objective_ao_potential
                        - expected_objective_ao_potential.T
                    ),
                    initial=0.0,
                )
            ),
            "adjoint_density_symmetry_residual": float(
                np.max(
                    np.abs(
                        expected_adjoint_density - expected_adjoint_density.T
                    ),
                    initial=0.0,
                )
            ),
            "adjoint_potential_symmetry_residual": float(
                np.max(
                    np.abs(
                        expected_adjoint_potential
                        - expected_adjoint_potential.T
                    ),
                    initial=0.0,
                )
            ),
            "objective_gradient_norm": float(
                np.linalg.norm(expected_objective_gradient)
            ),
            "solution_norm": float(np.linalg.norm(zvector)),
            "maximum_residual": maximum_residual,
            "residual_rms": residual_rms,
        }
        for name, expected in measured.items():
            stored = getattr(diagnostics, name)
            if isinstance(expected, int):
                matches = stored == expected
            else:
                matches = np.isclose(
                    stored,
                    expected,
                    rtol=1.0e-10,
                    atol=1.0e-12,
                )
            if not matches:
                raise RHFAdjointError(
                    f"the supplied RHF adjoint {name} diagnostic is inconsistent"
                )
        if (
            maximum_residual > diagnostics.residual_tolerance
            or minimum_gap <= diagnostics.orbital_gap_tolerance
            or measured["objective_symmetry_residual"]
            > diagnostics.objective_symmetry_tolerance
            or measured["adjoint_density_symmetry_residual"]
            > diagnostics.objective_symmetry_tolerance
            or measured["adjoint_potential_symmetry_residual"]
            > diagnostics.objective_symmetry_tolerance
        ):
            raise RHFAdjointError(
                "the supplied RHF adjoint exceeds an accepted control"
            )
        validate_reference(self.reference)

    def solve(self, objective_ao_potential: np.ndarray, atom_indices=None) -> RHFAdjoint:
        """Return one audited Z-vector and selected nuclear contractions."""
        try:
            return self._solve(objective_ao_potential, atom_indices)
        except DeePHFCapabilityError:
            raise
        except RHFAdjointError:
            raise
        except (AdjointError, RHFResponseError) as error:
            raise RHFAdjointError(f"RHF adjoint evaluation failed: {error}") from error

    def _solve(self, objective_ao_potential: np.ndarray, atom_indices=None) -> RHFAdjoint:
        validate_reference(self.reference)
        atom_indices = self._response_atom_indices(atom_indices)
        objective_ao_potential = self._validated_objective_potential(
            objective_ao_potential
        )
        objective_symmetry_residual = float(
            np.max(
                np.abs(objective_ao_potential - objective_ao_potential.T),
                initial=0.0,
            )
        )
        (
            coefficient,
            energy,
            occupation,
            occupied,
            virtual,
            minimum_gap,
        ) = self._state()
        response_dimension = int(np.count_nonzero(occupied)) * int(
            np.count_nonzero(virtual)
        )
        occupied_coefficients = coefficient[:, occupied]
        virtual_coefficients = coefficient[:, virtual]
        objective_mo = coefficient.T @ objective_ao_potential @ coefficient
        objective_orbital_gradient = (
            objective_mo[virtual][:, occupied]
            + objective_mo.T[virtual][:, occupied]
        ) * occupation[occupied]
        objective_orbital_gradient = _validated_float64_array(
            objective_orbital_gradient,
            (
                int(np.count_nonzero(virtual)),
                int(np.count_nonzero(occupied)),
            ),
            "correction occupied-virtual objective gradient",
        )
        problem = _RHFScalarAdjointProblem(
            self,
            coefficient,
            energy,
            occupation,
            occupied,
            virtual,
        )
        linear_result = solve_scalar_adjoint(
            problem,
            objective_orbital_gradient.reshape(response_dimension),
            residual_tolerance=self.residual_tolerance,
            require_physical_residual=True,
            solver="gmres",
            max_cycle=self.max_cycle,
            restart=self.krylov_restart,
        )
        nocc = int(np.count_nonzero(occupied))
        nvir = int(np.count_nonzero(virtual))
        zvector = linear_result.solution.reshape(nvir, nocc)
        rotated_occupied_coefficients = virtual_coefficients @ zvector
        adjoint_ao_density = (
            rotated_occupied_coefficients
            @ (occupied_coefficients * occupation[occupied]).T
        )
        adjoint_ao_density = (
            adjoint_ao_density + adjoint_ao_density.T
        )
        adjoint_ao_density = _validated_float64_array(
            adjoint_ao_density,
            (self.molecule.nao, self.molecule.nao),
            "RHF adjoint AO density",
        )
        adjoint_density_symmetry_residual = float(
            np.max(
                np.abs(adjoint_ao_density - adjoint_ao_density.T),
                initial=0.0,
            )
        )
        if adjoint_density_symmetry_residual > self.objective_symmetry_tolerance:
            raise RHFAdjointError(
                "the RHF adjoint AO density violates symmetry: "
                f"{adjoint_density_symmetry_residual:.3e} > "
                f"{self.objective_symmetry_tolerance:.3e}"
            )
        adjoint_ao_potential = self._induced_potential(adjoint_ao_density)
        adjoint_ao_potential = _validated_float64_array(
            adjoint_ao_potential,
            (self.molecule.nao, self.molecule.nao),
            "RHF adjoint AO potential",
        )
        adjoint_potential_symmetry_residual = float(
            np.max(
                np.abs(adjoint_ao_potential - adjoint_ao_potential.T),
                initial=0.0,
            )
        )
        if adjoint_potential_symmetry_residual > self.objective_symmetry_tolerance:
            raise RHFAdjointError(
                "the RHF adjoint AO potential violates symmetry: "
                f"{adjoint_potential_symmetry_residual:.3e} > "
                f"{self.objective_symmetry_tolerance:.3e}"
            )
        overlap_derivative = self._overlap_derivative(atom_indices)
        hamiltonian_derivative = self._hamiltonian_derivative(
            coefficient,
            occupation,
            atom_indices,
        )
        overlap_mo = np.einsum(
            "mp,...mn,ni->...pi",
            coefficient,
            overlap_derivative,
            occupied_coefficients,
        )
        hamiltonian_mo = np.einsum(
            "mp,...mn,ni->...pi",
            coefficient,
            hamiltonian_derivative,
            occupied_coefficients,
        )
        bare_nuclear_rhs = (
            hamiltonian_mo[..., virtual, :]
            - overlap_mo[..., virtual, :] * energy[occupied]
        )
        correction_gradient_adjoint_nuclear = -np.einsum(
            "ai,...ai->...",
            zvector,
            bare_nuclear_rhs,
        )
        objective_occupied = objective_mo[occupied][:, occupied]
        objective_occupied = 0.5 * (
            objective_occupied + objective_occupied.T
        )
        adjoint_potential_mo = (
            coefficient.T @ adjoint_ao_potential @ coefficient
        )
        adjoint_potential_occupied = adjoint_potential_mo[occupied][
            :, occupied
        ]
        adjoint_potential_occupied = 0.5 * (
            adjoint_potential_occupied + adjoint_potential_occupied.T
        )
        overlap_occupied = overlap_mo[..., occupied, :]
        correction_gradient_metric = np.einsum(
            "...ij,ij->...",
            overlap_occupied,
            -2.0 * objective_occupied,
        )
        correction_gradient_adjoint_metric = np.einsum(
            "...ij,ij->...",
            overlap_occupied,
            0.5 * adjoint_potential_occupied,
        )
        correction_gradient_occupied_virtual = (
            correction_gradient_adjoint_nuclear
            + correction_gradient_adjoint_metric
        )
        correction_gradient_response = (
            correction_gradient_metric
            + correction_gradient_occupied_virtual
        )
        gradient_fields = {
            "RHF objective metric gradient": correction_gradient_metric,
            "RHF adjoint nuclear gradient": (
                correction_gradient_adjoint_nuclear
            ),
            "RHF adjoint metric gradient": correction_gradient_adjoint_metric,
            "RHF occupied-virtual gradient": (
                correction_gradient_occupied_virtual
            ),
            "RHF adjoint response gradient": correction_gradient_response,
        }
        for name, value in gradient_fields.items():
            _validated_float64_array(
                value,
                (len(atom_indices), 3),
                name,
            )
        validate_reference(self.reference)
        state_fingerprint = reference_fingerprint(self.reference)
        linear_diagnostics = linear_result.diagnostics
        diagnostics = RHFAdjointDiagnostics(
            minimum_orbital_gap=minimum_gap,
            pyscf_version=pyscf.__version__,
            residual_tolerance=self.residual_tolerance,
            orbital_gap_tolerance=self.orbital_gap_tolerance,
            response_dimension=response_dimension,
            operator_is_self_adjoint=True,
            objective_symmetry_tolerance=self.objective_symmetry_tolerance,
            objective_symmetry_residual=objective_symmetry_residual,
            adjoint_density_symmetry_residual=(
                adjoint_density_symmetry_residual
            ),
            adjoint_potential_symmetry_residual=(
                adjoint_potential_symmetry_residual
            ),
            solver=linear_diagnostics.solver,
            solve_count=linear_diagnostics.solve_count,
            objective_gradient_norm=(
                linear_diagnostics.objective_gradient_norm
            ),
            solution_norm=linear_diagnostics.solution_norm,
            maximum_residual=linear_diagnostics.maximum_residual,
            residual_rms=linear_diagnostics.residual_rms,
            max_cycle=self.max_cycle,
            krylov_restart=self.krylov_restart,
            iteration_count=linear_diagnostics.iteration_count,
        )
        adjoint = RHFAdjoint(
            reference_identity=id(self.reference),
            state_fingerprint=state_fingerprint,
            integrity_fingerprint="",
            operator_fingerprint=linear_result.operator_fingerprint,
            atom_indices=atom_indices,
            objective_ao_potential=_immutable_array(objective_ao_potential),
            objective_orbital_gradient=_immutable_array(
                objective_orbital_gradient
            ),
            zvector=_immutable_array(zvector),
            residual=_immutable_array(
                linear_result.residual.reshape(nvir, nocc)
            ),
            adjoint_ao_density=_immutable_array(adjoint_ao_density),
            adjoint_ao_potential=_immutable_array(adjoint_ao_potential),
            correction_gradient_metric=_immutable_array(
                correction_gradient_metric
            ),
            correction_gradient_adjoint_nuclear=_immutable_array(
                correction_gradient_adjoint_nuclear
            ),
            correction_gradient_adjoint_metric=_immutable_array(
                correction_gradient_adjoint_metric
            ),
            correction_gradient_occupied_virtual=_immutable_array(
                correction_gradient_occupied_virtual
            ),
            correction_gradient_response=_immutable_array(
                correction_gradient_response
            ),
            diagnostics=diagnostics,
        )
        return replace(
            adjoint,
            integrity_fingerprint=adjoint_integrity_fingerprint(adjoint),
        )
