"""Isolated PySCF 2.14 adapter for molecular RHF nuclear response."""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field, fields
import hashlib
import operator
from types import MappingProxyType
from typing import Any

import numpy as np
import pyscf
from pyscf import gto
from pyscf.gto import mole as gto_mole
from pyscf.scf import hf as scf_hf


from .adjoint import AdjointError
from .capabilities import DeePHFCapabilityError, transaction_reference_fingerprint
from .contracts import (
    immutable_array as _immutable_array,
    update_digest as _update_fingerprint_value,
    version_series as _canonical_version_series,
)


SUPPORTED_PYSCF_SERIES = (2, 14)


class RHFResponseError(RuntimeError):
    """Raised when the RHF response equations fail the strict contract."""


class RHFAdjointError(AdjointError):
    """Raised when the RHF scalar adjoint fails the strict contract."""


class RHFScannerReferenceError(RuntimeError):
    """Raised when a fresh scanner reference violates its strict contract."""


def validate_reference(reference):
    from .audits.restricted_reference import validate_reference as audit
    return audit(reference)


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
    return _canonical_version_series(version, DeePHFCapabilityError)


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


def reference_fingerprint(reference, *, use_transaction=True) -> str:
    """Return a scratch-independent fingerprint of the scientific RHF state."""
    trusted = (
        transaction_reference_fingerprint(reference)
        if use_transaction
        else None
    )
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
