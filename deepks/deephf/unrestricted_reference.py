"""Strict PySCF adapters for molecular unrestricted response and adjoints."""

from dataclasses import dataclass
from functools import partial
import hashlib
from numbers import Real
import weakref

import numpy as np
import pyscf
from pyscf import dft
from pyscf.dft import libxc, numint
from pyscf.gto import mole as gto_mole
from pyscf.grad import uks as uks_grad
from pyscf.scf import hf as scf_hf

from .adjoint import AdjointError
from .capabilities import (
    DeePHFCapabilityError,
    reference_is_transaction_validated,
    transaction_reference_fingerprint,
)
from .contracts import (
    array_fingerprint as _array_fingerprint,
    dataclass_fingerprint,
    immutable_array as _immutable_array,
    integer_control,
    real_control,
    update_digest as _update_fingerprint_value,
    validated_float64_array,
    version_series as _canonical_version_series,
)
from .pyscf_dft_provenance import (
    RKSFunctionalProvenance,
    RKSGridProvenance,
    SUPPORTED_LIBXC_VERSION,
    SUPPORTED_NUMINT_CUTOFF,
    _GRID_PROVENANCE_CACHE,
    _normalized_functional_components,
    _static_callable_definitions,
    _validate_dft_implementations,
)
from .pyscf_rks_reference import _dft_reference_validation_fingerprint


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
    from .audits.unrestricted_reference import validate_uhf_reference as audit
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


def uhf_reference_fingerprint(reference, *, use_transaction=True) -> str:
    """Return a scratch-independent fingerprint of the scientific UHF state."""
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
        uhf_molecule_science_fingerprint(reference.mol),
        float(reference.e_tot),
        np.asarray(reference.mo_coeff),
        np.asarray(reference.mo_energy),
        np.asarray(reference.mo_occ),
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

_SUPPORTED_NATIVE_UNRESTRICTED_GRADIENT = _native_unrestricted_gradient
_NATIVE_UKS_GRADIENT_METHODS = (
    "hcore_generator",
    "get_ovlp",
    "_tag_rdm1",
    "make_rdm1e",
    "get_veff",
    "get_j",
    "grad_nuc",
)
_SUPPORTED_NATIVE_UKS_GRADIENT = (
    _static_callable_definitions(uks_grad.Gradients, _NATIVE_UKS_GRADIENT_METHODS),
    tuple(
        (name, value)
        for name, value in vars(uks_grad).items()
        if callable(value)
    ),
)


def _validate_native_uks_gradient() -> None:
    _validate_dft_implementations("UKS")
    expected_methods, expected_module = _SUPPORTED_NATIVE_UKS_GRADIENT
    current_methods = _static_callable_definitions(
        uks_grad.Gradients, _NATIVE_UKS_GRADIENT_METHODS
    )
    changed = [
        name
        for (name, owner, definition), (expected_name, expected_owner, expected_definition)
        in zip(current_methods, expected_methods, strict=True)
        if name != expected_name or owner is not expected_owner or definition is not expected_definition
    ]
    changed.extend(
        name
        for name, implementation in expected_module
        if vars(uks_grad).get(name) is not implementation
    )
    if changed or _native_unrestricted_gradient is not _SUPPORTED_NATIVE_UNRESTRICTED_GRADIENT:
        raise DeePHFCapabilityError("the strict UKS native-gradient implementation changed")


class UKSResponseError(UHFResponseError):
    """Raised when a strict finite-grid UKS response fails its contract."""


class UKSAdjointError(UHFAdjointError):
    """Raised when a correction-specific UKS adjoint fails its contract."""


@dataclass(frozen=True)
class UKSResponseDiagnostics:
    """Finite-grid provenance and coupled-response diagnostics."""

    core: UHFResponseDiagnostics
    functional: RKSFunctionalProvenance
    grid: RKSGridProvenance
    hamiltonian_reconstruction_residual: float

    def __getattr__(self, name):
        return getattr(self.core, name)


@dataclass(frozen=True)
class UKSResponse:
    """Complete spin-resolved finite-grid UKS nuclear response."""

    core: UHFResponse
    functional: RKSFunctionalProvenance
    grid: RKSGridProvenance
    hamiltonian_derivative_fixed_grid_spin: np.ndarray
    xc_hamiltonian_derivative_grid_coordinate_spin: np.ndarray
    xc_hamiltonian_derivative_grid_weight_spin: np.ndarray
    diagnostics: UKSResponseDiagnostics
    integrity_fingerprint: str

    def __getattr__(self, name):
        return getattr(self.core, name)


@dataclass(frozen=True)
class UKSAdjointDiagnostics:
    """Finite-grid provenance and coupled scalar-adjoint diagnostics."""

    core: UHFAdjointDiagnostics
    functional: RKSFunctionalProvenance
    grid: RKSGridProvenance
    nuclear_partition_residual: float | None

    def __getattr__(self, name):
        return getattr(self.core, name)


@dataclass(frozen=True)
class UKSAdjoint:
    """One immutable coupled alpha/beta UKS scalar-adjoint result."""

    core: UHFAdjoint
    functional: RKSFunctionalProvenance
    grid: RKSGridProvenance
    correction_gradient_adjoint_fixed_grid_spin: np.ndarray
    correction_gradient_adjoint_grid_coordinate_spin: np.ndarray
    correction_gradient_adjoint_grid_weight_spin: np.ndarray
    correction_gradient_adjoint_fixed_grid: np.ndarray
    correction_gradient_adjoint_grid_coordinate: np.ndarray
    correction_gradient_adjoint_grid_weight: np.ndarray
    diagnostics: UKSAdjointDiagnostics
    integrity_fingerprint: str

    def __getattr__(self, name):
        return getattr(self.core, name)


def uks_response_integrity_fingerprint(response: UKSResponse) -> str:
    """Return a digest covering every public UKS response field."""
    return dataclass_fingerprint(
        response,
        excluded=frozenset({"integrity_fingerprint"}),
    )


def uks_adjoint_integrity_fingerprint(adjoint: UKSAdjoint) -> str:
    """Return a digest covering every public UKS adjoint field."""
    return dataclass_fingerprint(
        adjoint,
        excluded=frozenset({"integrity_fingerprint"}),
    )


def _uks_functional_provenance(reference) -> RKSFunctionalProvenance:
    _validate_dft_implementations("UKS")
    integration = reference._numint
    if type(integration) is not numint.NumInt:
        raise DeePHFCapabilityError(
            "the strict UKS tier requires an exact native pyscf.dft.numint.NumInt"
        )
    if integration.libxc is not libxc:
        raise DeePHFCapabilityError("the strict UKS tier requires native LibXC")
    if str(libxc.__version__) != SUPPORTED_LIBXC_VERSION:
        raise DeePHFCapabilityError(
            "the strict UKS tier requires the characterized LibXC 7.0.0 backend"
        )
    if integration.omega is not None:
        raise DeePHFCapabilityError(
            "the strict UKS tier requires an unset range-separation parameter"
        )
    cutoff = integration.cutoff
    if (
        isinstance(cutoff, (bool, np.bool_))
        or not isinstance(cutoff, Real)
        or not np.isfinite(float(cutoff))
        or float(cutoff) != SUPPORTED_NUMINT_CUTOFF
    ):
        raise DeePHFCapabilityError("the strict UKS tier requires the native NumInt cutoff")
    hooks = sorted(name for name, value in integration.__dict__.items() if callable(value))
    if hooks:
        raise DeePHFCapabilityError(
            "the UKS NumInt object has unsupported instance hooks: " + ", ".join(hooks)
        )
    if reference.xc in libxc._CUSTOM_FUNC_R:
        raise DeePHFCapabilityError(
            "the strict UKS tier does not support registered custom LibXC functionals"
        )
    components = _normalized_functional_components(reference.xc)
    try:
        xc_type = integration._xc_type(reference.xc)
        hybrid = float(integration.hybrid_coeff(reference.xc, spin=reference.mol.spin))
        range_separation = tuple(float(value) for value in integration.rsh_coeff(reference.xc))
        has_nlc = bool(libxc.is_nlc(reference.xc))
        libxc.test_deriv_order(reference.xc, 2, raise_error=True)
    except Exception as error:
        raise DeePHFCapabilityError(
            f"the UKS LibXC functional metadata is invalid: {error}"
        ) from error
    if xc_type != "LDA" or hybrid != 0.0 or range_separation != (0.0, 0.0, 0.0):
        raise DeePHFCapabilityError(
            "the strict UKS tier requires the pure LDA_X + LDA_C_VWN functional"
        )
    if has_nlc or reference.do_nlc():
        raise DeePHFCapabilityError("the strict UKS tier does not support NLC")
    sample_density = np.array(
        (
            (1.0e-8, 1.0e-5, 1.0e-3, 5.0e-2, 5.0e-1, 2.0),
            (2.0e-8, 5.0e-6, 2.0e-3, 2.0e-2, 2.5e-1, 1.0),
        ),
        dtype=np.float64,
    )
    try:
        actual = np.asarray(libxc.eval_xc1(reference.xc, sample_density, spin=1, deriv=2))
        canonical = np.asarray(
            libxc.eval_xc1("LDA_X,LDA_C_VWN", sample_density, spin=1, deriv=2)
        )
    except Exception as error:
        raise DeePHFCapabilityError(
            f"the UKS LibXC functional signature could not be evaluated: {error}"
        ) from error
    if (
        actual.dtype != np.dtype(np.float64)
        or not np.isfinite(actual).all()
        or not np.array_equal(actual, canonical)
    ):
        residual = float(np.max(np.abs(actual - canonical), initial=0.0))
        raise DeePHFCapabilityError(
            "the UKS LibXC parameters do not match the canonical tier: "
            f"residual {residual:.3e}"
        )
    return RKSFunctionalProvenance(
        backend_module=libxc.__name__,
        backend_version=str(libxc.__version__),
        backend_reference=str(libxc.__reference__),
        numint_cutoff=float(cutoff),
        xc_type=xc_type,
        components=components,
        hybrid_coefficient=hybrid,
        range_separation=range_separation,
        has_nlc=has_nlc,
        signature=_array_fingerprint(canonical),
    )


def _dense_uks_quadrature(reference, density: np.ndarray):
    integration = reference._numint
    coordinates = np.asarray(reference.grids.coords)
    weights = np.asarray(reference.grids.weights)
    try:
        ao = integration.eval_ao(reference.mol, coordinates, deriv=0)
        rho = np.stack(
            [
                np.einsum("gp,pq,gq->g", ao, spin_density, ao, optimize=True)
                for spin_density in density
            ]
        )
        values = integration.eval_xc_eff(
            reference.xc,
            rho,
            deriv=1,
            xctype="LDA",
            spin=1,
        )
        energy_density = np.asarray(values[0])
        potential = np.asarray(values[1])[:, 0]
        electron_counts = np.einsum("g,sg->s", weights, rho)
        xc_energy = float(np.dot(weights, rho.sum(axis=0) * energy_density))
        xc_potential = np.einsum(
            "g,sg,gp,gq->spq",
            weights,
            potential,
            ao,
            ao,
            optimize=True,
        )
    except Exception as error:
        raise DeePHFCapabilityError(
            f"the independent dense UKS LDA quadrature failed: {error}"
        ) from error
    values = (electron_counts, np.asarray(xc_energy), xc_potential)
    if not all(np.isfinite(value).all() for value in values):
        raise DeePHFCapabilityError("the independent dense UKS LDA quadrature is nonfinite")
    return electron_counts, xc_energy, xc_potential


_VALIDATED_UKS_REFERENCES = weakref.WeakKeyDictionary()


def _audit_uks_reference(reference):
    from .audits.unrestricted_reference import _audit_uks_reference as audit
    return audit(reference)


def validate_uks_reference(reference):
    """Validate a UKS reference once per unchanged scientific state."""
    if reference_is_transaction_validated(reference):
        return reference
    _validate_native_uks_gradient()
    if type(reference) is not dft.uks.UKS:
        return _audit_uks_reference(reference)
    try:
        fingerprint = _dft_reference_validation_fingerprint(reference)
    except Exception:
        return _audit_uks_reference(reference)
    if _VALIDATED_UKS_REFERENCES.get(reference) == fingerprint:
        return reference
    _audit_uks_reference(reference)
    _VALIDATED_UKS_REFERENCES[reference] = fingerprint
    _GRID_PROVENANCE_CACHE[reference] = (
        fingerprint,
        _GRID_PROVENANCE_CACHE[reference][1],
    )
    return reference


def audit_uks_reference(reference):
    """Run all expensive finite-grid and independent-quadrature checks."""
    result = _audit_uks_reference(reference)
    fingerprint = _dft_reference_validation_fingerprint(reference)
    _VALIDATED_UKS_REFERENCES[reference] = fingerprint
    _GRID_PROVENANCE_CACHE[reference] = (
        fingerprint,
        _GRID_PROVENANCE_CACHE[reference][1],
    )
    return result


def uks_reference_fingerprint(reference, *, use_transaction=True) -> str:
    """Fingerprint one accepted UKS molecular, orbital, functional, and grid state."""
    trusted = (
        transaction_reference_fingerprint(reference)
        if use_transaction
        else None
    )
    if trusted is not None:
        return trusted
    return _dft_reference_validation_fingerprint(reference)
