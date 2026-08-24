"""Isolated PySCF 2.14 adapter for molecular pure-LDA RKS response."""

from dataclasses import dataclass, fields, replace
import ctypes
import hashlib
import inspect
from numbers import Real
import operator
from typing import Any
import weakref

import numpy as np
import pyscf
from pyscf import dft, gto
from pyscf.dft import gen_grid, libxc, numint, radi
from pyscf.gto import mole as gto_mole
from pyscf.grad import rhf as rhf_grad, rks as rks_grad
from pyscf.hessian import rks as rks_hessian
from pyscf.scf import cphf, hf as scf_hf

from deepks.descriptor import is_ghost_atom

from .adjoint import (
    AdjointError,
    ScalarAdjointProblem,
    scalar_operator_fingerprint,
    solve_scalar_adjoint,
)
from .capabilities import (
    DeePHFCapabilityError,
    reference_is_transaction_validated,
    transaction_reference_fingerprint,
)
from .pyscf_rhf import RHFResponse


SUPPORTED_PYSCF_SERIES = (2, 14)
SUPPORTED_LIBXC_VERSION = "7.0.0"
SUPPORTED_LIBXC_COMPONENTS = ((1, 1.0), (7, 1.0))
SUPPORTED_ATOM_GRID = (20, 50)
SUPPORTED_NUMINT_CUTOFF = 1.0e-13
SUPPORTED_GRID_CUTOFF = 1.0e-15
SUPPORTED_BRAGG_RADII_FINGERPRINT = (
    "d5eeefc53bb8261154cd2317ff60e5e642dd9cde1d1f283647b7956756b74a43"
)
_SUPPORTED_RADI_METHOD = radi.treutler_ahlrichs
_SUPPORTED_RADII_ADJUST = radi.treutler_atomic_radii_adjust
_SUPPORTED_BECKE_SCHEME = gen_grid.original_becke
_SUPPORTED_GRIDS_RESPONSE = rks_grad.grids_response_cc
_SUPPORTED_NUMINT_IMPLEMENTATIONS = tuple(
    (name, getattr(numint.NumInt, name))
    for name in (
        "_xc_type",
        "eval_ao",
        "eval_xc_eff",
        "hybrid_coeff",
        "nr_rks",
        "nr_rks_fxc",
        "rsh_coeff",
    )
)
_SUPPORTED_LIBXC_IMPLEMENTATIONS = tuple(
    (name, getattr(libxc, name))
    for name in (
        "XCFunctionalCache",
        "eval_xc1",
        "is_nlc",
        "parse_xc",
        "test_deriv_order",
    )
)
_UKS_REFERENCE_TYPE = getattr(getattr(dft, "uks"), "UKS")
_SUPPORTED_REFERENCE_IMPLEMENTATIONS = {
    "RKS": tuple(
        (name, getattr(dft.rks.RKS, name))
        for name in ("get_hcore", "get_ovlp", "get_veff", "make_rdm1")
    ),
    "UKS": tuple(
        (name, getattr(_UKS_REFERENCE_TYPE, name))
        for name in ("get_hcore", "get_ovlp", "get_veff", "make_rdm1")
    ),
}


def _static_callable_definitions(cls, names):
    return tuple(
        (
            name,
            next(owner for owner in cls.__mro__ if name in vars(owner)),
            inspect.getattr_static(cls, name),
        )
        for name in names
    )


_NATIVE_RKS_GRADIENT_METHODS = (
    "kernel",
    "grad_elec",
    "hcore_generator",
    "get_ovlp",
    "_tag_rdm1",
    "make_rdm1e",
    "get_veff",
    "get_j",
    "grad_nuc",
    "extra_force",
    "_finalize",
)
_SUPPORTED_NATIVE_RKS_GRADIENT = (
    _static_callable_definitions(rks_grad.Gradients, _NATIVE_RKS_GRADIENT_METHODS),
    tuple(
        (module, name, value)
        for module in (rhf_grad, rks_grad)
        for name, value in vars(module).items()
        if callable(value) and name != "grids_response_cc"
    ),
)
_GRID_WEIGHT_FD_STEP = 1.0e-5
_GRID_WEIGHT_DERIVATIVE_ATOL = 1.0e-6
_GRID_WEIGHT_DERIVATIVE_RTOL = 1.0e-7
_GRID_RESPONSE_WEIGHT_ATOL = 1.0e-180
_VALIDATED_RKS_REFERENCES = weakref.WeakKeyDictionary()
_GRID_PROVENANCE_CACHE = weakref.WeakKeyDictionary()


class RKSResponseError(RuntimeError):
    """Raised when the strict RKS response contract fails."""


class RKSAdjointError(AdjointError):
    """Raised when the strict RKS scalar-adjoint contract fails."""


@dataclass(frozen=True)
class RKSFunctionalProvenance:
    """Normalized identity of the supported pure-LDA LibXC functional."""

    backend_module: str
    backend_version: str
    backend_reference: str
    numint_cutoff: float
    xc_type: str
    components: tuple[tuple[int, float], ...]
    hybrid_coefficient: float
    range_separation: tuple[float, float, float]
    has_nlc: bool
    signature: str


@dataclass(frozen=True)
class RKSGridProvenance:
    """Identity of one deterministic, unpruned atom-centered grid."""

    grid_class: str
    generator: str
    response_generator: str
    atom_grid: tuple[tuple[str, tuple[int, int]], ...]
    radi_method: str
    radii_adjust: str
    atomic_radii_fingerprint: str
    becke_scheme: str
    prune: None
    alignment: int
    cutoff: float
    small_rho_cutoff: float
    sort_grids: bool
    point_count: int
    coordinates_fingerprint: str
    weights_fingerprint: str
    weight_derivatives_fingerprint: str
    atom_indices_fingerprint: str
    quadrature_weights_fingerprint: str
    nonzero_table_fingerprint: str
    ao_order_fingerprint: str


@dataclass(frozen=True)
class RKSResponseDiagnostics:
    """Independent diagnostics for one complete molecular RKS CPKS solve."""

    minimum_orbital_gap: float
    pyscf_version: str
    libxc_version: str
    functional_components: tuple[tuple[int, float], ...]
    grid_point_count: int
    grid_coordinates_fingerprint: str
    grid_weights_fingerprint: str
    quadrature_electron_count: float
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
    hamiltonian_reconstruction_residual: float
    metric_residual: float
    idempotency_residual: float
    particle_number_residual: float
    translation_residual: float | None
    refinement_cycles: int
    residual_history: tuple[float, ...]


@dataclass(frozen=True)
class RKSResponse(RHFResponse):
    """RKS response extending the canonical RHF MO representation."""

    functional_provenance: RKSFunctionalProvenance
    grid_provenance: RKSGridProvenance
    hamiltonian_derivative_fixed_grid: np.ndarray
    xc_hamiltonian_derivative_grid_coordinate: np.ndarray
    xc_hamiltonian_derivative_grid_weight: np.ndarray
    diagnostics: RKSResponseDiagnostics


@dataclass(frozen=True)
class RKSAdjointDiagnostics:
    """Independent diagnostics for one correction-specific RKS adjoint."""

    minimum_orbital_gap: float
    pyscf_version: str
    libxc_version: str
    functional_components: tuple[tuple[int, float], ...]
    grid_point_count: int
    grid_coordinates_fingerprint: str
    grid_weights_fingerprint: str
    residual_tolerance: float
    invariant_tolerance: float
    orbital_gap_tolerance: float
    response_dimension: int
    operator_is_self_adjoint: bool
    hamiltonian_reconstruction_residual: float
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
class RKSAdjoint:
    """Immutable RKS Z-vector and its finite-grid nuclear contractions."""

    reference_identity: int
    state_fingerprint: str
    integrity_fingerprint: str
    operator_fingerprint: str
    atom_indices: tuple[int, ...]
    functional_provenance: RKSFunctionalProvenance
    grid_provenance: RKSGridProvenance
    objective_ao_potential: np.ndarray
    objective_orbital_gradient: np.ndarray
    zvector: np.ndarray
    residual: np.ndarray
    adjoint_ao_density: np.ndarray
    adjoint_ao_potential: np.ndarray
    correction_gradient_metric: np.ndarray
    correction_gradient_adjoint_fixed_grid: np.ndarray
    correction_gradient_adjoint_grid_coordinate: np.ndarray
    correction_gradient_adjoint_grid_weight: np.ndarray
    correction_gradient_adjoint_nuclear: np.ndarray
    correction_gradient_adjoint_metric: np.ndarray
    correction_gradient_occupied_virtual: np.ndarray
    correction_gradient_response: np.ndarray
    diagnostics: RKSAdjointDiagnostics


def _version_series(version: str) -> tuple[int, int]:
    components = version.split(".")
    try:
        return int(components[0]), int(components[1])
    except (IndexError, ValueError) as error:
        raise DeePHFCapabilityError(
            f"cannot interpret the PySCF version {version!r}"
        ) from error


def validate_pyscf_version() -> None:
    """Require the PySCF series characterized by this adapter."""
    if _version_series(pyscf.__version__) != SUPPORTED_PYSCF_SERIES:
        raise DeePHFCapabilityError(
            "the RKS response adapter supports PySCF 2.14; "
            f"found {pyscf.__version__}"
        )


def _immutable_array(value: np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(value)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _array_fingerprint(value: np.ndarray) -> str:
    digest = hashlib.sha256()
    array = np.ascontiguousarray(value)
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(repr(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _qualified_name(value) -> str:
    return f"{value.__module__}.{value.__qualname__}"


def _update_fingerprint_value(digest, value: Any) -> None:
    if isinstance(value, np.ndarray):
        digest.update(_array_fingerprint(value).encode("ascii"))
    elif isinstance(value, np.generic):
        _update_fingerprint_value(digest, value.item())
    elif isinstance(value, dict):
        for key in sorted(value, key=lambda item: (type(item).__name__, repr(item))):
            _update_fingerprint_value(digest, key)
            _update_fingerprint_value(digest, value[key])
    elif isinstance(value, (tuple, list)):
        for item in value:
            _update_fingerprint_value(digest, item)
    elif value is None or isinstance(value, (bool, int, float, str)):
        digest.update(type(value).__name__.encode("ascii"))
        digest.update(repr(value).encode("utf-8"))
    else:
        digest.update(type(value).__name__.encode("utf-8"))
        digest.update(repr(value).encode("utf-8"))


def _validated_float64_array(value, expected_shape, name: str) -> np.ndarray:
    try:
        array = np.asarray(value)
    except Exception as error:
        raise RKSResponseError(f"{name} is not a numerical array: {error}") from error
    if array.shape != expected_shape:
        raise RKSResponseError(
            f"unexpected {name} shape {array.shape}; expected {expected_shape}"
        )
    if array.dtype != np.dtype(np.float64) or np.iscomplexobj(array):
        raise RKSResponseError(f"{name} must be a real float64 array")
    if not np.isfinite(array).all():
        raise RKSResponseError(f"{name} must be finite")
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


def _normalized_functional_components(xc_code: str) -> tuple[tuple[int, float], ...]:
    if not isinstance(xc_code, str):
        raise DeePHFCapabilityError("the RKS functional identifier must be a string")
    try:
        hybrid, components = libxc.parse_xc(xc_code)
    except Exception as error:
        raise DeePHFCapabilityError(
            f"the RKS LibXC functional could not be parsed: {error}"
        ) from error
    if tuple(float(value) for value in hybrid) != (0.0, 0.0, 0.0):
        raise DeePHFCapabilityError("the initial RKS tier requires a pure functional")
    combined: dict[int, float] = {}
    for component_id, factor in components:
        key = int(component_id)
        combined[key] = combined.get(key, 0.0) + float(factor)
    normalized = tuple(sorted(combined.items()))
    if normalized != SUPPORTED_LIBXC_COMPONENTS:
        raise DeePHFCapabilityError(
            "the initial RKS tier requires normalized LibXC components "
            f"{SUPPORTED_LIBXC_COMPONENTS}; found {normalized}"
        )
    return normalized


def _validate_dft_implementations(method: str) -> None:
    changed = [
        f"NumInt.{name}"
        for name, implementation in _SUPPORTED_NUMINT_IMPLEMENTATIONS
        if getattr(numint.NumInt, name) is not implementation
    ]
    changed.extend(
        f"libxc.{name}"
        for name, implementation in _SUPPORTED_LIBXC_IMPLEMENTATIONS
        if getattr(libxc, name) is not implementation
    )
    reference_type = getattr(dft, method.lower()).__dict__[method]
    changed.extend(
        f"{method}.{name}"
        for name, implementation in _SUPPORTED_REFERENCE_IMPLEMENTATIONS[method]
        if getattr(reference_type, name) is not implementation
    )
    if method == "RKS":
        class_definitions, module_definitions = _SUPPORTED_NATIVE_RKS_GRADIENT
        current_class_definitions = _static_callable_definitions(
            rks_grad.Gradients, _NATIVE_RKS_GRADIENT_METHODS
        )
        changed.extend(
            f"{method} gradient {name}"
            for (name, owner, definition), (
                expected_name,
                expected_owner,
                expected_definition,
            ) in zip(current_class_definitions, class_definitions, strict=True)
            if name != expected_name
            or owner is not expected_owner
            or definition is not expected_definition
        )
        changed.extend(
            f"{module.__name__}.{name}"
            for module, name, implementation in module_definitions
            if vars(module).get(name) is not implementation
        )
    if changed:
        raise DeePHFCapabilityError(
            f"the strict {method} DFT implementation changed: {', '.join(changed)}"
        )


def _evaluate_libxc_cache(cache, density: np.ndarray) -> np.ndarray:
    """Evaluate LDA energy density, potential, and kernel for one cache."""
    density = np.ascontiguousarray(density, dtype=np.float64)
    output = np.zeros((3, density.size), dtype=np.float64)
    factors = (ctypes.c_double * cache.nfunc)(*cache.facs)
    libxc._itrf.LIBXC_eval_xc(
        cache.nfunc,
        cache.xc_arr,
        factors,
        0,
        2,
        1,
        density.size,
        3,
        density.ctypes,
        output.ctypes,
    )
    return output


def _functional_provenance(reference) -> RKSFunctionalProvenance:
    _validate_dft_implementations("RKS")
    integration = reference._numint
    if type(integration) is not numint.NumInt:
        raise DeePHFCapabilityError(
            "the initial RKS tier requires an exact native pyscf.dft.numint.NumInt"
        )
    if integration.libxc is not libxc:
        raise DeePHFCapabilityError(
            "the initial RKS tier requires the native PySCF LibXC backend"
        )
    if str(integration.libxc.__version__) != SUPPORTED_LIBXC_VERSION:
        raise DeePHFCapabilityError(
            "the initial RKS tier requires the characterized LibXC 7.0.0 backend"
        )
    if integration.omega is not None:
        raise DeePHFCapabilityError(
            "the initial RKS tier requires an unset NumInt range-separation parameter"
        )
    integration_cutoff = integration.cutoff
    if (
        isinstance(integration_cutoff, (bool, np.bool_))
        or not isinstance(integration_cutoff, Real)
        or not np.isfinite(float(integration_cutoff))
        or float(integration_cutoff) != SUPPORTED_NUMINT_CUTOFF
    ):
        raise DeePHFCapabilityError(
            "the initial RKS tier requires the native NumInt cutoff"
        )
    custom_hooks = sorted(
        name for name, value in integration.__dict__.items() if callable(value)
    )
    if custom_hooks:
        raise DeePHFCapabilityError(
            "the RKS NumInt object has unsupported instance hooks: "
            + ", ".join(custom_hooks)
        )
    if reference.xc in libxc._CUSTOM_FUNC_R:
        raise DeePHFCapabilityError(
            "the initial RKS tier does not support registered custom LibXC functionals"
        )
    components = _normalized_functional_components(reference.xc)
    try:
        xc_type = integration._xc_type(reference.xc)
        hybrid_coefficient = float(integration.hybrid_coeff(reference.xc, spin=0))
        range_separation = tuple(
            float(value) for value in integration.rsh_coeff(reference.xc)
        )
        has_nlc = bool(integration.libxc.is_nlc(reference.xc))
        integration.libxc.test_deriv_order(reference.xc, 2, raise_error=True)
    except Exception as error:
        raise DeePHFCapabilityError(
            f"the RKS LibXC functional metadata is invalid: {error}"
        ) from error
    if xc_type != "LDA":
        raise DeePHFCapabilityError("the initial RKS tier requires a pure LDA functional")
    if hybrid_coefficient != 0.0 or range_separation != (0.0, 0.0, 0.0):
        raise DeePHFCapabilityError(
            "the initial RKS tier does not support hybrid or range-separated exchange"
        )
    if has_nlc or reference.do_nlc():
        raise DeePHFCapabilityError("the initial RKS tier does not support NLC")
    sample_density = np.array(
        (1.0e-8, 1.0e-5, 1.0e-3, 5.0e-2, 5.0e-1, 2.0),
        dtype=np.float64,
    )
    try:
        actual_values = integration.eval_xc_eff(
            reference.xc,
            sample_density,
            deriv=2,
            xctype="LDA",
            spin=0,
        )
        actual = np.stack(
            (
                np.asarray(actual_values[0]),
                np.asarray(actual_values[1])[0],
                np.asarray(actual_values[2])[0, 0],
            )
        )
        canonical_cache = libxc.XCFunctionalCache(
            "LDA_X,LDA_C_VWN",
            spin=0,
        )
        canonical = _evaluate_libxc_cache(canonical_cache, sample_density)
    except Exception as error:
        raise DeePHFCapabilityError(
            f"the RKS LibXC functional signature could not be evaluated: {error}"
        ) from error
    if (
        actual.dtype != np.dtype(np.float64)
        or not np.isfinite(actual).all()
        or not np.array_equal(actual, canonical)
    ):
        residual = float(np.max(np.abs(actual - canonical), initial=0.0))
        raise DeePHFCapabilityError(
            "the RKS LibXC parameters do not match the canonical LDA_X + "
            f"LDA_C_VWN tier: residual {residual:.3e}"
        )
    return RKSFunctionalProvenance(
        backend_module=integration.libxc.__name__,
        backend_version=str(integration.libxc.__version__),
        backend_reference=str(integration.libxc.__reference__),
        numint_cutoff=float(integration_cutoff),
        xc_type=xc_type,
        components=components,
        hybrid_coefficient=hybrid_coefficient,
        range_separation=range_separation,
        has_nlc=has_nlc,
        signature=_array_fingerprint(canonical),
    )


def _normalized_atom_grid(molecule, atom_grid) -> tuple[tuple[str, tuple[int, int]], ...]:
    symbols = tuple(sorted(set(molecule.atom_symbol(index) for index in range(molecule.natm))))
    if isinstance(atom_grid, (tuple, list)) and len(atom_grid) == 2:
        resolved = {symbol: atom_grid for symbol in symbols}
    elif isinstance(atom_grid, dict):
        default = atom_grid.get("default")
        allowed = set(symbols) | {"default"}
        if set(atom_grid) - allowed:
            raise DeePHFCapabilityError(
                "the RKS atom_grid contains entries outside the molecular elements"
            )
        resolved = {symbol: atom_grid.get(symbol, default) for symbol in symbols}
    else:
        raise DeePHFCapabilityError(
            "the initial RKS tier requires an explicit atom_grid specification"
        )
    for value in resolved.values():
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise DeePHFCapabilityError(
                "each RKS atom_grid specification must contain radial and angular counts"
            )
        counts = []
        for count in value:
            if isinstance(count, (bool, np.bool_)):
                raise DeePHFCapabilityError(
                    "RKS atom_grid counts must be positive integers"
                )
            try:
                normalized = operator.index(count)
            except TypeError as error:
                raise DeePHFCapabilityError(
                    "RKS atom_grid counts must be positive integers"
                ) from error
            if normalized <= 0:
                raise DeePHFCapabilityError(
                    "RKS atom_grid counts must be positive integers"
                )
            counts.append(normalized)
        if tuple(counts) != SUPPORTED_ATOM_GRID:
            raise DeePHFCapabilityError(
                f"the initial RKS tier requires {SUPPORTED_ATOM_GRID} grids for every element"
            )
    return tuple((symbol, SUPPORTED_ATOM_GRID) for symbol in symbols)


def _grid_arrays(grid) -> tuple[np.ndarray, ...]:
    arrays = (
        grid.coords,
        grid.weights,
        grid.atm_idx,
        grid.quadrature_weights,
        grid.non0tab,
    )
    if any(value is None for value in arrays):
        raise DeePHFCapabilityError(
            "the strict RKS grid must be prebuilt with nonzero-table metadata"
        )
    return tuple(np.asarray(value) for value in arrays)


def _build_strict_grid(molecule, atom_grid):
    grid = gen_grid.Grids(molecule)
    grid.atom_grid = {symbol: specification for symbol, specification in atom_grid}
    grid.radi_method = _SUPPORTED_RADI_METHOD
    grid.radii_adjust = _SUPPORTED_RADII_ADJUST
    grid.atomic_radii = radi.BRAGG_RADII
    grid.becke_scheme = _SUPPORTED_BECKE_SCHEME
    grid.prune = None
    grid.alignment = 1
    grid.symmetry = False
    grid.cutoff = SUPPORTED_GRID_CUTOFF
    grid.build(with_non0tab=True, sort_grids=False)
    return grid


def _finite_difference_grid_weight_derivative(
    molecule,
    atom_grid,
    central_atom_indices: np.ndarray,
) -> np.ndarray:
    coordinates = np.asarray(gto_mole.Mole.atom_coords(molecule, unit="Bohr"))
    derivative = np.empty((molecule.natm, 3, central_atom_indices.size))
    for atom_index in range(molecule.natm):
        for axis in range(3):
            displaced_weights = []
            for sign in (1.0, -1.0):
                displaced_coordinates = coordinates.copy()
                displaced_coordinates[atom_index, axis] += sign * _GRID_WEIGHT_FD_STEP
                displaced_molecule = gto_mole.Mole.set_geom_(
                    molecule,
                    displaced_coordinates,
                    unit="Bohr",
                    symmetry=False,
                    inplace=False,
                )
                displaced_grid = _build_strict_grid(displaced_molecule, atom_grid)
                displaced_atom_indices = np.asarray(displaced_grid.atm_idx)
                displaced_grid_weights = np.asarray(displaced_grid.weights)
                if (
                    displaced_grid_weights.dtype != np.dtype(np.float64)
                    or displaced_grid_weights.shape != central_atom_indices.shape
                    or not np.isfinite(displaced_grid_weights).all()
                    or not np.array_equal(
                        displaced_atom_indices,
                        central_atom_indices,
                    )
                ):
                    raise DeePHFCapabilityError(
                        "the displaced strict RKS grids do not preserve point ordering"
                    )
                displaced_weights.append(displaced_grid_weights)
            derivative[atom_index, axis] = (
                displaced_weights[0] - displaced_weights[1]
            ) / (2.0 * _GRID_WEIGHT_FD_STEP)
    return derivative


def _validated_grid_response_blocks(
    reference,
    atom_grid,
    *,
    audit_weight_derivative: bool,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...]:
    if rks_grad.grids_response_cc is not _SUPPORTED_GRIDS_RESPONSE:
        raise DeePHFCapabilityError(
            "the native PySCF RKS grid-response generator was modified"
        )
    grid = reference.grids
    molecule = reference.mol
    coordinates, weights, atom_indices, _quadrature_weights, _nonzero_table = (
        _grid_arrays(grid)
    )
    try:
        raw_blocks = tuple(_SUPPORTED_GRIDS_RESPONSE(grid))
    except Exception as error:
        raise DeePHFCapabilityError(
            f"the RKS grid-response generator failed: {error}"
        ) from error
    if len(raw_blocks) != molecule.natm:
        raise DeePHFCapabilityError(
            "the RKS grid-response generator returned an invalid host-atom partition"
        )
    blocks = []
    for host_atom, raw_block in enumerate(raw_blocks):
        if not isinstance(raw_block, (tuple, list)) or len(raw_block) != 3:
            raise DeePHFCapabilityError(
                "the RKS grid-response generator returned an invalid block"
            )
        block_coordinates, block_weights, weight_derivative = (
            np.asarray(value) for value in raw_block
        )
        host_points = np.flatnonzero(atom_indices == host_atom)
        if (
            host_points.size == 0
            or not np.array_equal(
                host_points,
                np.arange(host_points[0], host_points[-1] + 1),
            )
            or block_coordinates.shape != (host_points.size, 3)
            or block_weights.shape != (host_points.size,)
            or weight_derivative.shape
            != (molecule.natm, 3, host_points.size)
        ):
            raise DeePHFCapabilityError(
                "the RKS grid-response host-atom block shape is invalid"
            )
        values = (block_coordinates, block_weights, weight_derivative)
        if any(value.dtype != np.dtype(np.float64) for value in values):
            raise DeePHFCapabilityError(
                "the RKS grid-response blocks must use numpy.float64"
            )
        if not all(np.isfinite(value).all() for value in values):
            raise DeePHFCapabilityError(
                "the RKS grid-response blocks must be finite"
            )
        canonical_block_weights = weights[host_points]
        if not np.array_equal(
            block_coordinates,
            coordinates[host_points],
        ) or not np.allclose(
            block_weights,
            canonical_block_weights,
            rtol=0.0,
            atol=_GRID_RESPONSE_WEIGHT_ATOL,
        ):
            raise DeePHFCapabilityError(
                "the RKS grid-response host block does not match the energy grid"
            )
        blocks.append(
            (block_coordinates, canonical_block_weights, weight_derivative)
        )
    analytic_derivative = np.concatenate(
        [block[2] for block in blocks],
        axis=2,
    )
    translation_residual = float(
        np.max(np.abs(np.sum(analytic_derivative, axis=0)), initial=0.0)
    )
    if translation_residual > 1.0e-10:
        raise DeePHFCapabilityError(
            "the RKS grid-weight derivative violates nuclear translation invariance"
        )
    if audit_weight_derivative:
        finite_difference_derivative = _finite_difference_grid_weight_derivative(
            molecule,
            atom_grid,
            atom_indices,
        )
        if not np.allclose(
            analytic_derivative,
            finite_difference_derivative,
            rtol=_GRID_WEIGHT_DERIVATIVE_RTOL,
            atol=_GRID_WEIGHT_DERIVATIVE_ATOL,
        ):
            maximum_residual = float(
                np.max(
                    np.abs(analytic_derivative - finite_difference_derivative),
                    initial=0.0,
                )
            )
            raise DeePHFCapabilityError(
                "the RKS grid-weight derivative does not match independent finite "
                f"differences: residual {maximum_residual:.3e}"
            )
    return tuple(blocks)


def _build_grid_provenance(reference) -> RKSGridProvenance:
    molecule = reference.mol
    grid = reference.grids
    if type(grid) is not gen_grid.Grids or grid.mol is not molecule:
        raise DeePHFCapabilityError(
            "the initial RKS tier requires an exact native Grids bound to the reference molecule"
        )
    atom_grid = _normalized_atom_grid(molecule, grid.atom_grid)
    required_functions = (
        (grid.radi_method, _SUPPORTED_RADI_METHOD, "radial method"),
        (
            grid.radii_adjust,
            _SUPPORTED_RADII_ADJUST,
            "atomic-radii adjustment",
        ),
        (grid.becke_scheme, _SUPPORTED_BECKE_SCHEME, "Becke partition"),
    )
    for actual, expected, name in required_functions:
        if actual is not expected:
            raise DeePHFCapabilityError(
                f"the initial RKS grid requires the native default {name}"
            )
    if grid.prune is not None:
        raise DeePHFCapabilityError("the initial RKS grid must be unpruned")
    if grid.atomic_radii is not radi.BRAGG_RADII:
        raise DeePHFCapabilityError(
            "the initial RKS grid requires the native BRAGG_RADII table"
        )
    atomic_radii = np.asarray(grid.atomic_radii)
    if (
        type(grid.atomic_radii) is not np.ndarray
        or atomic_radii.dtype != np.dtype(np.float64)
        or atomic_radii.shape != (131,)
        or not np.isfinite(atomic_radii).all()
        or _array_fingerprint(atomic_radii)
        != SUPPORTED_BRAGG_RADII_FINGERPRINT
    ):
        raise DeePHFCapabilityError(
            "the native PySCF 2.14 BRAGG_RADII table was modified"
        )
    if isinstance(grid.alignment, (bool, np.bool_)):
        raise DeePHFCapabilityError(
            "the initial RKS grid requires integral alignment=1 without padding"
        )
    try:
        grid_alignment = operator.index(grid.alignment)
    except TypeError as error:
        raise DeePHFCapabilityError(
            "the initial RKS grid requires integral alignment=1 without padding"
        ) from error
    if grid_alignment != 1:
        raise DeePHFCapabilityError(
            "the initial RKS grid requires alignment=1 without padding"
        )
    if grid.symmetry is not False:
        raise DeePHFCapabilityError(
            "the initial RKS grid requires symmetry=False"
        )
    if isinstance(grid.cutoff, (bool, np.bool_)) or not isinstance(grid.cutoff, Real):
        raise DeePHFCapabilityError(
            "the initial RKS grid cutoff must be a finite real scalar"
        )
    grid_cutoff = float(grid.cutoff)
    if not np.isfinite(grid_cutoff):
        raise DeePHFCapabilityError(
            "the initial RKS grid cutoff must be a finite real scalar"
        )
    if grid_cutoff != SUPPORTED_GRID_CUTOFF:
        raise DeePHFCapabilityError(
            "the initial RKS grid requires the native PySCF cutoff"
        )
    grid_hooks = sorted(
        name
        for name, value in grid.__dict__.items()
        if name
        not in {"mol", "radi_method", "radii_adjust", "becke_scheme"}
        and callable(value)
    )
    if grid_hooks:
        raise DeePHFCapabilityError(
            "the RKS grid has unsupported instance hooks: " + ", ".join(grid_hooks)
        )
    coordinates, weights, atom_indices, quadrature_weights, nonzero_table = (
        _grid_arrays(grid)
    )
    if coordinates.dtype != np.dtype(np.float64) or weights.dtype != np.dtype(np.float64):
        raise DeePHFCapabilityError("the RKS grid coordinates and weights must be float64")
    if quadrature_weights.dtype != np.dtype(np.float64):
        raise DeePHFCapabilityError("the RKS quadrature weights must be float64")
    if coordinates.shape != (weights.size, 3):
        raise DeePHFCapabilityError("the RKS grid coordinate and weight shapes are invalid")
    if atom_indices.shape != weights.shape or quadrature_weights.shape != weights.shape:
        raise DeePHFCapabilityError("the RKS grid provenance array shapes are invalid")
    if not all(
        np.isfinite(value).all()
        for value in (coordinates, weights, quadrature_weights)
    ):
        raise DeePHFCapabilityError("the RKS grid contains nonfinite values")
    fresh = _build_strict_grid(molecule, atom_grid)
    fresh_arrays = _grid_arrays(fresh)
    names = (
        "coordinates",
        "weights",
        "atom indices",
        "quadrature weights",
        "nonzero table",
    )
    for actual, expected, name in zip(
        (coordinates, weights, atom_indices, quadrature_weights, nonzero_table),
        fresh_arrays,
        names,
        strict=True,
    ):
        if (
            actual.dtype != expected.dtype
            or actual.shape != expected.shape
            or not np.array_equal(actual, expected)
        ):
            raise DeePHFCapabilityError(
                f"the prebuilt RKS grid {name} do not match a fresh deterministic build"
            )
    response_blocks = _validated_grid_response_blocks(
        reference,
        atom_grid,
        audit_weight_derivative=True,
    )
    weight_derivatives = np.concatenate(
        [block[2] for block in response_blocks],
        axis=2,
    )
    ao_labels = np.asarray(tuple(molecule.ao_labels()), dtype=np.str_)
    return RKSGridProvenance(
        grid_class=_qualified_name(type(grid)),
        generator="pyscf.dft.gen_grid.Grids.build(with_non0tab=True,sort_grids=False)",
        response_generator=_qualified_name(_SUPPORTED_GRIDS_RESPONSE),
        atom_grid=atom_grid,
        radi_method=_qualified_name(grid.radi_method),
        radii_adjust=_qualified_name(grid.radii_adjust),
        atomic_radii_fingerprint=_array_fingerprint(np.asarray(grid.atomic_radii)),
        becke_scheme=_qualified_name(grid.becke_scheme),
        prune=None,
        alignment=grid_alignment,
        cutoff=grid_cutoff,
        small_rho_cutoff=float(reference.small_rho_cutoff),
        sort_grids=False,
        point_count=int(weights.size),
        coordinates_fingerprint=_array_fingerprint(coordinates),
        weights_fingerprint=_array_fingerprint(weights),
        weight_derivatives_fingerprint=_array_fingerprint(weight_derivatives),
        atom_indices_fingerprint=_array_fingerprint(atom_indices),
        quadrature_weights_fingerprint=_array_fingerprint(quadrature_weights),
        nonzero_table_fingerprint=_array_fingerprint(nonzero_table),
        ao_order_fingerprint=_array_fingerprint(ao_labels),
    )


def _grid_provenance(reference) -> RKSGridProvenance:
    """Return grid provenance, auditing it once per unchanged DFT state."""
    trusted_fingerprint = transaction_reference_fingerprint(reference)
    if trusted_fingerprint is not None:
        cached = _GRID_PROVENANCE_CACHE.get(reference)
        if cached is not None and cached[0] == trusted_fingerprint:
            return cached[1]
        provenance = _build_grid_provenance(reference)
        _GRID_PROVENANCE_CACHE[reference] = (
            trusted_fingerprint,
            provenance,
        )
        return provenance
    try:
        fingerprint = _dft_reference_validation_fingerprint(reference)
    except Exception:
        return _build_grid_provenance(reference)
    cached = _GRID_PROVENANCE_CACHE.get(reference)
    if cached is not None and cached[0] == fingerprint:
        return cached[1]
    provenance = _build_grid_provenance(reference)
    _GRID_PROVENANCE_CACHE[reference] = (fingerprint, provenance)
    return provenance


def _dense_ground_state_lda_quadrature(
    reference,
    density: np.ndarray,
) -> tuple[float, float, np.ndarray]:
    """Independently integrate the finite-grid LDA ground-state quantities."""
    coordinates = np.asarray(reference.grids.coords)
    weights = np.asarray(reference.grids.weights)
    integration = reference._numint
    try:
        ao = integration.eval_ao(reference.mol, coordinates, deriv=0)
        rho = np.einsum(
            "gp,pq,gq->g",
            ao,
            density,
            ao,
            optimize=True,
        )
        xc_values = integration.eval_xc_eff(
            reference.xc,
            rho,
            deriv=1,
            xctype="LDA",
            spin=0,
        )
        energy_density = np.asarray(xc_values[0])
        potential = np.asarray(xc_values[1])[0]
        electron_count = float(np.dot(weights, rho))
        xc_energy = float(np.dot(weights, rho * energy_density))
        xc_potential = np.einsum(
            "g,g,gp,gq->pq",
            weights,
            potential,
            ao,
            ao,
            optimize=True,
        )
    except Exception as error:
        raise DeePHFCapabilityError(
            f"the independent dense RKS LDA quadrature failed: {error}"
        ) from error
    if not np.isfinite((electron_count, xc_energy)).all() or not np.isfinite(
        xc_potential
    ).all():
        raise DeePHFCapabilityError(
            "the independent dense RKS LDA quadrature is nonfinite"
        )
    return electron_count, xc_energy, xc_potential


def _audit_rks_reference(reference):
    """Audit the exact native, converged, finite-grid pure-LDA RKS tier."""
    validate_pyscf_version()
    if type(reference) is not dft.rks.RKS:
        raise DeePHFCapabilityError(
            "RKS DeePHF requires an undecorated native pyscf.dft.rks.RKS reference"
        )
    if not reference.converged:
        raise DeePHFCapabilityError("the RKS reference must be converged")
    molecule = reference.mol
    if type(molecule) is not gto_mole.Mole:
        raise DeePHFCapabilityError(
            "the RKS reference must use a native molecular pyscf.gto.Mole"
        )
    if molecule.spin != 0 or molecule.nelectron % 2:
        raise DeePHFCapabilityError(
            "the initial RKS tier requires an even-electron closed-shell molecule"
        )
    if molecule.symmetry is not False:
        raise DeePHFCapabilityError(
            "the RKS reference must not use symmetry-constrained occupations"
        )
    if molecule.cart:
        raise DeePHFCapabilityError(
            "the initial RKS force tier requires spherical AO functions"
        )
    if getattr(molecule, "_pseudo", None):
        raise DeePHFCapabilityError(
            "the initial RKS force tier does not support pseudopotentials"
        )
    if getattr(molecule, "_ecp", None) or molecule.has_ecp():
        raise DeePHFCapabilityError(
            "the initial RKS force tier requires an all-electron reference"
        )
    if float(getattr(molecule, "omega", 0.0)) != 0.0:
        raise DeePHFCapabilityError(
            "the initial RKS force tier requires the full Coulomb interaction"
        )
    if getattr(molecule, "nucmod", None):
        raise DeePHFCapabilityError(
            "the initial RKS force tier requires point nuclei"
        )
    ghost_indices = [
        atom_index
        for atom_index in range(molecule.natm)
        if is_ghost_atom(molecule, atom_index)
    ]
    if ghost_indices:
        raise DeePHFCapabilityError(
            "the initial RKS force tier requires real atoms; ghost indices: "
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
            "the RKS reference has unsupported decorations: "
            + ", ".join(active_decorations)
        )
    if reference.nlc not in ("", None, 0, False):
        raise DeePHFCapabilityError("the initial RKS tier does not support NLC")
    if (
        isinstance(reference.small_rho_cutoff, (bool, np.bool_))
        or not isinstance(reference.small_rho_cutoff, Real)
    ):
        raise DeePHFCapabilityError(
            "the initial RKS grid small_rho_cutoff must be a finite real scalar"
        )
    small_rho_cutoff = float(reference.small_rho_cutoff)
    if not np.isfinite(small_rho_cutoff):
        raise DeePHFCapabilityError(
            "the initial RKS grid small_rho_cutoff must be a finite real scalar"
        )
    if small_rho_cutoff != 0.0:
        raise DeePHFCapabilityError(
            "the initial RKS grid requires small_rho_cutoff=0 without density pruning"
        )
    custom_hooks = sorted(
        name
        for name, value in reference.__dict__.items()
        if name not in {"mol", "grids", "nlcgrids", "_numint"}
        and callable(value)
    )
    if custom_hooks:
        raise DeePHFCapabilityError(
            "the RKS reference has unsupported instance hooks: "
            + ", ".join(custom_hooks)
        )
    molecule_hooks = sorted(
        name for name, value in molecule.__dict__.items() if callable(value)
    )
    if molecule_hooks:
        raise DeePHFCapabilityError(
            "the RKS molecule has unsupported instance hooks: "
            + ", ".join(molecule_hooks)
        )
    functional_provenance = _functional_provenance(reference)
    grid_provenance = _build_grid_provenance(reference)
    _GRID_PROVENANCE_CACHE[reference] = (None, grid_provenance)
    if reference.mo_coeff is None or reference.mo_energy is None:
        raise DeePHFCapabilityError("the RKS reference orbital state is incomplete")
    if reference.mo_occ is None:
        raise DeePHFCapabilityError("the RKS reference occupations are missing")
    coefficient = np.asarray(reference.mo_coeff)
    energy = np.asarray(reference.mo_energy)
    occupation = np.asarray(reference.mo_occ)
    orbital_values = (coefficient, energy, occupation)
    if any(np.iscomplexobj(value) for value in orbital_values):
        raise DeePHFCapabilityError("the RKS orbitals must be real")
    if any(value.dtype != np.dtype(np.float64) for value in orbital_values):
        raise DeePHFCapabilityError("the RKS orbital state must use numpy.float64")
    if not all(np.isfinite(value).all() for value in orbital_values):
        raise DeePHFCapabilityError("the RKS orbital state must be finite")
    if coefficient.shape != (molecule.nao, molecule.nao):
        raise DeePHFCapabilityError(
            "the RKS response requires a complete square MO coefficient matrix"
        )
    if energy.shape != (molecule.nao,) or occupation.shape != (molecule.nao,):
        raise DeePHFCapabilityError("the RKS orbital energy or occupation shape is invalid")
    if not np.all(np.isin(occupation, (0.0, 2.0))):
        raise DeePHFCapabilityError(
            "the RKS occupations must be integer closed-shell occupations"
        )
    occupied_count = molecule.nelectron // 2
    expected_occupation = np.zeros_like(occupation)
    expected_occupation[:occupied_count] = 2.0
    if not np.array_equal(occupation, expected_occupation):
        raise DeePHFCapabilityError(
            "the initial RKS force tier requires the Aufbau ground-state root"
        )
    if occupied_count == 0 or occupied_count == molecule.nao:
        raise DeePHFCapabilityError("RKS response requires occupied and virtual orbitals")
    if np.any(np.diff(energy) < -1.0e-10):
        raise DeePHFCapabilityError("the RKS canonical orbital energies are not ordered")
    if energy[occupied_count] - energy[occupied_count - 1] <= 0.0:
        raise DeePHFCapabilityError("the RKS occupied and virtual root spaces overlap")
    if not np.isfinite(reference.e_tot):
        raise DeePHFCapabilityError("the RKS reference energy must be finite")
    try:
        overlap = np.asarray(reference.get_ovlp())
        hcore = np.asarray(reference.get_hcore())
        density = np.asarray(reference.make_rdm1())
        effective_potential = np.asarray(reference.get_veff(molecule, density))
        coulomb, _exchange = scf_hf.get_jk(molecule, density, hermi=1)
        quadrature_electrons, xc_energy, xc_potential = reference._numint.nr_rks(
            molecule,
            reference.grids,
            reference.xc,
            density,
            hermi=1,
        )
        coulomb = np.asarray(coulomb)
        xc_potential = np.asarray(xc_potential)
        (
            dense_quadrature_electrons,
            dense_xc_energy,
            dense_xc_potential,
        ) = _dense_ground_state_lda_quadrature(reference, density)
        direct_effective_potential = coulomb + xc_potential
    except Exception as error:
        raise DeePHFCapabilityError(
            f"the RKS reference matrices could not be evaluated: {error}"
        ) from error
    expected_ao_shape = (molecule.nao, molecule.nao)
    ao_values = (
        overlap,
        hcore,
        density,
        effective_potential,
        coulomb,
        xc_potential,
        direct_effective_potential,
    )
    if any(value.shape != expected_ao_shape for value in ao_values):
        raise DeePHFCapabilityError("the RKS AO matrix shape is invalid")
    if any(np.iscomplexobj(value) for value in ao_values):
        raise DeePHFCapabilityError("the RKS AO matrices must be real")
    if any(value.dtype != np.dtype(np.float64) for value in ao_values):
        raise DeePHFCapabilityError("the RKS AO matrices must use numpy.float64")
    if not all(np.isfinite(value).all() for value in ao_values):
        raise DeePHFCapabilityError("the RKS AO matrices must be finite")
    if not np.isfinite((quadrature_electrons, xc_energy)).all():
        raise DeePHFCapabilityError("the RKS XC quadrature result must be finite")
    quadrature_residual = max(
        abs(float(quadrature_electrons) - dense_quadrature_electrons),
        abs(float(xc_energy) - dense_xc_energy),
        float(
            np.max(
                np.abs(xc_potential - dense_xc_potential),
                initial=0.0,
            )
        ),
    )
    if quadrature_residual > 1.0e-10:
        raise DeePHFCapabilityError(
            "the native RKS XC quadrature does not match the independent dense "
            f"LDA reconstruction: residual {quadrature_residual:.3e}"
        )
    interaction_residual = float(
        np.max(
            np.abs(effective_potential - direct_effective_potential),
            initial=0.0,
        )
    )
    if interaction_residual > 1.0e-10:
        raise DeePHFCapabilityError(
            "the RKS effective potential does not match direct Coulomb plus LibXC "
            f"quadrature: residual {interaction_residual:.3e}"
        )
    overlap_eigenvalues = np.linalg.eigvalsh(overlap)
    if overlap_eigenvalues[0] <= 1.0e-10:
        raise DeePHFCapabilityError("the RKS AO overlap is singular or ill conditioned")
    orthonormality_residual = float(
        np.max(
            np.abs(coefficient.T @ overlap @ coefficient - np.eye(molecule.nao)),
            initial=0.0,
        )
    )
    if orthonormality_residual > 1.0e-8:
        raise DeePHFCapabilityError(
            "the RKS orbitals violate AO-metric orthonormality: "
            f"{orthonormality_residual:.3e}"
        )
    density_symmetry_residual = float(
        np.max(np.abs(density - density.T), initial=0.0)
    )
    if density_symmetry_residual > 1.0e-10:
        raise DeePHFCapabilityError("the RKS AO density violates symmetry")
    electron_count = float(np.einsum("ij,ji->", density, overlap))
    if not np.isclose(electron_count, molecule.nelectron, rtol=0.0, atol=1.0e-8):
        raise DeePHFCapabilityError(
            f"the RKS AO density has an inconsistent electron count: {electron_count:.12g}"
        )
    idempotency_residual = float(
        np.max(np.abs(density @ overlap @ density - 2.0 * density), initial=0.0)
    )
    if idempotency_residual > 1.0e-8:
        raise DeePHFCapabilityError(
            "the RKS AO density violates closed-shell metric idempotency: "
            f"{idempotency_residual:.3e}"
        )
    fock = hcore + direct_effective_potential
    canonical_residual = fock @ coefficient - overlap @ (coefficient * energy)
    maximum_canonical_residual = float(
        np.max(np.abs(canonical_residual), initial=0.0)
    )
    if maximum_canonical_residual > 1.0e-7:
        raise DeePHFCapabilityError(
            "the stored RKS orbitals and energies do not satisfy the canonical "
            f"SCF equations: residual {maximum_canonical_residual:.3e}"
        )
    recomputed_energy = (
        np.einsum("ij,ji->", hcore, density)
        + 0.5 * np.einsum("ij,ji->", coulomb, density)
        + float(xc_energy)
        + molecule.energy_nuc()
    )
    if not np.isclose(recomputed_energy, reference.e_tot, rtol=0.0, atol=1.0e-8):
        raise DeePHFCapabilityError(
            "the stored RKS total energy is inconsistent with its finite-grid AO "
            f"state: {reference.e_tot:.12g} != {recomputed_energy:.12g}"
        )
    coordinates = np.asarray(molecule.atom_coords(unit="Bohr"))
    if coordinates.dtype != np.dtype(np.float64) or not np.isfinite(coordinates).all():
        raise DeePHFCapabilityError("the molecular geometry must be finite float64")
    if functional_provenance.components != SUPPORTED_LIBXC_COMPONENTS:
        raise DeePHFCapabilityError("the normalized RKS functional changed during validation")
    if grid_provenance.point_count != reference.grids.weights.size:
        raise DeePHFCapabilityError("the RKS grid changed during validation")
    return reference


def _dft_reference_validation_fingerprint(reference) -> str:
    method = "RKS" if type(reference) is dft.rks.RKS else "UKS"
    _validate_dft_implementations(method)
    grid = reference.grids
    integration = reference._numint
    decorations = tuple(
        bool(getattr(reference, name, None))
        for name in ("with_df", "with_solvent", "with_x2c", "mm_mol", "disp", "penalties")
    )
    digest = hashlib.sha256()
    values = (
        _qualified_name(type(reference)),
        bool(reference.converged),
        rks_molecule_science_fingerprint(reference.mol),
        float(reference.e_tot),
        np.asarray(reference.mo_coeff),
        np.asarray(reference.mo_energy),
        np.asarray(reference.mo_occ),
        reference.xc,
        reference.nlc,
        reference.small_rho_cutoff,
        decorations,
        _qualified_name(type(integration)),
        integration.libxc is libxc,
        str(getattr(integration.libxc, "__version__", None)),
        integration.omega,
        integration.cutoff,
        reference.xc in getattr(integration.libxc, "_CUSTOM_FUNC_R", ()),
        tuple(
            getattr(numint.NumInt, name) is implementation
            for name, implementation in _SUPPORTED_NUMINT_IMPLEMENTATIONS
        ),
        tuple(
            getattr(libxc, name) is implementation
            for name, implementation in _SUPPORTED_LIBXC_IMPLEMENTATIONS
        ),
        (
            tuple(
                (name, owner.__module__, owner.__qualname__)
                for name, owner, _definition in _static_callable_definitions(
                    rks_grad.Gradients, _NATIVE_RKS_GRADIENT_METHODS
                )
            )
            if method == "RKS"
            else ()
        ),
        tuple(sorted(name for name, value in integration.__dict__.items() if callable(value))),
        _qualified_name(type(grid)),
        grid.mol is reference.mol,
        grid.atom_grid,
        _qualified_name(grid.radi_method),
        grid.radi_method is _SUPPORTED_RADI_METHOD,
        _qualified_name(grid.radii_adjust),
        grid.radii_adjust is _SUPPORTED_RADII_ADJUST,
        np.asarray(grid.atomic_radii),
        _qualified_name(grid.becke_scheme),
        grid.becke_scheme is _SUPPORTED_BECKE_SCHEME,
        _qualified_name(rks_grad.grids_response_cc),
        rks_grad.grids_response_cc is _SUPPORTED_GRIDS_RESPONSE,
        grid.prune,
        grid.alignment,
        grid.symmetry,
        grid.cutoff,
        *_grid_arrays(grid),
        tuple(sorted(name for name, value in reference.__dict__.items() if callable(value))),
        tuple(sorted(name for name, value in reference.mol.__dict__.items() if callable(value))),
        tuple(sorted(name for name, value in grid.__dict__.items() if callable(value))),
    )
    for value in values:
        _update_fingerprint_value(digest, value)
    return digest.hexdigest()


def validate_rks_reference(reference):
    """Validate an RKS reference once per unchanged scientific state."""
    if reference_is_transaction_validated(reference):
        return reference
    if type(reference) is not dft.rks.RKS:
        return _audit_rks_reference(reference)
    try:
        fingerprint = _dft_reference_validation_fingerprint(reference)
    except Exception:
        return _audit_rks_reference(reference)
    if _VALIDATED_RKS_REFERENCES.get(reference) == fingerprint:
        return reference
    _audit_rks_reference(reference)
    _VALIDATED_RKS_REFERENCES[reference] = fingerprint
    _GRID_PROVENANCE_CACHE[reference] = (
        fingerprint,
        _GRID_PROVENANCE_CACHE[reference][1],
    )
    return reference


def audit_rks_reference(reference):
    """Run all expensive finite-grid and independent-quadrature checks."""
    result = _audit_rks_reference(reference)
    fingerprint = _dft_reference_validation_fingerprint(reference)
    _VALIDATED_RKS_REFERENCES[reference] = fingerprint
    _GRID_PROVENANCE_CACHE[reference] = (
        fingerprint,
        _GRID_PROVENANCE_CACHE[reference][1],
    )
    return result


def rks_molecule_science_fingerprint(molecule) -> str:
    """Fingerprint stable molecular geometry and AO data."""
    if type(molecule) is not gto_mole.Mole:
        raise DeePHFCapabilityError(
            "RKS science-state fingerprints require a native pyscf.gto.Mole"
        )
    environment = np.asarray(molecule._env).copy()
    environment[: gto_mole.PTR_ENV_START] = 0.0
    digest = hashlib.sha256()
    values = (
        pyscf.__version__,
        _qualified_name(type(molecule)),
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


def rks_reference_fingerprint(reference) -> str:
    """Return a scratch-independent fingerprint of the scientific RKS state."""
    trusted = transaction_reference_fingerprint(reference)
    if trusted is not None:
        return trusted
    return _dft_reference_validation_fingerprint(reference)


def rks_response_integrity_fingerprint(response: RKSResponse) -> str:
    """Return a digest covering every RKS response field except itself."""
    digest = hashlib.sha256()
    for response_field in fields(response):
        if response_field.name == "integrity_fingerprint":
            continue
        value = getattr(response, response_field.name)
        digest.update(response_field.name.encode("utf-8"))
        _update_fingerprint_value(digest, value)
    return digest.hexdigest()


def rks_adjoint_integrity_fingerprint(adjoint: RKSAdjoint) -> str:
    """Return a digest covering every RKS adjoint field except itself."""
    digest = hashlib.sha256()
    for adjoint_field in fields(adjoint):
        if adjoint_field.name == "integrity_fingerprint":
            continue
        value = getattr(adjoint, adjoint_field.name)
        digest.update(adjoint_field.name.encode("utf-8"))
        _update_fingerprint_value(digest, value)
    return digest.hexdigest()


class _RKSLinearResponseProblem:
    """Bind one action-only RKS operator to the reference-neutral protocol."""

    is_self_adjoint = True

    def __init__(self, adapter: "_RKSLinearResponseCore"):
        self._adapter = adapter
        self._state_fingerprint = rks_reference_fingerprint(adapter.reference)
        coefficient, _energy, occupation, occupied, virtual, _gap = adapter._state()
        self._dimension = int(np.count_nonzero(occupied)) * int(
            np.count_nonzero(virtual)
        )

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def operator_fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"pyscf-2.14-rks-occupied-virtual-operator-v1")
        digest.update(self._state_fingerprint.encode("ascii"))
        return digest.hexdigest()

    def _validate_state(self) -> None:
        validate_rks_reference(self._adapter.reference)
        if rks_reference_fingerprint(self._adapter.reference) != self._state_fingerprint:
            raise RKSResponseError(
                "the RKS reference changed after the linear-response problem was built"
            )

    def apply(self, vector: np.ndarray) -> np.ndarray:
        self._validate_state()
        vector = _validated_float64_array(
            vector,
            (self.dimension,),
            "RKS linear-response vector",
        )
        coefficient, energy, occupation, occupied, virtual, _gap = (
            self._adapter._state()
        )
        response = vector.reshape(
            int(np.count_nonzero(virtual)),
            int(np.count_nonzero(occupied)),
        )
        return self._adapter._apply_occupied_virtual_operator(
            response,
            coefficient,
            energy,
            occupation,
            occupied,
            virtual,
        ).reshape(-1)

    def apply_transpose(self, vector: np.ndarray) -> np.ndarray:
        self._validate_state()
        vector = _validated_float64_array(
            vector,
            (self.dimension,),
            "RKS transpose linear-response vector",
        )
        return self.apply(vector)

    def precondition(self, vector: np.ndarray) -> np.ndarray:
        self._validate_state()
        _coefficient, energy, _occupation, occupied, virtual, _gap = (
            self._adapter._state()
        )
        gaps = energy[virtual, None] - energy[occupied]
        return np.asarray(vector).reshape(self.dimension) / gaps.reshape(-1)


class _RKSLinearResponseCore:
    """Provide shared pure-LDA RKS operator and nuclear perturbation primitives."""

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
        self.reference = validate_rks_reference(reference)
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
        positive = (
            self.cphf_tolerance,
            self.residual_tolerance,
            self.invariant_tolerance,
            self.orbital_gap_tolerance,
            self.operator_stability_tolerance,
            self.operator_symmetry_tolerance,
        )
        if any(value <= 0.0 for value in positive):
            raise ValueError("response tolerances must be positive")
        if self.operator_condition_tolerance <= 1.0:
            raise ValueError("operator_condition_tolerance must exceed one")
        if self.max_cycle <= 0 or self.max_refinement_cycles < 0:
            raise ValueError("response cycle limits are invalid")
        if self.operator_dimension_limit <= 0:
            raise ValueError("operator_dimension_limit must be positive")

    @property
    def molecule(self):
        return self.reference.mol

    def _response_atom_indices(self, atom_indices) -> tuple[int, ...]:
        from .gradient import _validate_atom_indices

        selected = _validate_atom_indices(self.molecule, atom_indices)
        return tuple(range(self.molecule.natm)) if selected is None else selected

    def _state(self):
        coefficient = np.asarray(self.reference.mo_coeff)
        energy = np.asarray(self.reference.mo_energy)
        occupation = np.asarray(self.reference.mo_occ)
        occupied = occupation > 0
        virtual = occupation == 0
        gaps = energy[virtual, None] - energy[occupied]
        minimum_gap = float(np.min(gaps))
        if not np.isfinite(minimum_gap) or minimum_gap <= self.orbital_gap_tolerance:
            raise DeePHFCapabilityError(
                "RKS occupied-virtual gap is outside the strict response domain: "
                f"{minimum_gap:.3e} <= {self.orbital_gap_tolerance:.3e}"
            )
        return coefficient, energy, occupation, occupied, virtual, minimum_gap

    def _overlap_derivative(self, atom_indices=None) -> np.ndarray:
        molecule = self.molecule
        atom_indices = self._response_atom_indices(atom_indices)
        try:
            integral = -molecule.intor("int1e_ipovlp", comp=3)
        except Exception as error:
            raise RKSResponseError(
                f"PySCF overlap-derivative construction failed: {error}"
            ) from error
        integral = _validated_float64_array(
            integral,
            (3, molecule.nao, molecule.nao),
            "overlap-derivative integral",
        )
        result = np.zeros((len(atom_indices), 3, molecule.nao, molecule.nao))
        atom_slices = molecule.aoslice_by_atom()
        for result_index, atom_index in enumerate(atom_indices):
            atom_slice = atom_slices[atom_index]
            ao_start, ao_stop = atom_slice[2:]
            result[result_index, :, ao_start:ao_stop] += integral[:, ao_start:ao_stop]
            result[result_index, :, :, ao_start:ao_stop] += integral[
                :, ao_start:ao_stop
            ].transpose(0, 2, 1)
        return result

    @staticmethod
    def _density_from_mo_response(
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
        one_sided = np.einsum(
            "...pi,qi,i->...pq",
            coefficient_response,
            occupied_coefficients,
            occupation[occupied],
        )
        return one_sided + one_sided.swapaxes(-1, -2)

    def _xc_nuclear_derivative_components(
        self,
        density: np.ndarray,
        atom_indices=None,
    ) -> tuple[np.ndarray, np.ndarray]:
        molecule = self.molecule
        atom_indices = self._response_atom_indices(atom_indices)
        result_positions = {
            atom_index: result_index
            for result_index, atom_index in enumerate(atom_indices)
        }
        integration = self.reference._numint
        shape = (len(atom_indices), 3, molecule.nao, molecule.nao)
        grid_coordinate = np.zeros(shape)
        grid_weight = np.zeros(shape)
        atom_grid = _normalized_atom_grid(
            molecule,
            self.reference.grids.atom_grid,
        )
        blocks = _validated_grid_response_blocks(
            self.reference,
            atom_grid,
            audit_weight_derivative=False,
        )
        for host_atom, (coordinates, weights, weight_derivative) in enumerate(blocks):
            coordinates = np.asarray(coordinates)
            weights = np.asarray(weights)
            weight_derivative = np.asarray(weight_derivative)
            if weight_derivative.shape != (molecule.natm, 3, weights.size):
                raise RKSResponseError("the RKS grid-weight derivative shape is invalid")
            try:
                ao = integration.eval_ao(molecule, coordinates, deriv=1)
                values = ao[0]
                gradients = ao[1:4]
                rho = np.einsum(
                    "gp,pq,gq->g",
                    values,
                    density,
                    values,
                    optimize=True,
                )
                xc_values = integration.eval_xc_eff(
                    self.reference.xc,
                    rho,
                    deriv=2,
                    xctype="LDA",
                    spin=0,
                )
                potential = np.asarray(xc_values[1])[0]
                kernel = np.asarray(xc_values[2])[0, 0]
            except Exception as error:
                raise RKSResponseError(
                    f"RKS LDA nuclear quadrature failed: {error}"
                ) from error
            quadrature_values = (
                coordinates,
                weights,
                weight_derivative,
                values,
                gradients,
                rho,
                potential,
                kernel,
            )
            if not all(np.isfinite(value).all() for value in quadrature_values):
                raise RKSResponseError("RKS LDA nuclear quadrature is nonfinite")
            grid_weight += np.einsum(
                "axg,g,gp,gq->axpq",
                weight_derivative[list(atom_indices)],
                potential,
                values,
                values,
                optimize=True,
            )

            def accumulate(target, atom_index, axis, derivative_values):
                density_derivative = np.einsum(
                    "gp,pq,gq->g",
                    derivative_values,
                    density,
                    values,
                    optimize=True,
                )
                density_derivative += np.einsum(
                    "gp,pq,gq->g",
                    values,
                    density,
                    derivative_values,
                    optimize=True,
                )
                target[atom_index, axis] += np.einsum(
                    "g,g,g,gp,gq->pq",
                    weights,
                    kernel,
                    density_derivative,
                    values,
                    values,
                    optimize=True,
                )
                target[atom_index, axis] += np.einsum(
                    "g,g,gp,gq->pq",
                    weights,
                    potential,
                    derivative_values,
                    values,
                    optimize=True,
                )
                target[atom_index, axis] += np.einsum(
                    "g,g,gp,gq->pq",
                    weights,
                    potential,
                    values,
                    derivative_values,
                    optimize=True,
                )

            if host_atom in result_positions:
                result_index = result_positions[host_atom]
                for axis in range(3):
                    accumulate(
                        grid_coordinate,
                        result_index,
                        axis,
                        gradients[axis],
                    )
        return grid_coordinate, grid_weight

    def _hamiltonian_derivative(
        self,
        coefficient: np.ndarray,
        occupation: np.ndarray,
        atom_indices=None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        atom_indices = self._response_atom_indices(atom_indices)
        density = np.asarray(self.reference.make_rdm1(coefficient, occupation))
        expected_shape = (
            len(atom_indices),
            3,
            self.molecule.nao,
            self.molecule.nao,
        )
        try:
            hessian = rks_hessian.Hessian(self.reference)
            fixed_grid_hamiltonian = hessian.make_h1(
                coefficient,
                occupation,
                atmlst=atom_indices,
            )
            fixed_grid_hamiltonian = [
                fixed_grid_hamiltonian[index] for index in atom_indices
            ]
        except Exception as error:
            raise RKSResponseError(
                f"PySCF RKS Hamiltonian derivative construction failed: {error}"
            ) from error
        fixed_grid_hamiltonian = _validated_float64_array(
            fixed_grid_hamiltonian,
            expected_shape,
            "fixed-grid Hamiltonian derivative",
        )
        grid_coordinate, grid_weight = (
            self._xc_nuclear_derivative_components(density, atom_indices)
        )
        full_hamiltonian = fixed_grid_hamiltonian + grid_coordinate + grid_weight
        arrays = (
            full_hamiltonian,
            fixed_grid_hamiltonian,
            grid_coordinate,
            grid_weight,
        )
        if not all(np.isfinite(value).all() for value in arrays):
            raise RKSResponseError("the complete RKS Hamiltonian derivative is nonfinite")
        return arrays

    def _induced_potential_components(
        self,
        density_response: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        perturbation_shape = density_response.shape[:-2]
        flat_density = np.asarray(density_response).reshape(
            -1,
            self.molecule.nao,
            self.molecule.nao,
        )
        if not np.isfinite(flat_density).all():
            raise RKSResponseError("the RKS trial density response is nonfinite")
        symmetry_residual = float(
            np.max(
                np.abs(flat_density - flat_density.swapaxes(-1, -2)),
                initial=0.0,
            )
        )
        if symmetry_residual > 1.0e-10:
            raise RKSResponseError("the RKS trial density response is not symmetric")
        try:
            coulomb, _exchange = scf_hf.get_jk(
                self.molecule,
                flat_density,
                hermi=1,
            )
        except Exception as error:
            raise RKSResponseError(
                f"PySCF direct Coulomb response failed: {error}"
            ) from error
        coulomb = _validated_float64_array(
            coulomb,
            flat_density.shape,
            "induced Coulomb response",
        )
        xc_response = np.zeros_like(flat_density)
        ground_density = np.asarray(self.reference.make_rdm1())
        coordinates = np.asarray(self.reference.grids.coords)
        weights = np.asarray(self.reference.grids.weights)
        integration = self.reference._numint
        block_size = 2048
        try:
            for start in range(0, weights.size, block_size):
                stop = min(start + block_size, weights.size)
                block_coordinates = coordinates[start:stop]
                block_weights = weights[start:stop]
                ao = integration.eval_ao(
                    self.molecule,
                    block_coordinates,
                    deriv=0,
                )
                rho = np.einsum(
                    "gp,pq,gq->g",
                    ao,
                    ground_density,
                    ao,
                    optimize=True,
                )
                kernel = np.asarray(
                    integration.eval_xc_eff(
                        self.reference.xc,
                        rho,
                        deriv=2,
                        xctype="LDA",
                        spin=0,
                    )[2]
                )[0, 0]
                rho_response = np.einsum(
                    "gp,xpq,gq->xg",
                    ao,
                    flat_density,
                    ao,
                    optimize=True,
                )
                xc_response += np.einsum(
                    "g,g,xg,gp,gq->xpq",
                    block_weights,
                    kernel,
                    rho_response,
                    ao,
                    ao,
                    optimize=True,
                )
        except Exception as error:
            raise RKSResponseError(
                f"independent dense LDA f_xc response failed: {error}"
            ) from error
        if not np.isfinite(xc_response).all():
            raise RKSResponseError("the induced LDA f_xc response is nonfinite")
        expected_shape = (*perturbation_shape, self.molecule.nao, self.molecule.nao)
        return coulomb.reshape(expected_shape), xc_response.reshape(expected_shape)

    def _induced_potential(self, density_response: np.ndarray) -> np.ndarray:
        coulomb, xc_response = self._induced_potential_components(density_response)
        return coulomb + xc_response

    def _pyscf_induced_potential(self, density_response: np.ndarray) -> np.ndarray:
        """Apply PySCF's CPKS J + nr_rks_fxc action for the iterative solver."""
        perturbation_shape = density_response.shape[:-2]
        flat_density = np.asarray(density_response).reshape(
            -1,
            self.molecule.nao,
            self.molecule.nao,
        )
        try:
            coulomb, _exchange = scf_hf.get_jk(
                self.molecule,
                flat_density,
                hermi=1,
            )
            xc_response = self.reference._numint.nr_rks_fxc(
                self.molecule,
                self.reference.grids,
                self.reference.xc,
                np.asarray(self.reference.make_rdm1()),
                flat_density,
                hermi=1,
            )
        except Exception as error:
            raise RKSResponseError(
                f"PySCF iterative J + LDA f_xc action failed: {error}"
            ) from error
        expected = flat_density.shape
        coulomb = _validated_float64_array(
            coulomb,
            expected,
            "PySCF iterative Coulomb response",
        )
        xc_response = _validated_float64_array(
            xc_response,
            expected,
            "PySCF iterative LDA f_xc response",
        )
        return (coulomb + xc_response).reshape(
            *perturbation_shape,
            self.molecule.nao,
            self.molecule.nao,
        )

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

    def _pyscf_induced_mo_potential(
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
        induced = self._pyscf_induced_potential(density_response)
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
        nocc = int(np.count_nonzero(occupied))
        nvir = int(np.count_nonzero(virtual))
        response = np.asarray(occupied_virtual_response)
        if response.shape[-2:] != (nvir, nocc):
            raise RKSResponseError("the RKS occupied-virtual response shape is invalid")
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
        gaps = energy[virtual, None] - energy[occupied]
        return gaps * response + induced

    def _response_operator_matrix_and_diagnostics(
        self,
        coefficient: np.ndarray,
        energy: np.ndarray,
        occupation: np.ndarray,
        occupied: np.ndarray,
        virtual: np.ndarray,
    ) -> tuple[np.ndarray, int, float, float, float, float, float]:
        nocc = int(np.count_nonzero(occupied))
        nvir = int(np.count_nonzero(virtual))
        dimension = nocc * nvir
        if dimension > self.operator_dimension_limit:
            raise DeePHFCapabilityError(
                "RKS occupied-virtual response dimension exceeds the explicit "
                f"condition-audit limit: {dimension} > {self.operator_dimension_limit}"
            )
        identity = np.eye(dimension, dtype=np.float64)
        matrix = np.empty((dimension, dimension), dtype=np.float64)
        reconstruction_residual = 0.0
        reference_response = self.reference.gen_response(
            coefficient,
            occupation,
            hermi=1,
        )
        batch_size = min(32, dimension)
        for start in range(0, dimension, batch_size):
            stop = min(start + batch_size, dimension)
            roots = identity[start:stop].reshape(-1, nvir, nocc)
            images = self._apply_occupied_virtual_operator(
                roots,
                coefficient,
                energy,
                occupation,
                occupied,
                virtual,
            )
            matrix[:, start:stop] = images.reshape(stop - start, dimension).T
            full_roots = np.zeros(
                (stop - start, coefficient.shape[1], nocc),
                dtype=np.float64,
            )
            full_roots[:, virtual] = roots
            density_roots = self._density_from_mo_response(
                full_roots,
                coefficient,
                occupation,
                occupied,
            )
            independent = self._induced_potential(density_roots)
            try:
                pyscf_response = np.asarray(reference_response(density_roots))
            except Exception as error:
                raise RKSResponseError(
                    f"PySCF RKS induced-response reconstruction failed: {error}"
                ) from error
            pyscf_response = _validated_float64_array(
                pyscf_response,
                density_roots.shape,
                "PySCF induced RKS response",
            )
            reconstruction_residual = max(
                reconstruction_residual,
                float(
                    np.max(
                        np.abs(independent - pyscf_response),
                        initial=0.0,
                    )
                ),
            )
        if not np.isfinite(matrix).all():
            raise RKSResponseError("the RKS occupied-virtual response operator is nonfinite")
        if reconstruction_residual > self.invariant_tolerance:
            raise RKSResponseError(
                "the independent direct-J plus dense-LDA response does not match "
                f"PySCF: residual {reconstruction_residual:.3e}"
            )
        symmetry_residual = float(
            np.max(np.abs(matrix - matrix.T), initial=0.0)
        )
        if symmetry_residual > self.operator_symmetry_tolerance:
            raise RKSResponseError(
                "the RKS occupied-virtual response operator violates symmetry: "
                f"{symmetry_residual:.3e} > {self.operator_symmetry_tolerance:.3e}"
            )
        try:
            eigenvalues = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
        except np.linalg.LinAlgError as error:
            raise RKSResponseError(
                f"the RKS response-operator eigensolve failed: {error}"
            ) from error
        minimum_eigenvalue = float(eigenvalues[0])
        maximum_eigenvalue = float(eigenvalues[-1])
        if minimum_eigenvalue <= self.operator_stability_tolerance:
            raise DeePHFCapabilityError(
                "the RKS occupied-virtual response operator is unstable or singular: "
                f"minimum eigenvalue {minimum_eigenvalue:.3e} <= "
                f"{self.operator_stability_tolerance:.3e}"
            )
        condition_number = maximum_eigenvalue / minimum_eigenvalue
        if (
            not np.isfinite(condition_number)
            or condition_number > self.operator_condition_tolerance
        ):
            raise DeePHFCapabilityError(
                "the RKS occupied-virtual response operator is ill conditioned: "
                f"{condition_number:.3e} > {self.operator_condition_tolerance:.3e}"
            )
        return (
            matrix,
            dimension,
            minimum_eigenvalue,
            maximum_eigenvalue,
            float(condition_number),
            symmetry_residual,
            reconstruction_residual,
        )

    def validate_response_operator_exact(
        self,
    ) -> tuple[int, float, float, float, float, float]:
        """Run an explicit dense stability audit for a bounded debug problem."""
        coefficient, energy, occupation, occupied, virtual, _gap = self._state()
        return self._response_operator_matrix_and_diagnostics(
            coefficient,
            energy,
            occupation,
            occupied,
            virtual,
        )[1:]

    def linear_response_problem(self) -> ScalarAdjointProblem:
        """Return the action-only RKS operator through the neutral protocol."""
        validate_rks_reference(self.reference)
        problem = _RKSLinearResponseProblem(self)
        if not isinstance(problem, ScalarAdjointProblem):
            raise RKSResponseError(
                "the RKS linear-response problem violates the neutral adjoint protocol"
            )
        return problem


class RKSResponseAdapter(_RKSLinearResponseCore):
    """Solve and independently audit molecular pure-LDA RKS nuclear CPKS."""

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
        induced = self._induced_mo_potential(
            mo_response,
            coefficient,
            occupation,
            occupied,
        )
        residual = (
            hamiltonian_mo
            + induced
            - overlap_mo * energy[occupied]
            + (energy[:, None] - energy[occupied]) * mo_response
        )
        return residual[..., virtual, :]

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
        nvir = int(np.count_nonzero(virtual))
        flattened_hamiltonian = hamiltonian_mo.reshape(-1, nmo, nocc)
        flattened_overlap = overlap_mo.reshape(-1, nmo, nocc)

        def solver_induced_full(response):
            response = np.asarray(response).reshape(-1, nmo, nocc)
            return self._pyscf_induced_mo_potential(
                response,
                coefficient,
                occupation,
                occupied,
            )

        def physical_induced_full(response):
            response = np.asarray(response).reshape(-1, nmo, nocc)
            return self._induced_mo_potential(
                response,
                coefficient,
                occupation,
                occupied,
            )

        try:
            response, _ = cphf.solve(
                solver_induced_full,
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
            raise RKSResponseError(f"PySCF RKS CPKS solve failed: {error}") from error
        response = _validated_float64_array(
            response,
            flattened_hamiltonian.shape,
            "PySCF RKS CPKS response",
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
            flat_residual = residual.reshape(-1, nvir, nocc)
            root_scales = np.linalg.norm(
                flat_residual.reshape(len(flat_residual), -1),
                axis=1,
            )
            active = root_scales > np.finfo(float).eps
            correction = np.zeros_like(flat_residual)

            def induced_virtual(virtual_response):
                virtual_response = np.asarray(virtual_response).reshape(-1, nvir, nocc)
                full_response = np.zeros(
                    (len(virtual_response), nmo, nocc),
                    dtype=np.float64,
                )
                full_response[:, virtual] = virtual_response
                return physical_induced_full(full_response)[:, virtual]

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
                    raise RKSResponseError(
                        f"PySCF RKS CPKS residual refinement failed: {error}"
                    ) from error
                normalized_correction = _validated_float64_array(
                    normalized_correction,
                    flat_residual[active].shape,
                    "PySCF RKS CPKS refinement response",
                )
                correction[active] = (
                    normalized_correction * root_scales[active, None, None]
                )
                response[..., virtual, :] += correction.reshape(
                    *perturbation_shape,
                    nvir,
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
            residual_history.append(float(np.max(np.abs(residual), initial=0.0)))
        return response, residual, tuple(residual_history)

    def solve(self, atom_indices=None) -> RKSResponse:
        """Return the audited finite-grid response for selected atoms."""
        return self._solve(atom_indices, "response")

    def _solve_with_density_partitions(self, atom_indices=None):
        """Return a response and its transient AO density work arrays."""
        return self._solve(atom_indices, "partitions")

    def _solve_for_gradient(self, objective, atom_indices=None):
        """Return compact diagnostics and the final density contraction."""
        return self._solve(atom_indices, "gradient", objective)

    def _solve(self, atom_indices, result_mode, objective=None):
        validate_rks_reference(self.reference)
        atom_indices = self._response_atom_indices(atom_indices)
        initial_fingerprint = rks_reference_fingerprint(self.reference)
        functional_provenance = _functional_provenance(self.reference)
        grid_provenance = _grid_provenance(self.reference)
        coefficient, energy, occupation, occupied, virtual, minimum_gap = self._state()
        response_dimension = int(np.count_nonzero(occupied)) * int(
            np.count_nonzero(virtual)
        )
        overlap = np.asarray(self.reference.get_ovlp())
        overlap_derivative = self._overlap_derivative(atom_indices)
        (
            hamiltonian_derivative,
            hamiltonian_derivative_fixed_grid,
            xc_grid_coordinate,
            xc_grid_weight,
        ) = self._hamiltonian_derivative(coefficient, occupation, atom_indices)
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
        if result_mode == "partitions":
            metric_response = np.zeros_like(mo_response)
            metric_response[..., occupied, :] = mo_response[..., occupied, :]
            occupied_virtual_response = np.zeros_like(mo_response)
            occupied_virtual_response[..., virtual, :] = mo_response[..., virtual, :]
            density_metric = self._density_from_mo_response(
                metric_response, coefficient, occupation, occupied
            )
            density_occupied_virtual = self._density_from_mo_response(
                occupied_virtual_response, coefficient, occupation, occupied
            )
            density_response = density_metric + density_occupied_virtual
        else:
            density_response = self._density_from_mo_response(
                mo_response, coefficient, occupation, occupied
            )
        hamiltonian_reconstruction_residual = float(
            np.max(
                np.abs(
                    hamiltonian_derivative
                    - hamiltonian_derivative_fixed_grid
                    - xc_grid_coordinate
                    - xc_grid_weight
                ),
                initial=0.0,
            )
        )
        overlap_occupied = overlap_mo[..., occupied, :]
        occupied_occupied_response = mo_response[..., occupied, :]
        metric_residual = max(
            float(
                np.max(
                    np.abs(
                        occupied_occupied_response
                        + occupied_occupied_response.swapaxes(-1, -2)
                        + overlap_occupied
                    ),
                    initial=0.0,
                )
            ),
            float(
                np.max(
                    np.abs(occupied_occupied_response + 0.5 * overlap_occupied),
                    initial=0.0,
                )
            ),
        )
        density_ground = np.asarray(self.reference.make_rdm1())
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
        translation_residual = (
            float(np.max(np.abs(np.sum(density_response, axis=0)), initial=0.0))
            if len(atom_indices) == self.molecule.natm
            else None
        )
        try:
            ao = self.reference._numint.eval_ao(
                self.molecule,
                self.reference.grids.coords,
                deriv=0,
            )
            rho = np.einsum(
                "gp,pq,gq->g",
                ao,
                density_ground,
                ao,
                optimize=True,
            )
            quadrature_electron_count = float(
                np.dot(self.reference.grids.weights, rho)
            )
        except Exception as error:
            raise RKSResponseError(
                f"RKS quadrature electron-count audit failed: {error}"
            ) from error
        diagnostics = RKSResponseDiagnostics(
            minimum_orbital_gap=minimum_gap,
            pyscf_version=pyscf.__version__,
            libxc_version=str(libxc.__version__),
            functional_components=functional_provenance.components,
            grid_point_count=grid_provenance.point_count,
            grid_coordinates_fingerprint=grid_provenance.coordinates_fingerprint,
            grid_weights_fingerprint=grid_provenance.weights_fingerprint,
            quadrature_electron_count=quadrature_electron_count,
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
            hamiltonian_reconstruction_residual=hamiltonian_reconstruction_residual,
            metric_residual=metric_residual,
            idempotency_residual=float(np.max(np.abs(idempotency), initial=0.0)),
            particle_number_residual=float(
                np.max(np.abs(particle_number), initial=0.0)
            ),
            translation_residual=translation_residual,
            refinement_cycles=len(residual_history) - 1,
            residual_history=residual_history,
        )
        arrays = {
            "mo_response": mo_response,
            "density_response": density_response,
            "overlap_derivative": overlap_derivative,
            "hamiltonian_derivative": hamiltonian_derivative,
            "hamiltonian_derivative_fixed_grid": hamiltonian_derivative_fixed_grid,
            "xc_hamiltonian_derivative_grid_coordinate": xc_grid_coordinate,
            "xc_hamiltonian_derivative_grid_weight": xc_grid_weight,
            "orbital_response_residual": residual,
        }
        if not all(np.isfinite(value).all() for value in arrays.values()):
            raise RKSResponseError("the RKS response contains nonfinite arrays")
        diagnostic_values = tuple(
            value
            for value in diagnostics.__dict__.values()
            if isinstance(value, (float, int))
        ) + diagnostics.residual_history
        if not np.isfinite(diagnostic_values).all():
            raise RKSResponseError("the RKS response diagnostics are nonfinite")
        if diagnostics.maximum_residual > self.residual_tolerance:
            history = " -> ".join(f"{value:.3e}" for value in residual_history)
            raise RKSResponseError(
                "RKS response residual exceeds tolerance: "
                f"{diagnostics.maximum_residual:.3e} > {self.residual_tolerance:.3e}; "
                f"refinement history: {history}"
            )
        invariant_failures = {
            "Hamiltonian reconstruction": diagnostics.hamiltonian_reconstruction_residual,
            "metric": diagnostics.metric_residual,
            "idempotency": diagnostics.idempotency_residual,
            "particle number": diagnostics.particle_number_residual,
            "translation": diagnostics.translation_residual,
        }
        invariant_failures = {
            name: value
            for name, value in invariant_failures.items()
            if value is not None and value > self.invariant_tolerance
        }
        if invariant_failures:
            details = ", ".join(
                f"{name}={value:.3e}" for name, value in invariant_failures.items()
            )
            raise RKSResponseError(
                "RKS response invariant exceeds tolerance "
                f"{self.invariant_tolerance:.3e}: {details}"
            )
        validate_rks_reference(self.reference)
        if rks_reference_fingerprint(self.reference) != initial_fingerprint:
            raise RKSResponseError("the RKS reference changed during the response solve")
        if result_mode == "gradient":
            return diagnostics, np.einsum("ij,bxij->bx", objective, density_response)
        response = RKSResponse(
            reference_identity=id(self.reference),
            state_fingerprint=initial_fingerprint,
            integrity_fingerprint="",
            atom_indices=atom_indices,
            mo_response=_immutable_array(mo_response),
            _mo_coefficients=_immutable_array(coefficient),
            _mo_occupations=_immutable_array(occupation),
            overlap_derivative=_immutable_array(overlap_derivative),
            hamiltonian_derivative=_immutable_array(hamiltonian_derivative),
            orbital_response_residual=_immutable_array(residual),
            functional_provenance=functional_provenance,
            grid_provenance=grid_provenance,
            hamiltonian_derivative_fixed_grid=_immutable_array(
                hamiltonian_derivative_fixed_grid
            ),
            xc_hamiltonian_derivative_grid_coordinate=_immutable_array(
                xc_grid_coordinate
            ),
            xc_hamiltonian_derivative_grid_weight=_immutable_array(xc_grid_weight),
            diagnostics=diagnostics,
        )
        response = replace(
            response,
            integrity_fingerprint=rks_response_integrity_fingerprint(response),
        )
        return (
            (response, (density_response, density_metric, density_occupied_virtual))
            if result_mode == "partitions"
            else response
        )

    def audit_response_equations(self, response: RKSResponse) -> None:
        """Independently rebuild every supplied equation without another solve."""
        validate_rks_reference(self.reference)
        if type(response) is not RKSResponse:
            raise RKSResponseError("the supplied RKS response has an invalid type")
        if type(response.diagnostics) is not RKSResponseDiagnostics:
            raise RKSResponseError(
                "the supplied RKS response diagnostics have an invalid type"
            )
        if response.reference_identity != id(self.reference):
            raise RKSResponseError("the supplied RKS response belongs to another reference")
        if response.state_fingerprint != rks_reference_fingerprint(self.reference):
            raise RKSResponseError("the supplied RKS response state is stale")
        if response.integrity_fingerprint != rks_response_integrity_fingerprint(response):
            raise RKSResponseError("the supplied RKS response integrity check failed")
        functional_provenance = _functional_provenance(self.reference)
        grid_provenance = _grid_provenance(self.reference)
        if (
            type(response.functional_provenance) is not RKSFunctionalProvenance
            or response.functional_provenance != functional_provenance
        ):
            raise RKSResponseError(
                "the supplied RKS response functional provenance is invalid"
            )
        if (
            type(response.grid_provenance) is not RKSGridProvenance
            or response.grid_provenance != grid_provenance
        ):
            raise RKSResponseError("the supplied RKS response grid provenance is invalid")
        coefficient, energy, occupation, occupied, virtual, minimum_gap = self._state()
        nmo = coefficient.shape[1]
        nocc = int(np.count_nonzero(occupied))
        nvir = int(np.count_nonzero(virtual))
        atom_indices = self._response_atom_indices(response.atom_indices)
        if atom_indices != response.atom_indices:
            raise RKSResponseError("the supplied RKS response atom selection is invalid")
        perturbation_shape = (len(atom_indices), 3)
        mo_shape = (*perturbation_shape, nmo, nocc)
        coefficient_shape = (*perturbation_shape, self.molecule.nao, nocc)
        density_shape = (
            *perturbation_shape,
            self.molecule.nao,
            self.molecule.nao,
        )
        residual_shape = (*perturbation_shape, nvir, nocc)
        expected_shapes = {
            "mo_response": mo_shape,
            "mo_response_occupied_virtual": mo_shape,
            "mo_response_metric": mo_shape,
            "coefficient_response": coefficient_shape,
            "coefficient_response_occupied_virtual": coefficient_shape,
            "coefficient_response_metric": coefficient_shape,
            "density_response": density_shape,
            "density_response_occupied_virtual": density_shape,
            "density_response_metric": density_shape,
            "overlap_derivative": density_shape,
            "hamiltonian_derivative": density_shape,
            "hamiltonian_derivative_fixed_grid": density_shape,
            "xc_hamiltonian_derivative_grid_coordinate": density_shape,
            "xc_hamiltonian_derivative_grid_weight": density_shape,
            "orbital_response_residual": residual_shape,
        }
        for name, expected_shape in expected_shapes.items():
            value = getattr(response, name)
            if type(value) is not np.ndarray or value.flags.writeable:
                raise RKSResponseError(
                    f"the supplied RKS response {name} must be an immutable ndarray"
                )
            _validated_float64_array(value, expected_shape, f"supplied {name}")
        expected_overlap_derivative = self._overlap_derivative(atom_indices)
        (
            expected_hamiltonian_derivative,
            expected_hamiltonian_fixed_grid,
            expected_xc_grid_coordinate,
            expected_xc_grid_weight,
        ) = self._hamiltonian_derivative(coefficient, occupation, atom_indices)
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
            (
                response.hamiltonian_derivative_fixed_grid,
                expected_hamiltonian_fixed_grid,
                "fixed-grid Hamiltonian derivative",
            ),
            (
                response.xc_hamiltonian_derivative_grid_coordinate,
                expected_xc_grid_coordinate,
                "grid-coordinate XC Hamiltonian derivative",
            ),
            (
                response.xc_hamiltonian_derivative_grid_weight,
                expected_xc_grid_weight,
                "grid-weight XC Hamiltonian derivative",
            ),
        )
        for stored, expected, name in derivative_fields:
            if not np.allclose(stored, expected, rtol=0.0, atol=1.0e-11):
                raise RKSResponseError(
                    f"the supplied RKS response {name} is not independently reproducible"
                )
        mo_partition_residual = float(
            np.max(
                np.abs(
                    response.mo_response
                    - response.mo_response_metric
                    - response.mo_response_occupied_virtual
                ),
                initial=0.0,
            )
        )
        if (
            np.max(
                np.abs(response.mo_response_metric[..., virtual, :]),
                initial=0.0,
            )
            > 1.0e-12
            or np.max(
                np.abs(response.mo_response_occupied_virtual[..., occupied, :]),
                initial=0.0,
            )
            > 1.0e-12
            or mo_partition_residual > 1.0e-12
        ):
            raise RKSResponseError("the supplied RKS MO response partition is invalid")
        rebuilt_coefficients = {
            "coefficient_response": np.einsum(
                "mp,...pi->...mi",
                coefficient,
                response.mo_response,
            ),
            "coefficient_response_occupied_virtual": np.einsum(
                "mp,...pi->...mi",
                coefficient,
                response.mo_response_occupied_virtual,
            ),
            "coefficient_response_metric": np.einsum(
                "mp,...pi->...mi",
                coefficient,
                response.mo_response_metric,
            ),
        }
        for name, rebuilt in rebuilt_coefficients.items():
            if not np.allclose(getattr(response, name), rebuilt, rtol=0.0, atol=1.0e-11):
                raise RKSResponseError(
                    f"the supplied RKS response {name} does not follow from its MO response"
                )
        rebuilt_densities = {
            "density_response": self._density_from_mo_response(
                response.mo_response,
                coefficient,
                occupation,
                occupied,
            ),
            "density_response_occupied_virtual": self._density_from_mo_response(
                response.mo_response_occupied_virtual,
                coefficient,
                occupation,
                occupied,
            ),
            "density_response_metric": self._density_from_mo_response(
                response.mo_response_metric,
                coefficient,
                occupation,
                occupied,
            ),
        }
        for name, rebuilt in rebuilt_densities.items():
            if not np.allclose(getattr(response, name), rebuilt, rtol=0.0, atol=1.0e-11):
                raise RKSResponseError(
                    f"the supplied RKS response {name} does not follow from its MO response"
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
        physical_residual = self._orbital_residual(
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
            physical_residual,
            rtol=0.0,
            atol=1.0e-11,
        ):
            raise RKSResponseError(
                "the supplied RKS orbital residual is not independently reproducible"
            )
        response_dimension = int(np.count_nonzero(occupied)) * int(
            np.count_nonzero(virtual)
        )
        overlap = np.asarray(self.reference.get_ovlp())
        density_ground = np.asarray(self.reference.make_rdm1())
        overlap_occupied = overlap_mo[..., occupied, :]
        occupied_occupied_response = response.mo_response[..., occupied, :]
        metric_residual = max(
            float(
                np.max(
                    np.abs(
                        occupied_occupied_response
                        + occupied_occupied_response.swapaxes(-1, -2)
                        + overlap_occupied
                    ),
                    initial=0.0,
                )
            ),
            float(
                np.max(
                    np.abs(occupied_occupied_response + 0.5 * overlap_occupied),
                    initial=0.0,
                )
            ),
        )
        idempotency = (
            np.einsum(
                "...ij,jk,kl->...il",
                response.density_response,
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
                response.density_response,
            )
            - 2.0 * response.density_response
        )
        particle_number = (
            np.einsum("...ij,ji->...", response.density_response, overlap)
            + np.einsum(
                "ij,...ji->...",
                density_ground,
                expected_overlap_derivative,
            )
        )
        translation_residual = (
            float(
                np.max(
                    np.abs(np.sum(response.density_response, axis=0)),
                    initial=0.0,
                )
            )
            if len(atom_indices) == self.molecule.natm
            else None
        )
        hamiltonian_reconstruction_residual = float(
            np.max(
                np.abs(
                    expected_hamiltonian_derivative
                    - expected_hamiltonian_fixed_grid
                    - expected_xc_grid_coordinate
                    - expected_xc_grid_weight
                ),
                initial=0.0,
            )
        )
        try:
            ao = self.reference._numint.eval_ao(
                self.molecule,
                self.reference.grids.coords,
                deriv=0,
            )
            rho = np.einsum(
                "gp,pq,gq->g",
                ao,
                density_ground,
                ao,
                optimize=True,
            )
            quadrature_electron_count = float(
                np.dot(self.reference.grids.weights, rho)
            )
        except Exception as error:
            raise RKSResponseError(
                f"RKS supplied-response quadrature audit failed: {error}"
            ) from error
        measured = {
            "minimum_orbital_gap": minimum_gap,
            "response_dimension": response_dimension,
            "hamiltonian_reconstruction_residual": (
                hamiltonian_reconstruction_residual
            ),
            "metric_residual": metric_residual,
            "idempotency_residual": float(
                np.max(np.abs(idempotency), initial=0.0)
            ),
            "particle_number_residual": float(
                np.max(np.abs(particle_number), initial=0.0)
            ),
            "translation_residual": translation_residual,
            "maximum_residual": float(
                np.max(np.abs(physical_residual), initial=0.0)
            ),
            "residual_rms": float(np.sqrt(np.mean(np.square(physical_residual)))),
            "quadrature_electron_count": quadrature_electron_count,
        }
        diagnostics = response.diagnostics
        if diagnostics.operator_is_self_adjoint is not True:
            raise RKSResponseError("the supplied RKS response operator contract is invalid")
        exact_diagnostics = {
            "pyscf_version": pyscf.__version__,
            "libxc_version": str(libxc.__version__),
            "functional_components": functional_provenance.components,
            "grid_point_count": grid_provenance.point_count,
            "grid_coordinates_fingerprint": grid_provenance.coordinates_fingerprint,
            "grid_weights_fingerprint": grid_provenance.weights_fingerprint,
            "cphf_tolerance": self.cphf_tolerance,
            "residual_tolerance": self.residual_tolerance,
            "invariant_tolerance": self.invariant_tolerance,
            "orbital_gap_tolerance": self.orbital_gap_tolerance,
            "max_cycle": self.max_cycle,
            "max_refinement_cycles": self.max_refinement_cycles,
            "level_shift": self.level_shift,
        }
        for name, expected in exact_diagnostics.items():
            if getattr(diagnostics, name) != expected:
                raise RKSResponseError(
                    f"the supplied RKS response diagnostic {name} is invalid"
                )
        for name, expected in measured.items():
            if expected is None:
                if getattr(diagnostics, name) is not None:
                    raise RKSResponseError(
                        f"the supplied RKS response diagnostic {name} is not reproducible"
                    )
                continue
            if not np.isclose(
                getattr(diagnostics, name),
                expected,
                rtol=0.0,
                atol=1.0e-11,
            ):
                raise RKSResponseError(
                    f"the supplied RKS response diagnostic {name} is not reproducible"
                )
        history = diagnostics.residual_history
        if (
            type(history) is not tuple
            or not history
            or diagnostics.refinement_cycles != len(history) - 1
            or not np.isfinite(history).all()
            or any(value < 0.0 for value in history)
            or any(
                later > earlier + self.residual_tolerance
                for earlier, later in zip(history, history[1:])
            )
            or not np.isclose(
                history[-1],
                measured["maximum_residual"],
                rtol=0.0,
                atol=1.0e-11,
            )
        ):
            raise RKSResponseError(
                "the supplied RKS response residual-refinement history is invalid"
            )
        if measured["maximum_residual"] > self.residual_tolerance:
            raise RKSResponseError(
                "the supplied RKS response physical residual exceeds tolerance"
            )
        invariant_names = (
            "hamiltonian_reconstruction_residual",
            "metric_residual",
            "idempotency_residual",
            "particle_number_residual",
            "translation_residual",
        )
        if max(
            measured[name]
            for name in invariant_names
            if measured[name] is not None
        ) > self.invariant_tolerance:
            raise RKSResponseError(
                "the supplied RKS response invariant exceeds tolerance"
            )


class RKSAdjointAdapter(_RKSLinearResponseCore):
    """Solve one correction-specific pure-LDA RKS scalar adjoint."""

    def __init__(
        self,
        reference,
        *,
        residual_tolerance: float = 1.0e-9,
        invariant_tolerance: float = 1.0e-9,
        orbital_gap_tolerance: float = 1.0e-7,
        operator_stability_tolerance: float = 1.0e-6,
        operator_condition_tolerance: float = 1.0e8,
        operator_symmetry_tolerance: float = 1.0e-10,
        operator_dimension_limit: int = 512,
        objective_symmetry_tolerance: float = 1.0e-10,
        max_cycle: int = 100,
        krylov_restart: int = 50,
    ):
        super().__init__(
            reference,
            residual_tolerance=residual_tolerance,
            invariant_tolerance=invariant_tolerance,
            orbital_gap_tolerance=orbital_gap_tolerance,
            operator_stability_tolerance=operator_stability_tolerance,
            operator_condition_tolerance=operator_condition_tolerance,
            operator_symmetry_tolerance=operator_symmetry_tolerance,
            operator_dimension_limit=operator_dimension_limit,
            max_cycle=max_cycle,
        )
        self.krylov_restart = _cycle_limit(
            krylov_restart,
            "krylov_restart",
        )
        self.objective_symmetry_tolerance = _response_real_control(
            objective_symmetry_tolerance,
            "objective_symmetry_tolerance",
        )
        if self.objective_symmetry_tolerance <= 0.0:
            raise ValueError("adjoint objective_symmetry_tolerance must be positive")
        if self.krylov_restart <= 0:
            raise ValueError("adjoint krylov_restart must be positive")

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
            raise RKSAdjointError(
                "the correction AO objective potential violates symmetry: "
                f"{symmetry_residual:.3e} > "
                f"{self.objective_symmetry_tolerance:.3e}"
            )
        return potential

    @staticmethod
    def _audited_array(value, expected_shape, name: str) -> np.ndarray:
        if type(value) is not np.ndarray:
            raise RKSAdjointError(
                f"the supplied RKS adjoint field {name} has an invalid type"
            )
        if value.shape != expected_shape:
            raise RKSAdjointError(
                f"the supplied RKS adjoint field {name} has shape {value.shape}; "
                f"expected {expected_shape}"
            )
        if value.dtype != np.dtype(np.float64) or np.iscomplexobj(value):
            raise RKSAdjointError(
                f"the supplied RKS adjoint field {name} must use real numpy.float64"
            )
        if not np.isfinite(value).all():
            raise RKSAdjointError(
                f"the supplied RKS adjoint field {name} must be finite"
            )
        if value.flags.writeable:
            raise RKSAdjointError(
                f"the supplied RKS adjoint field {name} must be immutable"
            )
        return value

    @staticmethod
    def _require_close(stored, expected, name: str) -> None:
        if not np.allclose(stored, expected, rtol=1.0e-11, atol=1.0e-12):
            maximum_residual = float(
                np.max(np.abs(np.asarray(stored) - np.asarray(expected)), initial=0.0)
            )
            raise RKSAdjointError(
                f"the supplied RKS adjoint {name} is inconsistent: "
                f"residual {maximum_residual:.3e}"
            )

    @staticmethod
    def _residual_statistics(value: np.ndarray) -> tuple[float, float]:
        return (
            float(np.max(np.abs(value), initial=0.0)),
            float(np.sqrt(np.mean(np.square(value)))),
        )

    def _adjoint_density(
        self,
        zvector: np.ndarray,
        coefficient: np.ndarray,
        occupation: np.ndarray,
        occupied: np.ndarray,
        virtual: np.ndarray,
    ) -> np.ndarray:
        occupied_coefficients = coefficient[:, occupied]
        virtual_coefficients = coefficient[:, virtual]
        rotated_occupied = virtual_coefficients @ zvector
        one_sided = rotated_occupied @ (
            occupied_coefficients * occupation[occupied]
        ).T
        density = one_sided + one_sided.T
        return _validated_float64_array(
            density,
            (self.molecule.nao, self.molecule.nao),
            "RKS adjoint AO density",
        )

    def _gradient_partitions(
        self,
        objective_ao_potential: np.ndarray,
        zvector: np.ndarray,
        adjoint_ao_potential: np.ndarray,
        coefficient: np.ndarray,
        energy: np.ndarray,
        occupation: np.ndarray,
        occupied: np.ndarray,
        virtual: np.ndarray,
        atom_indices=None,
        compact=False,
    ):
        atom_indices = self._response_atom_indices(atom_indices)
        overlap_derivative = self._overlap_derivative(atom_indices)
        (
            hamiltonian_derivative,
            hamiltonian_fixed_grid,
            hamiltonian_grid_coordinate,
            hamiltonian_grid_weight,
        ) = self._hamiltonian_derivative(coefficient, occupation, atom_indices)
        hamiltonian_reconstruction_residual = float(
            np.max(
                np.abs(
                    hamiltonian_derivative
                    - hamiltonian_fixed_grid
                    - hamiltonian_grid_coordinate
                    - hamiltonian_grid_weight
                ),
                initial=0.0,
            )
        )
        occupied_coefficients = coefficient[:, occupied]

        def occupied_mo(value):
            return np.einsum(
                "mp,...mn,ni->...pi",
                coefficient,
                value,
                occupied_coefficients,
            )

        overlap_mo = occupied_mo(overlap_derivative)
        if compact:
            full_rhs = (
                occupied_mo(hamiltonian_derivative)[..., virtual, :]
                - overlap_mo[..., virtual, :] * energy[occupied]
            )
            correction_gradient_adjoint_nuclear = -np.einsum(
                "ai,...ai->...", zvector, full_rhs
            )
        else:
            fixed_grid_mo = occupied_mo(hamiltonian_fixed_grid)
            grid_coordinate_mo = occupied_mo(hamiltonian_grid_coordinate)
            grid_weight_mo = occupied_mo(hamiltonian_grid_weight)
            fixed_grid_rhs = (
                fixed_grid_mo[..., virtual, :]
                - overlap_mo[..., virtual, :] * energy[occupied]
            )
            correction_gradient_adjoint_fixed_grid = -np.einsum(
                "ai,...ai->...", zvector, fixed_grid_rhs
            )
            correction_gradient_adjoint_grid_coordinate = -np.einsum(
                "ai,...ai->...", zvector, grid_coordinate_mo[..., virtual, :]
            )
            correction_gradient_adjoint_grid_weight = -np.einsum(
                "ai,...ai->...", zvector, grid_weight_mo[..., virtual, :]
            )
            correction_gradient_adjoint_nuclear = (
                correction_gradient_adjoint_fixed_grid
                + correction_gradient_adjoint_grid_coordinate
                + correction_gradient_adjoint_grid_weight
            )
        objective_mo = coefficient.T @ objective_ao_potential @ coefficient
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
            adjoint_potential_occupied
            + adjoint_potential_occupied.T
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
        correction_gradient_response = (
            correction_gradient_metric
            + correction_gradient_adjoint_nuclear
            + correction_gradient_adjoint_metric
        )
        if compact:
            _validated_float64_array(
                correction_gradient_response,
                (len(atom_indices), 3),
                "RKS correction_gradient_response",
            )
            return correction_gradient_response, hamiltonian_reconstruction_residual
        correction_gradient_occupied_virtual = (
            correction_gradient_adjoint_nuclear
            + correction_gradient_adjoint_metric
        )
        partitions = {
            "correction_gradient_metric": correction_gradient_metric,
            "correction_gradient_adjoint_fixed_grid": (
                correction_gradient_adjoint_fixed_grid
            ),
            "correction_gradient_adjoint_grid_coordinate": (
                correction_gradient_adjoint_grid_coordinate
            ),
            "correction_gradient_adjoint_grid_weight": (
                correction_gradient_adjoint_grid_weight
            ),
            "correction_gradient_adjoint_nuclear": (
                correction_gradient_adjoint_nuclear
            ),
            "correction_gradient_adjoint_metric": (
                correction_gradient_adjoint_metric
            ),
            "correction_gradient_occupied_virtual": (
                correction_gradient_occupied_virtual
            ),
            "correction_gradient_response": correction_gradient_response,
        }
        expected_shape = (len(atom_indices), 3)
        for name, value in partitions.items():
            _validated_float64_array(value, expected_shape, f"RKS {name}")
        return (
            partitions,
            hamiltonian_reconstruction_residual,
        )

    def _expected_objective_gradient(
        self,
        objective_ao_potential: np.ndarray,
        coefficient: np.ndarray,
        occupation: np.ndarray,
        occupied: np.ndarray,
        virtual: np.ndarray,
    ) -> np.ndarray:
        objective_mo = coefficient.T @ objective_ao_potential @ coefficient
        objective_gradient = (
            objective_mo[virtual][:, occupied]
            + objective_mo.T[virtual][:, occupied]
        ) * occupation[occupied]
        return _validated_float64_array(
            objective_gradient,
            (
                int(np.count_nonzero(virtual)),
                int(np.count_nonzero(occupied)),
            ),
            "correction occupied-virtual objective gradient",
        )

    def solve(self, objective_ao_potential: np.ndarray, atom_indices=None, compact=False):
        """Return one audited RKS Z-vector and selected nuclear contractions."""
        try:
            return self._solve(objective_ao_potential, atom_indices, compact=compact)
        except DeePHFCapabilityError:
            raise
        except RKSAdjointError:
            raise
        except (AdjointError, RKSResponseError) as error:
            raise RKSAdjointError(
                f"RKS adjoint evaluation failed: {error}"
            ) from error

    def _solve(self, objective_ao_potential: np.ndarray, atom_indices=None, compact=False):
        validate_rks_reference(self.reference)
        atom_indices = self._response_atom_indices(atom_indices)
        initial_fingerprint = rks_reference_fingerprint(self.reference)
        functional_provenance = _functional_provenance(self.reference)
        grid_provenance = _grid_provenance(self.reference)
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
        objective_orbital_gradient = self._expected_objective_gradient(
            objective_ao_potential,
            coefficient,
            occupation,
            occupied,
            virtual,
        )
        problem = _RKSLinearResponseProblem(self)
        if not isinstance(problem, ScalarAdjointProblem):
            raise RKSAdjointError(
                "the RKS adjoint operator violates the neutral scalar protocol"
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
        adjoint_ao_density = self._adjoint_density(
            zvector,
            coefficient,
            occupation,
            occupied,
            virtual,
        )
        adjoint_density_symmetry_residual = float(
            np.max(
                np.abs(adjoint_ao_density - adjoint_ao_density.T),
                initial=0.0,
            )
        )
        if adjoint_density_symmetry_residual > self.objective_symmetry_tolerance:
            raise RKSAdjointError(
                "the RKS adjoint AO density violates symmetry: "
                f"{adjoint_density_symmetry_residual:.3e} > "
                f"{self.objective_symmetry_tolerance:.3e}"
            )
        adjoint_ao_potential = _validated_float64_array(
            self._induced_potential(adjoint_ao_density),
            (self.molecule.nao, self.molecule.nao),
            "RKS adjoint AO potential",
        )
        adjoint_potential_symmetry_residual = float(
            np.max(
                np.abs(adjoint_ao_potential - adjoint_ao_potential.T),
                initial=0.0,
            )
        )
        if adjoint_potential_symmetry_residual > self.objective_symmetry_tolerance:
            raise RKSAdjointError(
                "the RKS adjoint AO potential violates symmetry: "
                f"{adjoint_potential_symmetry_residual:.3e} > "
                f"{self.objective_symmetry_tolerance:.3e}"
            )
        (
            gradient_data,
            hamiltonian_reconstruction_residual,
        ) = self._gradient_partitions(
            objective_ao_potential,
            zvector,
            adjoint_ao_potential,
            coefficient,
            energy,
            occupation,
            occupied,
            virtual,
            atom_indices,
            compact=compact,
        )
        if compact:
            correction_gradient_response = gradient_data
        else:
            partitions = gradient_data
            self._require_close(
                partitions["correction_gradient_adjoint_nuclear"],
                partitions["correction_gradient_adjoint_fixed_grid"]
                + partitions["correction_gradient_adjoint_grid_coordinate"]
                + partitions["correction_gradient_adjoint_grid_weight"],
                "nuclear gradient partition",
            )
            self._require_close(
                partitions["correction_gradient_occupied_virtual"],
                partitions["correction_gradient_adjoint_nuclear"]
                + partitions["correction_gradient_adjoint_metric"],
                "occupied-virtual gradient partition",
            )
            self._require_close(
                partitions["correction_gradient_response"],
                partitions["correction_gradient_metric"]
                + partitions["correction_gradient_occupied_virtual"],
                "response gradient partition",
            )
        if hamiltonian_reconstruction_residual > self.invariant_tolerance:
            raise RKSAdjointError(
                "the RKS adjoint Hamiltonian reconstruction exceeds tolerance"
            )
        validate_rks_reference(self.reference)
        if rks_reference_fingerprint(self.reference) != initial_fingerprint:
            raise RKSAdjointError(
                "the RKS reference changed during the scalar-adjoint evaluation"
            )
        linear_diagnostics = linear_result.diagnostics
        diagnostics = RKSAdjointDiagnostics(
            minimum_orbital_gap=minimum_gap,
            pyscf_version=pyscf.__version__,
            libxc_version=str(libxc.__version__),
            functional_components=functional_provenance.components,
            grid_point_count=grid_provenance.point_count,
            grid_coordinates_fingerprint=grid_provenance.coordinates_fingerprint,
            grid_weights_fingerprint=grid_provenance.weights_fingerprint,
            residual_tolerance=self.residual_tolerance,
            invariant_tolerance=self.invariant_tolerance,
            orbital_gap_tolerance=self.orbital_gap_tolerance,
            response_dimension=response_dimension,
            operator_is_self_adjoint=True,
            hamiltonian_reconstruction_residual=(
                hamiltonian_reconstruction_residual
            ),
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
            objective_gradient_norm=linear_diagnostics.objective_gradient_norm,
            solution_norm=linear_diagnostics.solution_norm,
            maximum_residual=linear_diagnostics.maximum_residual,
            residual_rms=linear_diagnostics.residual_rms,
            max_cycle=self.max_cycle,
            krylov_restart=self.krylov_restart,
            iteration_count=linear_diagnostics.iteration_count,
        )
        if compact:
            return diagnostics, correction_gradient_response
        adjoint = RKSAdjoint(
            reference_identity=id(self.reference),
            state_fingerprint=initial_fingerprint,
            integrity_fingerprint="",
            operator_fingerprint=linear_result.operator_fingerprint,
            atom_indices=atom_indices,
            functional_provenance=functional_provenance,
            grid_provenance=grid_provenance,
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
            **{
                name: _immutable_array(value)
                for name, value in partitions.items()
            },
            diagnostics=diagnostics,
        )
        return replace(
            adjoint,
            integrity_fingerprint=rks_adjoint_integrity_fingerprint(adjoint),
        )

    def audit_adjoint(
        self,
        adjoint: RKSAdjoint,
        expected_objective_ao_potential: np.ndarray,
    ) -> None:
        """Independently audit one consumed RKS adjoint without another solve."""
        try:
            self._audit_adjoint(adjoint, expected_objective_ao_potential)
        except DeePHFCapabilityError:
            raise
        except RKSAdjointError:
            raise
        except (AdjointError, RKSResponseError) as error:
            raise RKSAdjointError(
                f"RKS adjoint audit failed: {error}"
            ) from error

    def _audit_adjoint(
        self,
        adjoint: RKSAdjoint,
        expected_objective_ao_potential: np.ndarray,
    ) -> None:
        validate_rks_reference(self.reference)
        if type(adjoint) is not RKSAdjoint:
            raise RKSAdjointError("the supplied RKS adjoint has an invalid type")
        diagnostics = adjoint.diagnostics
        if type(diagnostics) is not RKSAdjointDiagnostics:
            raise RKSAdjointError(
                "the supplied RKS adjoint diagnostics have an invalid type"
            )
        current_fingerprint = rks_reference_fingerprint(self.reference)
        if adjoint.reference_identity != id(self.reference):
            raise RKSAdjointError(
                "the supplied RKS adjoint belongs to another reference"
            )
        if adjoint.state_fingerprint != current_fingerprint:
            raise RKSAdjointError(
                "the supplied RKS adjoint does not match the current RKS state"
            )
        if adjoint.integrity_fingerprint != rks_adjoint_integrity_fingerprint(
            adjoint
        ):
            raise RKSAdjointError(
                "the supplied RKS adjoint failed its integrity check"
            )
        provenance_values = (
            adjoint.reference_identity,
            adjoint.state_fingerprint,
            adjoint.integrity_fingerprint,
            adjoint.operator_fingerprint,
        )
        if (
            type(provenance_values[0]) is not int
            or any(type(value) is not str for value in provenance_values[1:])
        ):
            raise RKSAdjointError(
                "the supplied RKS adjoint provenance fields have invalid types"
            )
        functional_provenance = _functional_provenance(self.reference)
        grid_provenance = _grid_provenance(self.reference)
        if (
            type(adjoint.functional_provenance) is not RKSFunctionalProvenance
            or adjoint.functional_provenance != functional_provenance
        ):
            raise RKSAdjointError(
                "the supplied RKS adjoint functional provenance is invalid"
            )
        if (
            type(adjoint.grid_provenance) is not RKSGridProvenance
            or adjoint.grid_provenance != grid_provenance
        ):
            raise RKSAdjointError(
                "the supplied RKS adjoint grid provenance is invalid"
            )
        if diagnostics.solver != "scipy.sparse.linalg.gmres(A.T, b)":
            raise RKSAdjointError(
                "the supplied RKS adjoint solver convention is invalid"
            )
        if type(diagnostics.solve_count) is not int or diagnostics.solve_count != 1:
            raise RKSAdjointError(
                "the supplied RKS adjoint must contain exactly one scalar solve"
            )
        integer_diagnostics = (
            diagnostics.grid_point_count,
            diagnostics.response_dimension,
            diagnostics.max_cycle,
            diagnostics.krylov_restart,
        )
        if any(type(value) is not int or value <= 0 for value in integer_diagnostics):
            raise RKSAdjointError(
                "the supplied RKS adjoint integer diagnostics are invalid"
            )
        if (
            type(diagnostics.iteration_count) is not int
            or diagnostics.iteration_count < 0
        ):
            raise RKSAdjointError(
                "the supplied RKS adjoint iteration count is invalid"
            )
        if diagnostics.operator_is_self_adjoint is not True:
            raise RKSAdjointError("the supplied RKS adjoint operator contract is invalid")
        diagnostic_reals = (
            diagnostics.minimum_orbital_gap,
            diagnostics.residual_tolerance,
            diagnostics.invariant_tolerance,
            diagnostics.orbital_gap_tolerance,
            diagnostics.hamiltonian_reconstruction_residual,
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
            raise RKSAdjointError(
                "the supplied RKS adjoint diagnostics must be finite real scalars"
            )
        controls = {
            "residual_tolerance": self.residual_tolerance,
            "invariant_tolerance": self.invariant_tolerance,
            "orbital_gap_tolerance": self.orbital_gap_tolerance,
            "objective_symmetry_tolerance": self.objective_symmetry_tolerance,
            "max_cycle": self.max_cycle,
            "krylov_restart": self.krylov_restart,
        }
        for name, expected in controls.items():
            if getattr(diagnostics, name) != expected:
                raise RKSAdjointError(
                    f"the supplied RKS adjoint {name} control is inconsistent"
                )
        if (
            diagnostics.residual_tolerance <= 0.0
            or diagnostics.invariant_tolerance <= 0.0
            or diagnostics.orbital_gap_tolerance <= 0.0
            or diagnostics.objective_symmetry_tolerance <= 0.0
        ):
            raise RKSAdjointError(
                "the supplied RKS adjoint controls are invalid"
            )
        exact_diagnostics = {
            "pyscf_version": pyscf.__version__,
            "libxc_version": str(libxc.__version__),
            "functional_components": functional_provenance.components,
            "grid_point_count": grid_provenance.point_count,
            "grid_coordinates_fingerprint": (
                grid_provenance.coordinates_fingerprint
            ),
            "grid_weights_fingerprint": grid_provenance.weights_fingerprint,
        }
        for name, expected in exact_diagnostics.items():
            if getattr(diagnostics, name) != expected:
                raise RKSAdjointError(
                    f"the supplied RKS adjoint diagnostic {name} is invalid"
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
            raise RKSAdjointError("the supplied RKS adjoint atom selection is invalid")
        natm = len(atom_indices)
        nao = int(self.molecule.nao)
        array_shapes = {
            "objective_ao_potential": (nao, nao),
            "objective_orbital_gradient": (nvir, nocc),
            "zvector": (nvir, nocc),
            "residual": (nvir, nocc),
            "adjoint_ao_density": (nao, nao),
            "adjoint_ao_potential": (nao, nao),
            "correction_gradient_metric": (natm, 3),
            "correction_gradient_adjoint_fixed_grid": (natm, 3),
            "correction_gradient_adjoint_grid_coordinate": (natm, 3),
            "correction_gradient_adjoint_grid_weight": (natm, 3),
            "correction_gradient_adjoint_nuclear": (natm, 3),
            "correction_gradient_adjoint_metric": (natm, 3),
            "correction_gradient_occupied_virtual": (natm, 3),
            "correction_gradient_response": (natm, 3),
        }
        for name, shape in array_shapes.items():
            self._audited_array(getattr(adjoint, name), shape, name)
        expected_objective_ao_potential = self._validated_objective_potential(
            expected_objective_ao_potential
        )
        self._require_close(
            adjoint.objective_ao_potential,
            expected_objective_ao_potential,
            "objective AO potential",
        )
        expected_objective_gradient = self._expected_objective_gradient(
            expected_objective_ao_potential,
            coefficient,
            occupation,
            occupied,
            virtual,
        )
        self._require_close(
            adjoint.objective_orbital_gradient,
            expected_objective_gradient,
            "bilateral occupied-virtual objective gradient",
        )
        response_dimension = dimension
        problem = _RKSLinearResponseProblem(self)
        expected_operator_fingerprint = scalar_operator_fingerprint(
            problem,
            solver="gmres",
        )
        if adjoint.operator_fingerprint != expected_operator_fingerprint:
            raise RKSAdjointError(
                "the supplied RKS adjoint response operator is inconsistent"
            )
        objective_vector = expected_objective_gradient.reshape(dimension)
        zvector = adjoint.zvector
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
        expected_adjoint_density = self._adjoint_density(
            zvector,
            coefficient,
            occupation,
            occupied,
            virtual,
        )
        self._require_close(
            adjoint.adjoint_ao_density,
            expected_adjoint_density,
            "AO density",
        )
        expected_adjoint_potential = _validated_float64_array(
            self._induced_potential(expected_adjoint_density),
            (nao, nao),
            "independently rebuilt RKS adjoint AO potential",
        )
        self._require_close(
            adjoint.adjoint_ao_potential,
            expected_adjoint_potential,
            "AO potential",
        )
        (
            expected_partitions,
            hamiltonian_reconstruction_residual,
        ) = self._gradient_partitions(
            expected_objective_ao_potential,
            zvector,
            expected_adjoint_potential,
            coefficient,
            energy,
            occupation,
            occupied,
            virtual,
            atom_indices,
        )
        for name, expected in expected_partitions.items():
            self._require_close(getattr(adjoint, name), expected, name)
        self._require_close(
            adjoint.correction_gradient_adjoint_nuclear,
            adjoint.correction_gradient_adjoint_fixed_grid
            + adjoint.correction_gradient_adjoint_grid_coordinate
            + adjoint.correction_gradient_adjoint_grid_weight,
            "nuclear gradient partition",
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
        maximum_residual, residual_rms = self._residual_statistics(residual)
        objective_symmetry_residual = float(
            np.max(
                np.abs(
                    expected_objective_ao_potential
                    - expected_objective_ao_potential.T
                ),
                initial=0.0,
            )
        )
        density_symmetry_residual = float(
            np.max(
                np.abs(expected_adjoint_density - expected_adjoint_density.T),
                initial=0.0,
            )
        )
        potential_symmetry_residual = float(
            np.max(
                np.abs(
                    expected_adjoint_potential
                    - expected_adjoint_potential.T
                ),
                initial=0.0,
            )
        )
        measured = {
            "minimum_orbital_gap": minimum_gap,
            "response_dimension": response_dimension,
            "hamiltonian_reconstruction_residual": (
                hamiltonian_reconstruction_residual
            ),
            "objective_symmetry_residual": objective_symmetry_residual,
            "adjoint_density_symmetry_residual": density_symmetry_residual,
            "adjoint_potential_symmetry_residual": potential_symmetry_residual,
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
                raise RKSAdjointError(
                    f"the supplied RKS adjoint {name} diagnostic is inconsistent"
                )
        if (
            maximum_residual > diagnostics.residual_tolerance
            or minimum_gap <= diagnostics.orbital_gap_tolerance
            or hamiltonian_reconstruction_residual
            > diagnostics.invariant_tolerance
            or objective_symmetry_residual
            > diagnostics.objective_symmetry_tolerance
            or density_symmetry_residual
            > diagnostics.objective_symmetry_tolerance
            or potential_symmetry_residual
            > diagnostics.objective_symmetry_tolerance
        ):
            raise RKSAdjointError(
                "the supplied RKS adjoint exceeds an accepted control"
            )
        validate_rks_reference(self.reference)
        if rks_reference_fingerprint(self.reference) != current_fingerprint:
            raise RKSAdjointError(
                "the RKS reference changed during the scalar-adjoint audit"
            )


def native_rks_gradient(reference, atom_indices=None) -> np.ndarray:
    """Evaluate one selected native RKS gradient with grid response."""
    validate_rks_reference(reference)
    _validate_dft_implementations("RKS")
    from .gradient import _validate_atom_indices

    selected = _validate_atom_indices(reference.mol, atom_indices)
    atom_indices = tuple(range(reference.mol.natm)) if selected is None else selected
    initial_fingerprint = rks_reference_fingerprint(reference)
    try:
        driver = rks_grad.Gradients(reference)
        if type(driver) is not rks_grad.Gradients:
            raise RKSResponseError("the native RKS gradient driver type is invalid")
        driver.grids = reference.grids
        driver.grid_response = True
        gradient = driver.kernel(atmlst=list(atom_indices))
    except RKSResponseError:
        raise
    except Exception as error:
        raise RKSResponseError(f"PySCF native RKS gradient failed: {error}") from error
    gradient = _validated_float64_array(
        gradient,
        (len(atom_indices), 3),
        "native RKS gradient",
    )
    _validate_dft_implementations("RKS")
    validate_rks_reference(reference)
    if rks_reference_fingerprint(reference) != initial_fingerprint:
        raise RKSResponseError("the RKS reference changed during native gradient evaluation")
    return gradient
