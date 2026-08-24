"""Isolated PySCF 2.14 adapter for molecular pure-LDA RKS response."""

from dataclasses import dataclass
import ctypes
import inspect
from numbers import Real
import operator
import weakref

import numpy as np
import pyscf
from pyscf import dft
from pyscf.dft import gen_grid, libxc, numint, radi
from pyscf.gto import mole as gto_mole
from pyscf.grad import rhf as rhf_grad, rks as rks_grad


from .adjoint import AdjointError
from .capabilities import DeePHFCapabilityError, transaction_reference_fingerprint
from .pyscf_rhf_reference import RHFResponse
from functools import partial

from .contracts import (
    array_fingerprint as _array_fingerprint,
    integer_control,
    real_control,
    validated_float64_array,
    version_series as _canonical_version_series,
)


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
    return _canonical_version_series(version, DeePHFCapabilityError)


def validate_pyscf_version() -> None:
    """Require the PySCF series characterized by this adapter."""
    if _version_series(pyscf.__version__) != SUPPORTED_PYSCF_SERIES:
        raise DeePHFCapabilityError(
            "the RKS response adapter supports PySCF 2.14; "
            f"found {pyscf.__version__}"
        )


def _qualified_name(value) -> str:
    return f"{value.__module__}.{value.__qualname__}"


_validated_float64_array = partial(
    validated_float64_array,
    error_type=RKSResponseError,
)
_cycle_limit = integer_control
_response_real_control = real_control


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
