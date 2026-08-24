"""Isolated PySCF 2.14 adapter for strict finite-grid molecular UKS response."""

from dataclasses import dataclass
from numbers import Real
import weakref

import numpy as np
from pyscf import dft
from pyscf.dft import libxc, numint
from pyscf.grad import uks as uks_grad


from .capabilities import (
    DeePHFCapabilityError,
    reference_is_transaction_validated,
    transaction_reference_fingerprint,
)
from .contracts import array_fingerprint as _array_fingerprint, dataclass_fingerprint
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
from .pyscf_uhf_reference import (
    UHFAdjoint,
    UHFAdjointDiagnostics,
    UHFAdjointError,
    UHFResponse,
    UHFResponseDiagnostics,
    UHFResponseError,
    _native_unrestricted_gradient,
)


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
    from .audits.uks_reference import _audit_uks_reference as audit
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


def uks_reference_fingerprint(reference) -> str:
    """Fingerprint one accepted UKS molecular, orbital, functional, and grid state."""
    trusted = transaction_reference_fingerprint(reference)
    if trusted is not None:
        return trusted
    return _dft_reference_validation_fingerprint(reference)
