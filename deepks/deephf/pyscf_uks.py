"""Isolated PySCF 2.14 adapter for strict finite-grid molecular UKS response."""

from dataclasses import dataclass, fields, replace
import hashlib
from numbers import Real
from typing import Any
import weakref

import numpy as np
import pyscf
from pyscf import dft
from pyscf.dft import libxc, numint
from pyscf.gto import mole as gto_mole
from pyscf.grad import uks as uks_grad
from pyscf.hessian import uks as uks_hessian
from pyscf.scf import hf as scf_hf

from deepks.descriptor import is_ghost_atom

from .capabilities import (
    DeePHFCapabilityError,
    reference_is_transaction_validated,
    transaction_reference_fingerprint,
)
from .pyscf_rks import (
    RKSFunctionalProvenance,
    RKSGridProvenance,
    SUPPORTED_LIBXC_COMPONENTS,
    SUPPORTED_LIBXC_VERSION,
    SUPPORTED_NUMINT_CUTOFF,
    _array_fingerprint,
    _build_grid_provenance,
    _dft_reference_validation_fingerprint,
    _GRID_PROVENANCE_CACHE,
    _grid_provenance,
    _normalized_atom_grid,
    _normalized_functional_components,
    _validate_dft_implementations,
    _validated_grid_response_blocks,
)
from .pyscf_uhf import (
    UHFAdjoint,
    UHFAdjointAdapter,
    UHFAdjointDiagnostics,
    UHFAdjointError,
    UHFResponse,
    UHFResponseAdapter,
    UHFResponseDiagnostics,
    UHFResponseError,
    _immutable_array,
    _native_unrestricted_gradient,
    _response_real_control,
    _validated_float64_array,
    uhf_adjoint_integrity_fingerprint,
    uhf_molecule_science_fingerprint,
    uhf_response_integrity_fingerprint,
    validate_pyscf_version,
)


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
    nuclear_partition_residual: float

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


def _update_fingerprint_value(digest, value: Any) -> None:
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(repr(array.shape).encode("ascii"))
        digest.update(array.tobytes())
        return
    if hasattr(value, "__dataclass_fields__"):
        for field in fields(value):
            digest.update(field.name.encode("utf-8"))
            _update_fingerprint_value(digest, getattr(value, field.name))
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            _update_fingerprint_value(digest, item)
        return
    if isinstance(value, dict):
        for key in sorted(value, key=repr):
            _update_fingerprint_value(digest, key)
            _update_fingerprint_value(digest, value[key])
        return
    if isinstance(value, np.generic):
        value = value.item()
    digest.update(type(value).__name__.encode("utf-8"))
    digest.update(repr(value).encode("utf-8"))


def uks_response_integrity_fingerprint(response: UKSResponse) -> str:
    """Return a digest covering every public UKS response field."""
    digest = hashlib.sha256()
    for field in fields(response):
        if field.name == "integrity_fingerprint":
            continue
        digest.update(field.name.encode("utf-8"))
        _update_fingerprint_value(digest, getattr(response, field.name))
    return digest.hexdigest()


def uks_adjoint_integrity_fingerprint(adjoint: UKSAdjoint) -> str:
    """Return a digest covering every public UKS adjoint field."""
    digest = hashlib.sha256()
    for field in fields(adjoint):
        if field.name == "integrity_fingerprint":
            continue
        digest.update(field.name.encode("utf-8"))
        _update_fingerprint_value(digest, getattr(adjoint, field.name))
    return digest.hexdigest()


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
    """Audit the exact native finite-grid pure-LDA UKS tier."""
    validate_pyscf_version()
    if type(reference) is not dft.uks.UKS:
        raise DeePHFCapabilityError(
            "UKS DeePHF requires an undecorated native pyscf.dft.uks.UKS reference"
        )
    if not reference.converged:
        raise DeePHFCapabilityError("the UKS reference must be converged")
    molecule = reference.mol
    if type(molecule) is not gto_mole.Mole:
        raise DeePHFCapabilityError("the UKS reference must use a native molecular Mole")
    if molecule.symmetry is not False or molecule.cart:
        raise DeePHFCapabilityError(
            "the strict UKS tier requires symmetry-disabled spherical molecular AOs"
        )
    if getattr(molecule, "_pseudo", None):
        raise DeePHFCapabilityError("the strict UKS tier does not support pseudopotentials")
    if getattr(molecule, "_ecp", None) or molecule.has_ecp():
        raise DeePHFCapabilityError("the strict UKS tier requires an all-electron reference")
    if float(getattr(molecule, "omega", 0.0)) != 0.0:
        raise DeePHFCapabilityError("the strict UKS tier requires full Coulomb interaction")
    if getattr(molecule, "nucmod", None):
        raise DeePHFCapabilityError("the strict UKS tier requires point nuclei")
    ghosts = [index for index in range(molecule.natm) if is_ghost_atom(molecule, index)]
    if ghosts:
        raise DeePHFCapabilityError(
            f"the strict UKS tier requires real atoms; ghost indices: {ghosts}"
        )
    decorated = {
        "density fitting": "with_df",
        "solvent": "with_solvent",
        "X2C": "with_x2c",
        "QM/MM": "mm_mol",
        "dispersion": "disp",
        "penalty": "penalties",
    }
    active = [name for name, attribute in decorated.items() if getattr(reference, attribute, None)]
    if active:
        raise DeePHFCapabilityError(
            "the UKS reference has unsupported decorations: " + ", ".join(active)
        )
    if reference.nlc not in ("", None, 0, False):
        raise DeePHFCapabilityError("the strict UKS tier does not support NLC")
    if (
        isinstance(reference.small_rho_cutoff, (bool, np.bool_))
        or not isinstance(reference.small_rho_cutoff, Real)
        or not np.isfinite(float(reference.small_rho_cutoff))
        or float(reference.small_rho_cutoff) != 0.0
    ):
        raise DeePHFCapabilityError(
            "the strict UKS grid requires small_rho_cutoff=0"
        )
    reference_hooks = sorted(
        name
        for name, value in reference.__dict__.items()
        if name not in {"mol", "grids", "nlcgrids", "_numint"} and callable(value)
    )
    molecule_hooks = sorted(
        name for name, value in molecule.__dict__.items() if callable(value)
    )
    if reference_hooks or molecule_hooks:
        raise DeePHFCapabilityError("the strict UKS reference contains instance hooks")
    functional = _uks_functional_provenance(reference)
    grid = _build_grid_provenance(reference)
    _GRID_PROVENANCE_CACHE[reference] = (None, grid)
    if reference.mo_coeff is None or reference.mo_energy is None or reference.mo_occ is None:
        raise DeePHFCapabilityError("the UKS orbital state is incomplete")
    coefficient = np.asarray(reference.mo_coeff)
    energy = np.asarray(reference.mo_energy)
    occupation = np.asarray(reference.mo_occ)
    orbital_values = (coefficient, energy, occupation)
    if any(np.iscomplexobj(value) for value in orbital_values):
        raise DeePHFCapabilityError("the UKS orbitals must be real")
    if any(value.dtype != np.dtype(np.float64) for value in orbital_values):
        raise DeePHFCapabilityError("the UKS orbital state must use numpy.float64")
    if not all(np.isfinite(value).all() for value in orbital_values):
        raise DeePHFCapabilityError("the UKS orbital state must be finite")
    if coefficient.shape != (2, molecule.nao, molecule.nao):
        raise DeePHFCapabilityError("UKS requires two complete square MO matrices")
    if energy.shape != (2, molecule.nao) or occupation.shape != (2, molecule.nao):
        raise DeePHFCapabilityError("the UKS orbital energy or occupation shape is invalid")
    if not np.all(np.isin(occupation, (0.0, 1.0))):
        raise DeePHFCapabilityError("the UKS occupations must be zero or one")
    expected_electrons = tuple(int(value) for value in molecule.nelec)
    actual_electrons = tuple(int(value) for value in occupation.sum(axis=1))
    if actual_electrons != expected_electrons:
        raise DeePHFCapabilityError("the UKS occupations do not match mol.nelec")
    for spin_index, spin_name in enumerate(("alpha", "beta")):
        occupied_count = expected_electrons[spin_index]
        expected_occupation = np.zeros_like(occupation[spin_index])
        expected_occupation[:occupied_count] = 1.0
        if not np.array_equal(occupation[spin_index], expected_occupation):
            raise DeePHFCapabilityError(
                f"the strict UKS tier requires the Aufbau {spin_name} root"
            )
        if occupied_count == 0 or occupied_count == molecule.nao:
            raise DeePHFCapabilityError(
                "UKS response requires occupied and virtual orbitals in each spin"
            )
        if np.any(np.diff(energy[spin_index]) < -1.0e-10):
            raise DeePHFCapabilityError(f"the UKS {spin_name} energies are not ordered")
        if energy[spin_index, occupied_count] <= energy[spin_index, occupied_count - 1]:
            raise DeePHFCapabilityError(f"the UKS {spin_name} root spaces overlap")
    if not np.isfinite(reference.e_tot):
        raise DeePHFCapabilityError("the UKS reference energy must be finite")
    try:
        overlap = np.asarray(reference.get_ovlp())
        hcore = np.asarray(reference.get_hcore())
        density = np.asarray(reference.make_rdm1())
        effective = np.asarray(reference.get_veff(molecule, density))
        coulomb, _exchange = scf_hf.get_jk(molecule, density, hermi=1)
        total_coulomb = np.asarray(coulomb[0] + coulomb[1])
        electron_counts, xc_energy, xc_potential = _dense_uks_quadrature(
            reference,
            density,
        )
        direct_effective = total_coulomb[None] + xc_potential
    except Exception as error:
        if isinstance(error, DeePHFCapabilityError):
            raise
        raise DeePHFCapabilityError(
            f"the UKS reference matrices could not be evaluated: {error}"
        ) from error
    ao_values = (overlap, hcore, density, effective, direct_effective)
    if any(np.iscomplexobj(value) for value in ao_values):
        raise DeePHFCapabilityError("the UKS AO matrices must be real")
    if any(value.dtype != np.dtype(np.float64) for value in ao_values):
        raise DeePHFCapabilityError("the UKS AO matrices must use numpy.float64")
    if not all(np.isfinite(value).all() for value in ao_values):
        raise DeePHFCapabilityError("the UKS AO matrices must be finite")
    if overlap.shape != (molecule.nao, molecule.nao) or hcore.shape != overlap.shape:
        raise DeePHFCapabilityError("the UKS spin-independent AO shapes are invalid")
    if any(value.shape != (2, molecule.nao, molecule.nao) for value in ao_values[2:]):
        raise DeePHFCapabilityError("the UKS spin-resolved AO shapes are invalid")
    interaction_residual = float(
        np.max(np.abs(effective - direct_effective), initial=0.0)
    )
    if interaction_residual > 1.0e-9:
        raise DeePHFCapabilityError(
            "the UKS finite-grid effective potential is inconsistent: "
            f"residual {interaction_residual:.3e}"
        )
    if np.linalg.eigvalsh(overlap)[0] <= 1.0e-10:
        raise DeePHFCapabilityError("the UKS AO overlap is singular")
    if not np.isfinite(electron_counts).all():
        raise DeePHFCapabilityError("the UKS grid electron counts are nonfinite")
    fock = hcore[None] + direct_effective
    for spin_index, spin_name in enumerate(("alpha", "beta")):
        spin_coefficient = coefficient[spin_index]
        spin_density = density[spin_index]
        if np.max(np.abs(spin_coefficient.T @ overlap @ spin_coefficient - np.eye(molecule.nao))) > 1.0e-8:
            raise DeePHFCapabilityError(f"the UKS {spin_name} orbitals are not orthonormal")
        if np.max(np.abs(spin_density - spin_density.T)) > 1.0e-10:
            raise DeePHFCapabilityError(f"the UKS {spin_name} density is not symmetric")
        if np.max(np.abs(spin_density @ overlap @ spin_density - spin_density)) > 1.0e-8:
            raise DeePHFCapabilityError(f"the UKS {spin_name} density is not idempotent")
        count = float(np.einsum("ij,ji->", spin_density, overlap))
        if not np.isclose(count, expected_electrons[spin_index], rtol=0.0, atol=1.0e-8):
            raise DeePHFCapabilityError(f"the UKS {spin_name} electron count is inconsistent")
        residual = fock[spin_index] @ spin_coefficient - overlap @ (
            spin_coefficient * energy[spin_index]
        )
        if np.max(np.abs(residual)) > 1.0e-7:
            raise DeePHFCapabilityError(
                f"the UKS {spin_name} canonical residual is excessive"
            )
    total_density = density.sum(axis=0)
    recomputed_energy = (
        np.einsum("ij,ji->", total_density, hcore)
        + 0.5 * np.einsum("ij,ji->", total_density, total_coulomb)
        + xc_energy
        + molecule.energy_nuc()
    )
    if not np.isclose(recomputed_energy, reference.e_tot, rtol=0.0, atol=1.0e-8):
        raise DeePHFCapabilityError(
            "the stored UKS total energy is inconsistent with its finite-grid AO state"
        )
    coordinates = np.asarray(gto_mole.Mole.atom_coords(molecule, unit="Bohr"))
    if coordinates.dtype != np.dtype(np.float64) or not np.isfinite(coordinates).all():
        raise DeePHFCapabilityError("the UKS molecular geometry must be finite float64")
    return reference


def validate_uks_reference(reference):
    """Validate a UKS reference once per unchanged scientific state."""
    if reference_is_transaction_validated(reference):
        return reference
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


class _UKSLinearResponseMixin:
    """Replace the UHF J/K response by strict finite-grid UKS J plus LDA f_xc."""

    @staticmethod
    def _validate_reference(reference):
        return validate_uks_reference(reference)

    @staticmethod
    def _reference_fingerprint(reference) -> str:
        return uks_reference_fingerprint(reference)

    def _xc_nuclear_derivative_components(self, density: np.ndarray, atom_indices=None):
        molecule = self.molecule
        atom_indices = self._response_atom_indices(atom_indices)
        result_positions = {
            atom_index: result_index
            for result_index, atom_index in enumerate(atom_indices)
        }
        integration = self.reference._numint
        shape = (2, len(atom_indices), 3, molecule.nao, molecule.nao)
        grid_coordinate = np.zeros(shape)
        grid_weight = np.zeros(shape)
        blocks = _validated_grid_response_blocks(
            self.reference,
            _normalized_atom_grid(molecule, self.reference.grids.atom_grid),
            audit_weight_derivative=False,
        )
        for host_atom, (coordinates, weights, weight_derivative) in enumerate(blocks):
            try:
                ao = integration.eval_ao(molecule, coordinates, deriv=1)
                values = ao[0]
                gradients = ao[1:4]
                rho = np.stack(
                    [
                        np.einsum("gp,pq,gq->g", values, spin_density, values, optimize=True)
                        for spin_density in density
                    ]
                )
                xc_values = integration.eval_xc_eff(
                    self.reference.xc,
                    rho,
                    deriv=2,
                    xctype="LDA",
                    spin=1,
                )
                potential = np.asarray(xc_values[1])[:, 0]
                kernel = np.asarray(xc_values[2])[:, 0, :, 0]
            except Exception as error:
                raise UKSResponseError(
                    f"UKS LDA nuclear quadrature failed: {error}"
                ) from error
            values_to_check = (
                np.asarray(coordinates),
                np.asarray(weights),
                np.asarray(weight_derivative),
                values,
                gradients,
                rho,
                potential,
                kernel,
            )
            if not all(np.isfinite(value).all() for value in values_to_check):
                raise UKSResponseError("the UKS LDA nuclear quadrature is nonfinite")
            grid_weight += np.einsum(
                "axg,sg,gp,gq->saxpq",
                weight_derivative[list(atom_indices)],
                potential,
                values,
                values,
                optimize=True,
            )

            def accumulate(target, atom_index, axis, derivative_values):
                density_derivative = np.stack(
                    [
                        np.einsum(
                            "gp,pq,gq->g",
                            derivative_values,
                            spin_density,
                            values,
                            optimize=True,
                        )
                        + np.einsum(
                            "gp,pq,gq->g",
                            values,
                            spin_density,
                            derivative_values,
                            optimize=True,
                        )
                        for spin_density in density
                    ]
                )
                potential_derivative = np.einsum(
                    "tsg,tg->sg",
                    kernel,
                    density_derivative,
                    optimize=True,
                )
                target[:, atom_index, axis] += np.einsum(
                    "g,sg,gp,gq->spq",
                    weights,
                    potential_derivative,
                    values,
                    values,
                    optimize=True,
                )
                target[:, atom_index, axis] += np.einsum(
                    "g,sg,gp,gq->spq",
                    weights,
                    potential,
                    derivative_values,
                    values,
                    optimize=True,
                )
                target[:, atom_index, axis] += np.einsum(
                    "g,sg,gp,gq->spq",
                    weights,
                    potential,
                    values,
                    derivative_values,
                    optimize=True,
                )

            if host_atom in result_positions:
                result_index = result_positions[host_atom]
                for axis in range(3):
                    accumulate(grid_coordinate, result_index, axis, gradients[axis])
        return grid_coordinate, grid_weight

    def _hamiltonian_derivative(self, coefficient, occupation, atom_indices=None):
        atom_indices = self._response_atom_indices(atom_indices)
        density = np.asarray(self.reference.make_rdm1(coefficient, occupation))
        expected = (2, len(atom_indices), 3, self.molecule.nao, self.molecule.nao)
        try:
            hessian = uks_hessian.Hessian(self.reference)
            fixed_grid = hessian.make_h1(
                coefficient,
                occupation,
                atmlst=atom_indices,
            )
            fixed_grid = np.stack(
                [
                    [fixed_grid[spin][atom_index] for atom_index in atom_indices]
                    for spin in range(2)
                ]
            )
        except Exception as error:
            raise UKSResponseError(
                f"PySCF UKS Hamiltonian derivative construction failed: {error}"
            ) from error
        fixed_grid = _validated_float64_array(
            fixed_grid,
            expected,
            "fixed-grid UKS Hamiltonian derivative",
        )
        grid_coordinate, grid_weight = self._xc_nuclear_derivative_components(
            density,
            atom_indices,
        )
        full = fixed_grid + grid_coordinate + grid_weight
        if not all(
            np.isfinite(value).all()
            for value in (full, fixed_grid, grid_coordinate, grid_weight)
        ):
            raise UKSResponseError("the complete UKS Hamiltonian derivative is nonfinite")
        self._last_hamiltonian_components = (
            full,
            fixed_grid,
            grid_coordinate,
            grid_weight,
        )
        return full[0], full[1]

    def _induced_potential(self, alpha_density, beta_density):
        perturbation_shape = alpha_density.shape[:-2]
        flat_alpha = np.asarray(alpha_density).reshape(-1, self.molecule.nao, self.molecule.nao)
        flat_beta = np.asarray(beta_density).reshape(flat_alpha.shape)
        if not np.isfinite(flat_alpha).all() or not np.isfinite(flat_beta).all():
            raise UKSResponseError("the UKS trial density response is nonfinite")
        if max(
            float(np.max(np.abs(flat_alpha - flat_alpha.swapaxes(-1, -2)), initial=0.0)),
            float(np.max(np.abs(flat_beta - flat_beta.swapaxes(-1, -2)), initial=0.0)),
        ) > 1.0e-10:
            raise UKSResponseError("the UKS trial density response is not symmetric")
        try:
            coulomb, _exchange = scf_hf.get_jk(
                self.molecule,
                np.stack((flat_alpha, flat_beta)),
                hermi=1,
            )
            total_coulomb = np.asarray(coulomb[0] + coulomb[1])
            coordinates = np.asarray(self.reference.grids.coords)
            weights = np.asarray(self.reference.grids.weights)
            integration = self.reference._numint
            ao = integration.eval_ao(self.molecule, coordinates, deriv=0)
            ground_density = np.asarray(self.reference.make_rdm1())
            rho = np.stack(
                [
                    np.einsum("gp,pq,gq->g", ao, spin_density, ao, optimize=True)
                    for spin_density in ground_density
                ]
            )
            kernel = np.asarray(
                integration.eval_xc_eff(
                    self.reference.xc,
                    rho,
                    deriv=2,
                    xctype="LDA",
                    spin=1,
                )[2]
            )[:, 0, :, 0]
            density_response = np.stack(
                [
                    np.einsum("gp,xpq,gq->xg", ao, spin_density, ao, optimize=True)
                    for spin_density in (flat_alpha, flat_beta)
                ]
            )
            potential_response = np.einsum(
                "tsg,txg->sxg",
                kernel,
                density_response,
                optimize=True,
            )
            xc_response = np.einsum(
                "g,sxg,gp,gq->sxpq",
                weights,
                potential_response,
                ao,
                ao,
                optimize=True,
            )
        except Exception as error:
            raise UKSResponseError(
                f"the independent UKS J plus LDA f_xc action failed: {error}"
            ) from error
        expected = (*perturbation_shape, self.molecule.nao, self.molecule.nao)
        alpha = total_coulomb + xc_response[0]
        beta = total_coulomb + xc_response[1]
        return alpha.reshape(expected), beta.reshape(expected)


class _UKSInternalResponseAdapter(_UKSLinearResponseMixin, UHFResponseAdapter):
    pass


class _UKSInternalAdjointAdapter(_UKSLinearResponseMixin, UHFAdjointAdapter):
    pass


def _require_wrapper_close(actual, expected, name: str, error_type) -> None:
    if not np.allclose(actual, expected, rtol=1.0e-11, atol=1.0e-12):
        residual = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
        raise error_type(f"the UKS {name} is inconsistent: residual {residual:.3e}")


class UKSResponseAdapter:
    """Solve and independently audit complete finite-grid UKS response."""

    def __init__(self, reference, **controls):
        try:
            self._core = _UKSInternalResponseAdapter(reference, **controls)
        except (DeePHFCapabilityError, UKSResponseError):
            raise
        except UHFResponseError as error:
            raise UKSResponseError(f"UKS response setup failed: {error}") from error
        self.reference = self._core.reference
        for name in (
            "cphf_tolerance",
            "residual_tolerance",
            "invariant_tolerance",
            "orbital_gap_tolerance",
            "max_cycle",
            "max_refinement_cycles",
            "level_shift",
            "operator_stability_tolerance",
            "operator_condition_tolerance",
            "operator_symmetry_tolerance",
            "operator_dimension_limit",
        ):
            setattr(self, name, getattr(self._core, name))

    @staticmethod
    def _components(core) -> tuple[np.ndarray, ...]:
        components = getattr(core, "_last_hamiltonian_components", None)
        if type(components) is not tuple or len(components) != 4:
            raise UKSResponseError("the UKS Hamiltonian derivative partitions are unavailable")
        return tuple(np.stack(value) if isinstance(value, tuple) else value for value in components)

    def solve(self, atom_indices=None) -> UKSResponse:
        """Return one immutable UKS response for selected atoms."""
        return self._solve(atom_indices, "response")

    def _solve_with_density_partitions(self, atom_indices=None):
        """Return a response and its transient spin-density work arrays."""
        return self._solve(atom_indices, "partitions")

    def _solve_for_gradient(self, atom_indices=None):
        """Return compact diagnostics and transient spin-density work arrays."""
        return self._solve(atom_indices, "gradient")

    def _solve(self, atom_indices, result_mode):
        try:
            if result_mode == "gradient":
                core_diagnostics, density_partitions = self._core._solve_for_gradient(
                    atom_indices=atom_indices
                )
                core_response = None
            elif result_mode == "partitions":
                core_response, density_partitions = (
                    self._core._solve_with_density_partitions(
                        atom_indices=atom_indices
                    )
                )
                core_diagnostics = core_response.diagnostics
            else:
                core_response = self._core.solve(atom_indices=atom_indices)
                core_diagnostics = core_response.diagnostics
            if result_mode == "gradient":
                components = self._core._last_hamiltonian_components
                reconstruction = max(
                    float(np.max(np.abs(a - b - c - d), initial=0.0))
                    for a, b, c, d in zip(*components, strict=True)
                )
            else:
                full, fixed, coordinate, weight = self._components(self._core)
                _require_wrapper_close(full, fixed + coordinate + weight, "Hamiltonian derivative partition", UKSResponseError)
                reconstruction = float(
                    np.max(np.abs(full - fixed - coordinate - weight), initial=0.0)
                )
            functional = _uks_functional_provenance(self.reference)
            grid = _grid_provenance(self.reference)
            diagnostics = UKSResponseDiagnostics(
                core=core_diagnostics,
                functional=functional,
                grid=grid,
                hamiltonian_reconstruction_residual=reconstruction,
            )
            if result_mode == "gradient":
                return diagnostics, density_partitions
            response = UKSResponse(
                core=core_response,
                functional=functional,
                grid=grid,
                hamiltonian_derivative_fixed_grid_spin=_immutable_array(fixed),
                xc_hamiltonian_derivative_grid_coordinate_spin=_immutable_array(coordinate),
                xc_hamiltonian_derivative_grid_weight_spin=_immutable_array(weight),
                diagnostics=diagnostics,
                integrity_fingerprint="",
            )
            response = replace(
                response,
                integrity_fingerprint=uks_response_integrity_fingerprint(response),
            )
            return (
                (response, density_partitions)
                if result_mode == "partitions"
                else response
            )
        except DeePHFCapabilityError:
            raise
        except UKSResponseError:
            raise
        except UHFResponseError as error:
            raise UKSResponseError(f"UKS response evaluation failed: {error}") from error

    def validate_response_operator_exact(self):
        """Run the bounded explicit debug audit of the internal UKS operator."""
        return self._core.validate_response_operator_exact()

    def audit_response_equations(self, response: UKSResponse) -> None:
        """Rebuild one supplied UKS response without another CPHF solve."""
        validate_uks_reference(self.reference)
        if type(response) is not UKSResponse or type(response.diagnostics) is not UKSResponseDiagnostics:
            raise UKSResponseError("the supplied UKS response has an invalid type")
        if response.integrity_fingerprint != uks_response_integrity_fingerprint(response):
            raise UKSResponseError("the supplied UKS response failed its integrity check")
        if response.functional != _uks_functional_provenance(self.reference) or response.grid != _grid_provenance(self.reference):
            raise UKSResponseError("the supplied UKS response provenance is inconsistent")
        if response.diagnostics.functional != response.functional or response.diagnostics.grid != response.grid:
            raise UKSResponseError("the supplied UKS response diagnostics provenance is inconsistent")
        try:
            self._core.audit_response_equations(response.core)
        except UHFResponseError as error:
            raise UKSResponseError(f"UKS response audit failed: {error}") from error
        full, fixed, coordinate, weight = self._components(self._core)
        expected_shape = (2, len(response.core.atom_indices), 3, self.reference.mol.nao, self.reference.mol.nao)
        arrays = {
            "fixed-grid Hamiltonian derivative": (response.hamiltonian_derivative_fixed_grid_spin, fixed),
            "grid-coordinate XC derivative": (response.xc_hamiltonian_derivative_grid_coordinate_spin, coordinate),
            "grid-weight XC derivative": (response.xc_hamiltonian_derivative_grid_weight_spin, weight),
        }
        for name, (actual, expected) in arrays.items():
            if type(actual) is not np.ndarray or actual.shape != expected_shape or actual.dtype != np.dtype(np.float64) or actual.flags.writeable or not np.isfinite(actual).all():
                raise UKSResponseError(f"the supplied UKS {name} is invalid")
            _require_wrapper_close(actual, expected, name, UKSResponseError)
        reconstruction = float(np.max(np.abs(full - fixed - coordinate - weight), initial=0.0))
        measured = {
            "hamiltonian_reconstruction_residual": reconstruction,
        }
        for name, expected in measured.items():
            stored = getattr(response.diagnostics, name)
            if isinstance(stored, (bool, np.bool_)) or not isinstance(stored, Real) or not np.isfinite(stored) or not np.isclose(stored, expected, rtol=1.0e-10, atol=1.0e-12):
                raise UKSResponseError(f"the supplied UKS {name} diagnostic is inconsistent")
        if max(measured.values()) > response.diagnostics.invariant_tolerance:
            raise UKSResponseError("the supplied UKS response invariant exceeds tolerance")


class UKSAdjointAdapter:
    """Solve one finite-grid UKS correction-specific coupled scalar adjoint."""

    def __init__(self, reference, **controls):
        try:
            self._core = _UKSInternalAdjointAdapter(reference, **controls)
        except (DeePHFCapabilityError, UKSAdjointError):
            raise
        except UHFAdjointError as error:
            raise UKSAdjointError(f"UKS adjoint setup failed: {error}") from error
        self.reference = self._core.reference
        for name in (
            "residual_tolerance",
            "invariant_tolerance",
            "orbital_gap_tolerance",
            "operator_stability_tolerance",
            "operator_condition_tolerance",
            "operator_symmetry_tolerance",
            "operator_dimension_limit",
            "objective_symmetry_tolerance",
            "max_cycle",
            "krylov_restart",
        ):
            setattr(self, name, getattr(self._core, name))

    def _nuclear_partitions(self, core_adjoint: UHFAdjoint):
        atom_indices = core_adjoint.atom_indices
        coefficient, energy, occupation, occupied, virtual, _ = self._core._state()
        overlap = self._core._overlap_derivative(atom_indices)
        full, fixed, coordinate, weight = UKSResponseAdapter._components(self._core)
        zvector = (core_adjoint.alpha_zvector, core_adjoint.beta_zvector)
        fixed_spin = []
        coordinate_spin = []
        weight_spin = []
        for spin in range(2):
            occupied_coefficients = coefficient[spin][:, occupied[spin]]
            overlap_mo = np.einsum("mp,...mn,ni->...pi", coefficient[spin], overlap, occupied_coefficients)
            fixed_mo = np.einsum("mp,...mn,ni->...pi", coefficient[spin], fixed[spin], occupied_coefficients)
            coordinate_mo = np.einsum("mp,...mn,ni->...pi", coefficient[spin], coordinate[spin], occupied_coefficients)
            weight_mo = np.einsum("mp,...mn,ni->...pi", coefficient[spin], weight[spin], occupied_coefficients)
            fixed_rhs = fixed_mo[..., virtual[spin], :] - overlap_mo[..., virtual[spin], :] * energy[spin, occupied[spin]]
            fixed_spin.append(-np.einsum("ai,...ai->...", zvector[spin], fixed_rhs))
            coordinate_spin.append(-np.einsum("ai,...ai->...", zvector[spin], coordinate_mo[..., virtual[spin], :]))
            weight_spin.append(-np.einsum("ai,...ai->...", zvector[spin], weight_mo[..., virtual[spin], :]))
        fixed_spin = np.stack(fixed_spin)
        coordinate_spin = np.stack(coordinate_spin)
        weight_spin = np.stack(weight_spin)
        residual = float(np.max(np.abs(core_adjoint.correction_gradient_adjoint_nuclear_spin - fixed_spin - coordinate_spin - weight_spin), initial=0.0))
        return fixed_spin, coordinate_spin, weight_spin, residual

    def validate_response_operator_exact(self):
        """Run the bounded explicit debug audit of the internal UKS operator."""
        return self._core.validate_response_operator_exact()

    def solve(self, objective_ao_potential: np.ndarray, atom_indices=None) -> UKSAdjoint:
        """Return one immutable UKS adjoint from exactly one transpose solve."""
        try:
            core_adjoint = self._core.solve(
                objective_ao_potential,
                atom_indices=atom_indices,
            )
            fixed_spin, coordinate_spin, weight_spin, partition_residual = self._nuclear_partitions(core_adjoint)
            if partition_residual > self.invariant_tolerance:
                raise UKSAdjointError("the UKS adjoint nuclear partitions are inconsistent")
            functional = _uks_functional_provenance(self.reference)
            grid = _grid_provenance(self.reference)
            diagnostics = UKSAdjointDiagnostics(
                core=core_adjoint.diagnostics,
                functional=functional,
                grid=grid,
                nuclear_partition_residual=partition_residual,
            )
            adjoint = UKSAdjoint(
                core=core_adjoint,
                functional=functional,
                grid=grid,
                correction_gradient_adjoint_fixed_grid_spin=_immutable_array(fixed_spin),
                correction_gradient_adjoint_grid_coordinate_spin=_immutable_array(coordinate_spin),
                correction_gradient_adjoint_grid_weight_spin=_immutable_array(weight_spin),
                correction_gradient_adjoint_fixed_grid=_immutable_array(fixed_spin.sum(axis=0)),
                correction_gradient_adjoint_grid_coordinate=_immutable_array(coordinate_spin.sum(axis=0)),
                correction_gradient_adjoint_grid_weight=_immutable_array(weight_spin.sum(axis=0)),
                diagnostics=diagnostics,
                integrity_fingerprint="",
            )
            return replace(adjoint, integrity_fingerprint=uks_adjoint_integrity_fingerprint(adjoint))
        except DeePHFCapabilityError:
            raise
        except UKSAdjointError:
            raise
        except UHFAdjointError as error:
            raise UKSAdjointError(f"UKS adjoint evaluation failed: {error}") from error

    def audit_adjoint(self, adjoint: UKSAdjoint, expected_objective_ao_potential: np.ndarray) -> None:
        """Rebuild one UKS adjoint without another transpose solve."""
        validate_uks_reference(self.reference)
        if type(adjoint) is not UKSAdjoint or type(adjoint.diagnostics) is not UKSAdjointDiagnostics:
            raise UKSAdjointError("the supplied UKS adjoint has an invalid type")
        if adjoint.integrity_fingerprint != uks_adjoint_integrity_fingerprint(adjoint):
            raise UKSAdjointError("the supplied UKS adjoint failed its integrity check")
        if adjoint.functional != _uks_functional_provenance(self.reference) or adjoint.grid != _grid_provenance(self.reference):
            raise UKSAdjointError("the supplied UKS adjoint provenance is inconsistent")
        if adjoint.diagnostics.functional != adjoint.functional or adjoint.diagnostics.grid != adjoint.grid:
            raise UKSAdjointError("the supplied UKS adjoint diagnostics provenance is inconsistent")
        try:
            self._core.audit_adjoint(adjoint.core, expected_objective_ao_potential)
        except UHFAdjointError as error:
            raise UKSAdjointError(f"UKS adjoint audit failed: {error}") from error
        fixed_spin, coordinate_spin, weight_spin, partition_residual = self._nuclear_partitions(adjoint.core)
        expected_shape = (2, len(adjoint.core.atom_indices), 3)
        spin_arrays = {
            "fixed-grid adjoint gradient": (adjoint.correction_gradient_adjoint_fixed_grid_spin, fixed_spin),
            "grid-coordinate adjoint gradient": (adjoint.correction_gradient_adjoint_grid_coordinate_spin, coordinate_spin),
            "grid-weight adjoint gradient": (adjoint.correction_gradient_adjoint_grid_weight_spin, weight_spin),
        }
        for name, (actual, expected) in spin_arrays.items():
            if type(actual) is not np.ndarray or actual.shape != expected_shape or actual.dtype != np.dtype(np.float64) or actual.flags.writeable or not np.isfinite(actual).all():
                raise UKSAdjointError(f"the supplied UKS {name} is invalid")
            _require_wrapper_close(actual, expected, name, UKSAdjointError)
        for name, spin_name in (
            ("correction_gradient_adjoint_fixed_grid", "correction_gradient_adjoint_fixed_grid_spin"),
            ("correction_gradient_adjoint_grid_coordinate", "correction_gradient_adjoint_grid_coordinate_spin"),
            ("correction_gradient_adjoint_grid_weight", "correction_gradient_adjoint_grid_weight_spin"),
        ):
            _require_wrapper_close(getattr(adjoint, name), getattr(adjoint, spin_name).sum(axis=0), name, UKSAdjointError)
        measured = {
            "nuclear_partition_residual": partition_residual,
        }
        for name, expected in measured.items():
            stored = getattr(adjoint.diagnostics, name)
            if isinstance(stored, (bool, np.bool_)) or not isinstance(stored, Real) or not np.isfinite(stored) or not np.isclose(stored, expected, rtol=1.0e-10, atol=1.0e-12):
                raise UKSAdjointError(f"the supplied UKS {name} diagnostic is inconsistent")
        if max(measured.values()) > adjoint.diagnostics.invariant_tolerance:
            raise UKSAdjointError("the supplied UKS adjoint invariant exceeds tolerance")


def native_uks_gradient(reference, atom_indices=None) -> np.ndarray:
    """Evaluate one selected native UKS gradient with grid response."""
    validate_uks_reference(reference)
    atom_indices = (
        tuple(range(reference.mol.natm))
        if atom_indices is None
        else tuple(atom_indices)
    )
    initial_fingerprint = uks_reference_fingerprint(reference)
    try:
        driver = uks_grad.Gradients(reference)
        if type(driver) is not uks_grad.Gradients:
            raise UKSResponseError("the native UKS gradient driver type is invalid")
        driver.grids = reference.grids
        driver.grid_response = True
        gradient = _native_unrestricted_gradient(reference, driver, atom_indices)
    except UKSResponseError:
        raise
    except Exception as error:
        raise UKSResponseError(f"PySCF native UKS gradient failed: {error}") from error
    gradient = _validated_float64_array(
        gradient,
        (len(atom_indices), 3),
        "native UKS gradient",
    )
    validate_uks_reference(reference)
    if uks_reference_fingerprint(reference) != initial_fingerprint:
        raise UKSResponseError("the UKS reference changed during native gradient evaluation")
    return gradient


__all__ = [
    "UKSAdjoint",
    "UKSAdjointAdapter",
    "UKSAdjointDiagnostics",
    "UKSAdjointError",
    "UKSResponse",
    "UKSResponseAdapter",
    "UKSResponseDiagnostics",
    "UKSResponseError",
    "native_uks_gradient",
    "uks_adjoint_integrity_fingerprint",
    "uks_reference_fingerprint",
    "uks_response_integrity_fingerprint",
    "validate_uks_reference",
]
