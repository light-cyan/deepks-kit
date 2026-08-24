"""Strict finite-grid RKS reference state and native gradient support."""

import hashlib
import numpy as np
import pyscf
from pyscf import dft
from pyscf.dft import libxc, numint
from pyscf.gto import mole as gto_mole
from pyscf.grad import rks as rks_grad
from .capabilities import (
    DeePHFCapabilityError,
    reference_is_transaction_validated,
    transaction_reference_fingerprint,
)
from .pyscf_dft_provenance import (
    RKSAdjoint,
    RKSResponse,
    RKSResponseError,
    _GRID_PROVENANCE_CACHE,
    _NATIVE_RKS_GRADIENT_METHODS,
    _SUPPORTED_BECKE_SCHEME,
    _SUPPORTED_GRIDS_RESPONSE,
    _SUPPORTED_LIBXC_IMPLEMENTATIONS,
    _SUPPORTED_NUMINT_IMPLEMENTATIONS,
    _SUPPORTED_RADII_ADJUST,
    _SUPPORTED_RADI_METHOD,
    _VALIDATED_RKS_REFERENCES,
    _grid_arrays,
    _qualified_name,
    _static_callable_definitions,
    _validated_float64_array,
    _validate_dft_implementations,
)
from .contracts import dataclass_fingerprint, update_digest as _update_fingerprint_value

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
    from .audits.restricted_reference import _audit_rks_reference as audit
    return audit(reference)


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


def rks_reference_fingerprint(reference, *, use_transaction=True) -> str:
    """Return a scratch-independent fingerprint of the scientific RKS state."""
    trusted = (
        transaction_reference_fingerprint(reference)
        if use_transaction
        else None
    )
    if trusted is not None:
        return trusted
    return _dft_reference_validation_fingerprint(reference)


def rks_response_integrity_fingerprint(response: RKSResponse) -> str:
    """Return a digest covering every RKS response field except itself."""
    return dataclass_fingerprint(
        response,
        excluded=frozenset({"integrity_fingerprint"}),
    )


def rks_adjoint_integrity_fingerprint(adjoint: RKSAdjoint) -> str:
    """Return a digest covering every RKS adjoint field except itself."""
    return dataclass_fingerprint(
        adjoint,
        excluded=frozenset({"integrity_fingerprint"}),
    )

def native_rks_gradient(reference, atom_indices=None) -> np.ndarray:
    """Evaluate one selected native RKS gradient with grid response."""
    validate_rks_reference(reference)
    _validate_dft_implementations("RKS")
    from .driver import validate_atom_indices

    selected = validate_atom_indices(reference.mol, atom_indices)
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
