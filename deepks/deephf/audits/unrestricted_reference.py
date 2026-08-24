"""Strict reference audits separated from reference state ownership."""

from __future__ import annotations

from numbers import Real

import numpy as np
from pyscf import dft
from pyscf.gto import mole as gto_mole
from pyscf.scf import hf as scf_hf
from pyscf.scf import uhf as scf_uhf

from deepks.descriptor import is_ghost_atom

from ..capabilities import DeePHFCapabilityError, reference_is_transaction_validated
from ..pyscf_dft_provenance import _GRID_PROVENANCE_CACHE, _build_grid_provenance
from ..unrestricted_reference import (
    UHFResponseError,
    _dense_uks_quadrature,
    _direct_effective_potential,
    _uks_functional_provenance,
    validate_pyscf_version,
)


def _validate_reference_contract(reference):
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
    return molecule


def _validated_orbital_state(reference, molecule):
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
    return coefficient, energy, expected_electrons


def _validate_ao_state(reference, molecule, state):
    coefficient, energy, expected_electrons = state
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


def validate_uhf_reference(reference):
    """Validate the native real-orbital integer-occupation UHF contract."""
    if type(reference) is not scf_uhf.UHF:
        raise DeePHFCapabilityError(
            "UHF DeePHF requires an undecorated native pyscf.scf.uhf.UHF reference"
        )
    if reference_is_transaction_validated(reference):
        return reference
    molecule = _validate_reference_contract(reference)
    state = _validated_orbital_state(reference, molecule)
    _validate_ao_state(reference, molecule, state)
    return reference


def _validated_uks_state(reference):
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
    return molecule, coefficient, energy, expected_electrons


def _validate_uks_ao_state(reference, state):
    molecule, coefficient, energy, expected_electrons = state
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


def _audit_uks_reference(reference):
    """Audit the exact native finite-grid pure-LDA UKS tier."""
    state = _validated_uks_state(reference)
    _validate_uks_ao_state(reference, state)
    return reference


__all__ = ["_audit_uks_reference", "validate_uhf_reference"]
