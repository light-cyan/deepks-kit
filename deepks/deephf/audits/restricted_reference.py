"""Strict reference audits separated from reference state ownership."""

from __future__ import annotations

from numbers import Real

import numpy as np
from pyscf import dft
from pyscf.gto import mole as gto_mole
from pyscf.scf import hf as scf_hf

from deepks.descriptor import is_ghost_atom

from ..capabilities import DeePHFCapabilityError, reference_is_transaction_validated
from ..pyscf_dft_provenance import (
    SUPPORTED_LIBXC_COMPONENTS,
    _GRID_PROVENANCE_CACHE,
    _build_grid_provenance,
    _functional_provenance,
    validate_pyscf_version,
)
from ..pyscf_rks_reference import _dense_ground_state_lda_quadrature


def _validate_molecule(mol):
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


def _validate_runtime_extensions(reference, mol):
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


def _validated_orbital_state(reference, mol):
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
    return mo_coeff, mo_energy


def _evaluated_ao_state(reference, mol):
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
    return overlap, hcore, density, effective_potential, direct_effective_potential


def _validate_ao_state(reference, mol, mo_coeff, mo_energy, ao_state):
    overlap, hcore, density, effective_potential, direct_effective_potential = ao_state
    if any(np.iscomplexobj(value) for value in ao_state):
        raise DeePHFCapabilityError("the RHF AO matrices must be real")
    if any(value.dtype != np.dtype(np.float64) for value in ao_state):
        raise DeePHFCapabilityError("the RHF AO matrices must use numpy.float64")
    if not all(np.isfinite(value).all() for value in ao_state):
        raise DeePHFCapabilityError("the RHF AO matrices must be finite")
    expected_ao_shape = (mol.nao, mol.nao)
    if any(value.shape != expected_ao_shape for value in ao_state):
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
    _validate_molecule(mol)
    _validate_runtime_extensions(reference, mol)
    mo_coeff, mo_energy = _validated_orbital_state(reference, mol)
    ao_state = _evaluated_ao_state(reference, mol)
    _validate_ao_state(reference, mol, mo_coeff, mo_energy, ao_state)
    return reference


def _validate_reference_contract(reference):
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
    return molecule


def _validated_rks_orbital_state(reference, molecule):
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
    return functional_provenance, grid_provenance, coefficient, energy


def _validate_rks_ao_state(reference, molecule, state):
    functional_provenance, grid_provenance, coefficient, energy = state
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


def _audit_rks_reference(reference):
    """Audit the exact native, converged, finite-grid pure-LDA RKS tier."""
    molecule = _validate_reference_contract(reference)
    state = _validated_rks_orbital_state(reference, molecule)
    _validate_rks_ao_state(reference, molecule, state)
    return reference


__all__ = ["_audit_rks_reference", "validate_reference"]
