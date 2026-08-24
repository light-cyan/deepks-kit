"""Strict reference audits separated from reference state ownership."""

from __future__ import annotations

from ..capabilities import DeePHFCapabilityError
from pyscf.gto import mole as gto_mole
from deepks.descriptor import is_ghost_atom
import numpy as np
from ..capabilities import reference_is_transaction_validated
from pyscf.scf import hf as scf_hf


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


__all__ = ['validate_reference']
