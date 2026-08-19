"""Strict capability checks for perturbative DeePHF references."""

import numpy as np
from pyscf import scf

from deepks.descriptor import is_ghost_atom


class DeePHFCapabilityError(ValueError):
    """Raised when a reference is outside the declared DeePHF domain."""


def validate_reference(reference):
    """Validate the molecular real-orbital integer-occupation RHF contract."""
    if type(reference) is not scf.hf.RHF:
        raise DeePHFCapabilityError(
            "DeePHF requires an undecorated native pyscf.scf.hf.RHF reference"
        )
    if not reference.converged:
        raise DeePHFCapabilityError("the RHF reference must be converged")
    mol = reference.mol
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
    if getattr(mol, "_ecp", None):
        raise DeePHFCapabilityError(
            "the initial DeePHF contract requires an all-electron reference"
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
    mo_coeff = np.asarray(reference.mo_coeff)
    if np.iscomplexobj(mo_coeff):
        raise DeePHFCapabilityError("the RHF orbitals must be real")
    occupations = np.asarray(reference.mo_occ)
    if not np.all(np.isin(occupations, (0.0, 2.0))):
        raise DeePHFCapabilityError(
            "the RHF occupations must be integer closed-shell occupations"
        )
    if not np.isfinite(reference.e_tot):
        raise DeePHFCapabilityError("the RHF reference energy must be finite")
    return reference
