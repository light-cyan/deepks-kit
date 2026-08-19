"""Strict capability checks for perturbative DeePHF references and models."""

import numpy as np
import torch
from pyscf import gto, scf

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
    if type(mol) is not gto.mole.Mole:
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
    if not all(
        np.isfinite(value).all()
        for value in (mo_coeff, mo_energy, occupations)
    ):
        raise DeePHFCapabilityError("the RHF orbital state must be finite")
    if any(
        value.dtype != np.dtype(np.float64)
        for value in (mo_coeff, mo_energy, occupations)
    ):
        raise DeePHFCapabilityError(
            "the RHF orbital state must use numpy.float64"
        )
    if mo_coeff.shape != (mol.nao, mol.nao):
        raise DeePHFCapabilityError(
            "the RHF response requires a complete square MO coefficient matrix"
        )
    if mo_energy.shape != (mo_coeff.shape[1],):
        raise DeePHFCapabilityError("the RHF orbital energy shape is invalid")
    if occupations.shape != mo_energy.shape:
        raise DeePHFCapabilityError("the RHF occupation shape is invalid")
    occupations = np.asarray(reference.mo_occ)
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
    except Exception as error:
        raise DeePHFCapabilityError(
            f"the RHF reference matrices could not be evaluated: {error}"
        ) from error
    if any(
        np.iscomplexobj(value)
        for value in (overlap, hcore, density, effective_potential)
    ):
        raise DeePHFCapabilityError("the RHF AO matrices must be real")
    if not all(
        np.isfinite(value).all()
        for value in (overlap, hcore, density, effective_potential)
    ):
        raise DeePHFCapabilityError("the RHF AO matrices must be finite")
    if any(
        value.dtype != np.dtype(np.float64)
        for value in (overlap, hcore, density, effective_potential)
    ):
        raise DeePHFCapabilityError("the RHF AO matrices must use numpy.float64")
    expected_ao_shape = (mol.nao, mol.nao)
    if any(
        value.shape != expected_ao_shape
        for value in (overlap, hcore, density, effective_potential)
    ):
        raise DeePHFCapabilityError("the RHF AO matrix shape is invalid")
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
    fock = hcore + effective_potential
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


def _metadata_signature(value):
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return tuple(_metadata_signature(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def validate_model(model, projector_basis, descriptor_features: int):
    """Validate the strict double-precision scalar correction-model contract."""
    if model is None:
        return None
    if not isinstance(model, torch.nn.Module):
        raise DeePHFCapabilityError(
            "the DeePHF correction model must be a torch.nn.Module or None"
        )
    tensors = list(model.parameters()) + list(model.buffers())
    for tensor in tensors:
        if tensor.is_complex():
            raise DeePHFCapabilityError("the correction model must be real")
        if tensor.is_floating_point() and tensor.dtype != torch.float64:
            raise DeePHFCapabilityError(
                "the correction model must use torch.float64"
            )
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise DeePHFCapabilityError(
                "the correction model parameters and buffers must be finite"
            )
    input_dimension = getattr(model, "input_dim", descriptor_features)
    if input_dimension != descriptor_features:
        raise DeePHFCapabilityError(
            "the correction model input dimension does not match the descriptor: "
            f"{input_dimension} != {descriptor_features}"
        )
    model_basis = getattr(model, "_pbas", None)
    if (
        model_basis is not None
        and _metadata_signature(model_basis) != _metadata_signature(projector_basis)
    ):
        raise DeePHFCapabilityError(
            "the correction model projector metadata does not match projector_basis"
        )
    return model


def validate_model_output(model, descriptor_values: torch.Tensor) -> torch.Tensor:
    """Evaluate and validate one real finite scalar correction energy."""
    if model is None:
        return torch.zeros((), dtype=torch.float64)
    try:
        reference_tensor = next(model.parameters())
    except StopIteration:
        try:
            reference_tensor = next(model.buffers())
        except StopIteration:
            reference_tensor = torch.empty((), dtype=torch.float64)
    try:
        output = model(descriptor_values.to(reference_tensor))
    except Exception as error:
        raise DeePHFCapabilityError(
            f"the correction model evaluation failed: {error}"
        ) from error
    if not isinstance(output, torch.Tensor):
        raise DeePHFCapabilityError("the correction model output must be a tensor")
    if output.is_complex():
        raise DeePHFCapabilityError("the correction model output must be real")
    if output.dtype != torch.float64:
        raise DeePHFCapabilityError(
            "the correction model output must use torch.float64"
        )
    if output.numel() != 1:
        raise DeePHFCapabilityError(
            "the correction model must produce exactly one scalar energy; "
            f"received shape {tuple(output.shape)}"
        )
    if not torch.isfinite(output).all():
        raise DeePHFCapabilityError("the correction model output must be finite")
    return output
