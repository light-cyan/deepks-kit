"""PySCF projection integrals and the shared atomic descriptor interface."""

from copy import deepcopy

import numpy as np
import torch
from pyscf import gto

from deepks.gpu import DEFAULT_CUDA_DEVICE, torch_from_array
from deepks.utils import get_shell_sec, load_basis

from .core import descriptor, dq_dP, projected_density
from .derivatives import (
    dD_dR_explicit,
    dq_dR_explicit,
    dq_dR_explicit_component,
    contract_descriptor_derivatives,
)


def is_ghost_atom(mol, atom_index: int) -> bool:
    """Return whether a PySCF atom has zero nuclear charge."""
    return int(mol.atom_charge(atom_index)) == 0


def descriptor_atom_indices(mol) -> tuple[int, ...]:
    """Return raw atom indices that carry descriptor projectors."""
    return tuple(
        atom_index
        for atom_index in range(mol.natm)
        if not is_ghost_atom(mol, atom_index)
    )


def spin_summed_ao_density(ao_density):
    """Return the AO density used by the spin-summed descriptor contract."""
    if not isinstance(ao_density, torch.Tensor) and not hasattr(
        ao_density,
        "__cuda_array_interface__",
    ):
        ao_density = np.asanyarray(ao_density)
    if ao_density.ndim == 3:
        if ao_density.shape[0] != 2:
            raise ValueError("spin-resolved ao_density must have two spin channels")
        ao_density = ao_density.sum(axis=0)
    if ao_density.ndim != 2:
        raise ValueError("ao_density must be rank 2 or spin-resolved rank 3")
    return ao_density


def build_projector_molecule(mol, projector_basis):
    """Build the zero-electron molecule that carries atomic projectors."""
    atom_indices = descriptor_atom_indices(mol)
    if not atom_indices:
        raise ValueError("the descriptor requires at least one non-ghost atom")
    coordinates = mol.atom_coords(unit="Angstrom")
    projector_mol = gto.Mole()
    projector_mol.atom = [("X", coordinates[index]) for index in atom_indices]
    projector_mol.basis = projector_basis
    projector_mol.verbose = 0
    projector_mol.build(0, 0, unit="Angstrom")
    return projector_mol


def _as_ao_density_tensor(ao_density, device=DEFAULT_CUDA_DEVICE) -> torch.Tensor:
    tensor = torch_from_array(ao_density, device=device)
    if tensor.ndim != 2 or tensor.shape[-1] != tensor.shape[-2]:
        raise ValueError("ao_density must be a square rank-2 matrix")
    return tensor


class AtomicDensityDescriptor:
    """Shared projected-density descriptor bound to one PySCF molecule."""

    def __init__(self, mol, projector_basis=None, device=DEFAULT_CUDA_DEVICE):
        self.device = device
        self.projector_basis = deepcopy(load_basis(projector_basis))
        self.shell_sizes = tuple(get_shell_sec(self.projector_basis))
        self.n_features = sum(self.shell_sizes)
        self.mol = None
        self.projector_mol = None
        self.descriptor_atom_indices: tuple[int, ...] = ()
        self.overlap_shells: tuple[torch.Tensor, ...] = ()
        self.reset(mol)

    @property
    def n_descriptor_atoms(self) -> int:
        return len(self.descriptor_atom_indices)

    def reset(self, mol):
        self.mol = mol
        self.descriptor_atom_indices = descriptor_atom_indices(mol)
        self.projector_mol = build_projector_molecule(mol, self.projector_basis)
        overlap = torch_from_array(self.projection_overlap(), device=self.device)
        self.overlap_shells = tuple(torch.split(overlap, self.shell_sizes, -1))
        self._derivative_overlap_shells = None
        return self

    def as_ao_density_tensor(self, ao_density) -> torch.Tensor:
        return _as_ao_density_tensor(ao_density, self.device)

    def cross_integral(self, integral: str) -> np.ndarray:
        return gto.intor_cross(integral, self.mol, self.projector_mol)

    def projection_overlap(self) -> np.ndarray:
        overlap = self.cross_integral("int1e_ovlp")
        return overlap.reshape(
            self.mol.nao,
            self.projector_mol.natm,
            self.projector_mol.nao // self.projector_mol.natm,
        )

    def derivative_overlap_shells(self) -> tuple[torch.Tensor, ...]:
        if self._derivative_overlap_shells is not None:
            return self._derivative_overlap_shells
        derivative_overlap = torch_from_array(
            self.cross_integral("int1e_ipovlp"),
            device=self.device,
        )
        derivative_overlap = derivative_overlap.reshape(
            3,
            self.mol.nao,
            self.projector_mol.natm,
            -1,
        )
        self._derivative_overlap_shells = tuple(
            torch.split(derivative_overlap, self.shell_sizes, -1)
        )
        return self._derivative_overlap_shells

    def derivative_workspace(self, ao_density, *, operation_hook=None):
        """Build one calculation-scoped owner of reusable derivative primitives."""
        from .workspace import DescriptorDerivativeWorkspace

        return DescriptorDerivativeWorkspace(
            self,
            ao_density,
            operation_hook=operation_hook,
        )

    def torch_projected_density(self, ao_density) -> list[torch.Tensor]:
        return projected_density(
            self.as_ao_density_tensor(ao_density),
            self.overlap_shells,
        )

    def projected_density(self, ao_density, flatten: bool = False):
        blocks = self.torch_projected_density(ao_density)
        if flatten:
            return torch.cat(
                [block.flatten(-2) for block in blocks],
                dim=-1,
            ).detach().cpu().numpy()
        return [block.detach().cpu().numpy() for block in blocks]

    def torch_descriptor(self, ao_density) -> torch.Tensor:
        return descriptor(
            self.as_ao_density_tensor(ao_density),
            self.overlap_shells,
        )

    def descriptor(self, ao_density) -> np.ndarray:
        return self.torch_descriptor(ao_density).detach().cpu().numpy()

    def dq_dP(self, ao_density) -> np.ndarray:
        result = dq_dP(
            self.as_ao_density_tensor(ao_density),
            self.overlap_shells,
        )
        return result.detach().cpu().numpy()

    def dD_dR_explicit(self, ao_density, flatten: bool = False):
        blocks = dD_dR_explicit(
            self.mol,
            self.as_ao_density_tensor(ao_density),
            self.overlap_shells,
            self.derivative_overlap_shells(),
            self.descriptor_atom_indices,
        )
        if flatten:
            return torch.cat(
                [block.flatten(-2) for block in blocks],
                dim=-1,
            ).detach().cpu().numpy()
        return [block.detach().cpu().numpy() for block in blocks]

    def dq_dR_explicit(self, ao_density, raw_atom_indices=None) -> np.ndarray:
        result = dq_dR_explicit(
            self.mol,
            self.as_ao_density_tensor(ao_density),
            self.overlap_shells,
            self.derivative_overlap_shells(),
            self.descriptor_atom_indices,
            raw_atom_indices,
        )
        return result.detach().cpu().numpy()

    def dq_dR_explicit_component(
        self,
        ao_density,
        component_density,
        raw_atom_indices=None,
    ) -> np.ndarray:
        """Return one additive component of fixed-density descriptor motion."""
        result = dq_dR_explicit_component(
            self.mol,
            self.as_ao_density_tensor(ao_density),
            self.as_ao_density_tensor(component_density),
            self.overlap_shells,
            self.derivative_overlap_shells(),
            self.descriptor_atom_indices,
            raw_atom_indices,
        )
        return result.detach().cpu().numpy()

    def correction_derivatives(
        self,
        ao_density,
        motion_density,
        sensitivity,
        raw_atom_indices=None,
    ) -> np.ndarray:
        """Return contracted explicit motion and the AO correction potential."""
        density = self.as_ao_density_tensor(ao_density)
        gradient, potential = contract_descriptor_derivatives(
            self.mol,
            density,
            self.as_ao_density_tensor(motion_density),
            torch.tensor(sensitivity, dtype=density.dtype, device=density.device),
            self.overlap_shells,
            self.derivative_overlap_shells(),
            self.descriptor_atom_indices,
            raw_atom_indices,
        )
        return (
            gradient.detach().cpu().numpy(),
            potential.detach().cpu().numpy(),
        )
