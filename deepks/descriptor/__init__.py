"""Shared atomic projected-density descriptors."""

from .core import descriptor, dq_dP, projected_density, shell_eigenvalues
from .derivatives import dD_dR_explicit, dq_dR_explicit
from .orbitals import occupied_virtual_gradient
from .projection import (
    AtomicDensityDescriptor,
    build_projector_molecule,
    descriptor_atom_indices,
    is_ghost_atom,
    spin_summed_ao_density,
)
from .validation import (
    DescriptorDiagnostics,
    DescriptorDifferentiabilityError,
    validate_differentiability,
)
from .workspace import DescriptorDerivativeWorkspace

__all__ = [
    "AtomicDensityDescriptor",
    "DescriptorDiagnostics",
    "DescriptorDifferentiabilityError",
    "DescriptorDerivativeWorkspace",
    "build_projector_molecule",
    "dD_dR_explicit",
    "descriptor",
    "descriptor_atom_indices",
    "dq_dP",
    "dq_dR_explicit",
    "is_ghost_atom",
    "occupied_virtual_gradient",
    "projected_density",
    "shell_eigenvalues",
    "spin_summed_ao_density",
    "validate_differentiability",
]
