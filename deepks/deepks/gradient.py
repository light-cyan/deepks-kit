"""Variational analytic nuclear gradients for self-consistent DeePKS."""

import abc
import time

import numpy as np
import torch
from gpu4pyscf.grad import rks as rks_grad
from gpu4pyscf.grad import uhf as uhf_grad
from gpu4pyscf.grad import uks as uks_grad
from gpu4pyscf.lib import logger
from gpu4pyscf.scf import uhf as gpu_uhf
from pyscf import gto

from deepks.descriptor import dD_dR_explicit, dq_dR_explicit
from deepks.descriptor import spin_summed_ao_density
from deepks.gpu import torch_from_array
from deepks.model.evaluate import correction_projected_density_gradients


def correction_explicit_gradient(
    mol,
    model,
    ao_density: torch.Tensor,
    overlap_shells,
    derivative_overlap_shells,
    descriptor_atom_indices,
    atom_indices=None,
) -> torch.Tensor:
    """Return the explicit dE_corr/dR contribution at fixed AO density."""
    if atom_indices is None:
        atom_indices = tuple(range(mol.natm))
    projected_density_gradients = correction_projected_density_gradients(
        model,
        ao_density,
        overlap_shells,
    )
    coordinate_jacobians = dD_dR_explicit(
        mol,
        ao_density,
        overlap_shells,
        derivative_overlap_shells,
        descriptor_atom_indices,
    )
    full_gradient = sum(
        torch.einsum("bxapq,apq->bx", coordinate, energy_gradient)
        for coordinate, energy_gradient in zip(
            coordinate_jacobians,
            projected_density_gradients,
        )
    )
    return full_gradient[list(atom_indices)]


class CorrectionGradientMixin(abc.ABC):
    """Add the explicit correction derivative to a native PySCF gradient."""

    def __init__(self):
        self.explicit_correction_gradient = None

    def grad_elec(self, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
        gradient = self.reference_electronic_gradient(
            mo_energy,
            mo_coeff,
            mo_occ,
            atmlst,
        )
        timer = (time.process_time(), time.perf_counter())
        correction_gradient = self.correction_gradient(
            self.base.make_rdm1(mo_coeff, mo_occ),
            atmlst,
        )
        logger.timer(self, "explicit correction gradient", *timer)
        self.explicit_correction_gradient = (
            self.symmetrize(correction_gradient, atmlst)
            if self.mol.symmetry
            else correction_gradient
        )
        return gradient + correction_gradient

    def reference_electronic_gradient(
        self,
        mo_energy=None,
        mo_coeff=None,
        mo_occ=None,
        atmlst=None,
    ):
        """Return the native electronic gradient at the converged GPU state."""
        return super().grad_elec(mo_energy, mo_coeff, mo_occ, atmlst)

    def reference_gradient(self):
        """Return the native-reference part at the converged DeePKS density."""
        if self.de is None or self.explicit_correction_gradient is None:
            raise RuntimeError("the gradient kernel must run before reference_gradient")
        return self.de - self.explicit_correction_gradient

    @abc.abstractmethod
    def correction_gradient(self, ao_density=None, atom_indices=None):
        if atom_indices is None:
            atom_indices = range(self.mol.natm)
        return np.zeros((len(tuple(atom_indices)), 3))


class ModelGradientMixin(CorrectionGradientMixin):
    """Use the shared descriptor derivative in a DeePKS gradient."""

    def __init__(self):
        CorrectionGradientMixin.__init__(self)
        self._descriptor = self.base._descriptor
        self.prepare_descriptor_derivatives()

    def prepare_descriptor_derivatives(self):
        self._derivative_overlap_shells = (
            self._descriptor.derivative_overlap_shells()
        )

    def correction_gradient(self, ao_density=None, atom_indices=None):
        if atom_indices is None:
            atom_indices = tuple(range(self.mol.natm))
        else:
            atom_indices = tuple(atom_indices)
        if self.base.model is None:
            return np.zeros((len(atom_indices), 3))
        if ao_density is None:
            ao_density = self.base.make_rdm1()
        tensor_density = torch_from_array(
            spin_summed_ao_density(ao_density),
            device=self.base.model_device,
        )
        result = correction_explicit_gradient(
            self.mol,
            self.base.model,
            tensor_density,
            self._descriptor.overlap_shells,
            self._derivative_overlap_shells,
            self._descriptor.descriptor_atom_indices,
            atom_indices,
        )
        return result.detach().cpu().numpy()

    def dD_dR_explicit(self, ao_density=None, flatten=False):
        if ao_density is None:
            ao_density = self.base.make_rdm1()
        tensor_density = torch_from_array(
            spin_summed_ao_density(ao_density),
            device=self.base.model_device,
        )
        blocks = dD_dR_explicit(
            self.mol,
            tensor_density,
            self._descriptor.overlap_shells,
            self._derivative_overlap_shells,
            self._descriptor.descriptor_atom_indices,
        )
        if flatten:
            return torch.cat(
                [block.flatten(-2) for block in blocks],
                dim=-1,
            ).detach().cpu().numpy()
        return [block.detach().cpu().numpy() for block in blocks]

    def dq_dR_explicit(self, ao_density=None):
        if ao_density is None:
            ao_density = self.base.make_rdm1()
        tensor_density = torch_from_array(
            spin_summed_ao_density(ao_density),
            device=self.base.model_device,
        )
        result = dq_dR_explicit(
            self.mol,
            tensor_density,
            self._descriptor.overlap_shells,
            self._derivative_overlap_shells,
            self._descriptor.descriptor_atom_indices,
        )
        return result.detach().cpu().numpy()

    def optimize_descriptor_potential(self, target_density, **kwargs):
        from .addons import optimize_descriptor_potential_from_gradient

        return optimize_descriptor_potential_from_gradient(
            self,
            target_density,
            **kwargs,
        )

    def as_scanner(self):
        scanner = super().as_scanner()

        class DeePKSGradientScanner(type(scanner)):
            def __call__(self, mol_or_geometry, **kwargs):
                if isinstance(mol_or_geometry, gto.Mole):
                    mol = mol_or_geometry
                else:
                    mol = self.mol.set_geom_(mol_or_geometry, inplace=False)
                energy = self.base(mol)
                self.mol = mol
                if getattr(self, "grids", None):
                    self.grids.reset(mol)
                self._descriptor = self.base._descriptor
                self.prepare_descriptor_derivatives()
                gradient = self.kernel(**kwargs)
                return energy, gradient

        scanner.__class__ = DeePKSGradientScanner
        return scanner


def build_gradient(mean_field):
    if isinstance(mean_field, gpu_uhf.UHF):
        if str(mean_field.xc).strip().upper() == "HF":
            return UHFDeePKSGradients(mean_field)
        return UDeePKSGradients(mean_field)
    return RDeePKSGradients(mean_field)


class RDeePKSGradients(ModelGradientMixin, rks_grad.Gradients):
    """Restricted DeePKS analytic nuclear gradient."""

    def __init__(self, mean_field):
        rks_grad.Gradients.__init__(self, mean_field)
        ModelGradientMixin.__init__(self)
        self._keys.update(self.__dict__.keys())


class UDeePKSGradients(ModelGradientMixin, uks_grad.Gradients):
    """Unrestricted DeePKS analytic nuclear gradient."""

    def __init__(self, mean_field):
        uks_grad.Gradients.__init__(self, mean_field)
        ModelGradientMixin.__init__(self)
        self._keys.update(self.__dict__.keys())


class UHFDeePKSGradients(ModelGradientMixin, uhf_grad.Gradients):
    """Unrestricted Hartree-Fock DeePKS analytic nuclear gradient."""

    def __init__(self, mean_field):
        uhf_grad.Gradients.__init__(self, mean_field)
        ModelGradientMixin.__init__(self)
        self._keys.update(self.__dict__.keys())
