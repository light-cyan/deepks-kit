"""Self-consistent DeePKS mean-field methods."""

import abc
import time

import numpy as np
import torch
from gpu4pyscf.dft import rks as gpu_rks
from gpu4pyscf.dft import uks as gpu_uks
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import tag_array

from deepks.descriptor import AtomicDensityDescriptor, spin_summed_ao_density
from deepks.gpu import (
    DEFAULT_CUDA_DEVICE,
    GPU_DIRECT_SCF_TOL,
    cupy_from_torch,
    require_cuda_device,
    torch_from_array,
)
from deepks.model.evaluate import correction as evaluate_correction
from deepks.model.model import CorrNet

from .penalty import PenaltyMixin


DEFAULT_DEVICE = DEFAULT_CUDA_DEVICE


class CorrectionMixin(abc.ABC):
    """Add a variational correction to a PySCF mean-field method."""

    def reference_effective_potential(self, *args, **kwargs):
        return super().get_veff(*args, **kwargs)

    def reference_orbital_gradient(self, mo_coeff=None, mo_occ=None):
        if mo_occ is None:
            mo_occ = self.mo_occ
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        return super().get_grad(
            mo_coeff,
            mo_occ,
            fock=self.get_fock(vhf=self.reference_effective_potential()),
        )

    def reference_electronic_energy(self, dm=None, h1e=None, vhf=None):
        if vhf is None:
            vhf = self.reference_effective_potential(dm=dm)
        return super().energy_elec(dm, h1e, vhf)

    def reference_energy(self, dm=None, h1e=None, vhf=None):
        if (
            dm is None
            and h1e is None
            and vhf is None
            and getattr(self, "e_base", None) is not None
        ):
            return self.e_base
        return self.reference_electronic_energy(dm, h1e, vhf)[0] + self.energy_nuc()

    def reference_nuclear_gradient_method(self):
        return super().nuc_grad_method()

    def get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        """Return the reference and learned correction potentials."""
        if mol is None:
            mol = self.mol
        if dm is None:
            dm = self.make_rdm1()
        timer = (time.process_time(), time.perf_counter())
        reference_last = getattr(vhf_last, "reference", 0)
        reference = self.reference_effective_potential(
            mol,
            dm,
            dm_last,
            reference_last,
            hermi,
        )
        timer = logger.timer(self, "reference potential", *timer)
        correction_energy, correction_potential = self.correction(dm)
        logger.timer(self, "correction potential", *timer)
        total = reference + correction_potential
        return tag_array(
            total,
            correction_energy=correction_energy,
            reference=reference,
        )

    def energy_elec(self, dm=None, h1e=None, vhf=None):
        """Return the self-consistent corrected electronic energy."""
        if dm is None:
            dm = self.make_rdm1()
        if h1e is None:
            h1e = self.get_hcore()
        if vhf is None or getattr(vhf, "correction_energy", None) is None:
            vhf = self.get_veff(dm=dm)
        total, two_electron = self.reference_electronic_energy(
            dm,
            h1e,
            vhf.reference,
        )
        correction_energy = vhf.correction_energy
        self.e_base = total.real + self.energy_nuc()
        logger.debug(self, "E_corr = %s", correction_energy)
        return (
            (total + correction_energy).real,
            two_electron + correction_energy,
        )

    @abc.abstractmethod
    def correction(self, dm=None):
        """Return the correction energy and AO potential."""
        if dm is None:
            dm = self.make_rdm1()
        return 0.0, np.zeros_like(dm)

    @abc.abstractmethod
    def nuc_grad_method(self):
        return self.reference_nuclear_gradient_method()


class ModelCorrectionMixin(CorrectionMixin):
    """Bind a neural correction and shared descriptor to a mean-field method."""

    def __init__(self, model, projector_basis=None, device=DEFAULT_DEVICE):
        self.model_device = require_cuda_device(device)
        self.e_base = None
        if isinstance(model, str):
            model = CorrNet.load(model).double()
        if isinstance(model, torch.nn.Module):
            model = model.to(self.model_device).eval()
        self.model = model
        if projector_basis is None:
            projector_basis = getattr(model, "_pbas", None)
        self._descriptor = AtomicDensityDescriptor(
            self.mol,
            projector_basis,
            device=self.model_device,
        )

    @property
    def projector_basis(self):
        return self._descriptor.projector_basis

    @property
    def descriptor_shell_sizes(self):
        return self._descriptor.shell_sizes

    @property
    def n_descriptor_features(self):
        return self._descriptor.n_features

    @property
    def n_descriptor_atoms(self):
        return self._descriptor.n_descriptor_atoms

    def correction(self, dm=None):
        """Return the model correction energy and AO potential."""
        if dm is None:
            dm = self.make_rdm1()
        ao_density = spin_summed_ao_density(dm)
        if self.model is None:
            return 0.0, ao_density * 0
        tensor_density = torch_from_array(
            ao_density,
            device=self.model_device,
        )
        tensor_energy, tensor_potential = evaluate_correction(
            self.model,
            tensor_density,
            self._descriptor.overlap_shells,
            with_potential=True,
        )
        energy = (
            tensor_energy.item()
            if tensor_energy.numel() == 1
            else tensor_energy.detach().cpu().numpy()
        )
        potential = cupy_from_torch(tensor_potential)
        real_charges = [
            int(charge) for charge in self.mol.atom_charges() if charge > 0
        ]
        energy += self.model.get_elem_const(real_charges)
        return energy, potential

    def nuc_grad_method(self):
        if self.penalties:
            raise RuntimeError(
                "DeePKS analytic gradients require a penalty-free method"
            )
        from .gradient import build_gradient

        return build_gradient(self)

    def reset(self, mol=None):
        super().reset(mol)
        self.e_base = None
        self._descriptor.reset(self.mol)
        return self

    def projected_density(self, ao_density=None, flatten=False):
        if ao_density is None:
            ao_density = self.make_rdm1()
        return self._descriptor.projected_density(
            spin_summed_ao_density(ao_density),
            flatten=flatten,
        )

    def descriptor(self, ao_density=None):
        if ao_density is None:
            ao_density = self.make_rdm1()
        return self._descriptor.descriptor(spin_summed_ao_density(ao_density))

    def dq_dP(self, ao_density=None):
        if ao_density is None:
            ao_density = self.make_rdm1()
        return self._descriptor.dq_dP(spin_summed_ao_density(ao_density))

    def descriptor_orbital_gradient_jacobian(
        self,
        mo_coeff=None,
        mo_occ=None,
        ao_jacobian=None,
    ):
        from .addons import descriptor_orbital_gradient_jacobian

        return descriptor_orbital_gradient_jacobian(
            self,
            mo_coeff=mo_coeff,
            mo_occ=mo_occ,
            ao_jacobian=ao_jacobian,
        )

    def coulomb_loss_descriptor_gradient(self, target_density):
        from .addons import coulomb_loss_descriptor_gradient

        return coulomb_loss_descriptor_gradient(self, target_density)

    def optimize_descriptor_potential(self, target_density, **kwargs):
        from .addons import optimize_descriptor_potential

        return optimize_descriptor_potential(self, target_density, **kwargs)


class RDeePKS(ModelCorrectionMixin, PenaltyMixin, gpu_rks.RKS):
    """Restricted self-consistent DeePKS method."""

    def __init__(
        self,
        mol,
        model,
        xc="HF",
        projector_basis=None,
        penalties=None,
        device=DEFAULT_DEVICE,
    ):
        gpu_rks.RKS.__init__(self, mol, xc=xc)
        self.direct_scf_tol = GPU_DIRECT_SCF_TOL
        ModelCorrectionMixin.__init__(
            self,
            model,
            projector_basis=projector_basis,
            device=device,
        )
        PenaltyMixin.__init__(self, penalties=penalties)
        self._keys.update(self.__dict__.keys())


class UDeePKS(ModelCorrectionMixin, PenaltyMixin, gpu_uks.UKS):
    """Unrestricted self-consistent DeePKS method."""

    def __init__(
        self,
        mol,
        model,
        xc="HF",
        projector_basis=None,
        penalties=None,
        device=DEFAULT_DEVICE,
    ):
        gpu_uks.UKS.__init__(self, mol, xc=xc)
        self.direct_scf_tol = GPU_DIRECT_SCF_TOL
        ModelCorrectionMixin.__init__(
            self,
            model,
            projector_basis=projector_basis,
            device=device,
        )
        PenaltyMixin.__init__(self, penalties=penalties)
        self._keys.update(self.__dict__.keys())

    def kernel(self, dm0=None, **kwargs):
        """Run GPU SCF and synchronize the unrestricted final state."""
        total_energy = super().kernel(dm0=dm0, **kwargs)
        if self.converged and str(self.xc).strip().upper() == "HF":
            total_energy = super().kernel(dm0=self.make_rdm1(), **kwargs)
        if self.mo_coeff is not None:
            fock = self.get_fock(dm=self.make_rdm1())
            transformed = torch_from_array(
                self.mo_coeff,
                device=self.model_device,
            ).conj().transpose(-2, -1) @ torch_from_array(
                fock,
                device=self.model_device,
            ) @ torch_from_array(
                self.mo_coeff,
                device=self.model_device,
            )
            self.mo_energy = cupy_from_torch(
                transformed.diagonal(dim1=-2, dim2=-1).real
            )
        return total_energy
