"""Self-consistent DeePKS mean-field methods."""

import abc
import time

import numpy as np
import torch
from pyscf import dft, lib
from pyscf.lib import logger

from deepks.descriptor import AtomicDensityDescriptor, spin_summed_ao_density
from deepks.model.evaluate import correction as evaluate_correction
from deepks.model.model import CorrNet

from .penalty import PenaltyMixin


DEFAULT_DEVICE = "cpu"


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
        return lib.tag_array(
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
        self.device = device or DEFAULT_DEVICE
        if isinstance(model, str):
            model = CorrNet.load(model).double()
        if isinstance(model, torch.nn.Module):
            model = model.to(self.device).eval()
        self.model = model
        if projector_basis is None:
            projector_basis = getattr(model, "_pbas", None)
        self._descriptor = AtomicDensityDescriptor(self.mol, projector_basis)

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
            return 0.0, np.zeros_like(ao_density)
        tensor_density = torch.from_numpy(ao_density).double()
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
        potential = tensor_potential.detach().cpu().numpy()
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


class RDeePKS(ModelCorrectionMixin, PenaltyMixin, dft.rks.RKS):
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
        dft.rks.RKS.__init__(self, mol, xc=xc)
        ModelCorrectionMixin.__init__(
            self,
            model,
            projector_basis=projector_basis,
            device=device,
        )
        PenaltyMixin.__init__(self, penalties=penalties)
        self._keys.update(self.__dict__.keys())


class UDeePKS(ModelCorrectionMixin, PenaltyMixin, dft.uks.UKS):
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
        dft.uks.UKS.__init__(self, mol, xc=xc)
        ModelCorrectionMixin.__init__(
            self,
            model,
            projector_basis=projector_basis,
            device=device,
        )
        PenaltyMixin.__init__(self, penalties=penalties)
        self._keys.update(self.__dict__.keys())
