"""Perturbative DeePHF energy method composed around a native reference."""

import numpy as np
import torch

from deepks.descriptor import AtomicDensityDescriptor, validate_differentiability
from deepks.model.evaluate import correction as evaluate_correction
from deepks.model.evaluate import descriptor_sensitivity
from deepks.model.model import CorrNet

from .capabilities import validate_reference


class DeePHF:
    """Evaluate a perturbative correction without modifying the RHF reference."""

    def __init__(
        self,
        reference,
        model,
        projector_basis=None,
        device="cpu",
    ):
        self.reference = validate_reference(reference)
        self.device = device or "cpu"
        if isinstance(model, str):
            model = CorrNet.load(model).double()
        if model is not None and not isinstance(model, torch.nn.Module):
            raise TypeError("model must be a torch.nn.Module, model path, or None")
        if isinstance(model, torch.nn.Module):
            model = model.to(self.device).eval()
        self.model = model
        if projector_basis is None:
            projector_basis = getattr(model, "_pbas", None)
        self._descriptor = AtomicDensityDescriptor(
            self.reference.mol,
            projector_basis,
        )
        self.e_base = None
        self.e_corr = None
        self.e_tot = None

    @property
    def mol(self):
        return self.reference.mol

    @property
    def n_descriptor_atoms(self):
        return self._descriptor.n_descriptor_atoms

    @property
    def n_descriptor_features(self):
        return self._descriptor.n_features

    def ao_density(self):
        return np.asarray(self.reference.make_rdm1())

    def projected_density(self, flatten=False):
        return self._descriptor.projected_density(
            self.ao_density(),
            flatten=flatten,
        )

    def descriptor(self):
        return self._descriptor.descriptor(self.ao_density())

    def dq_dP(self):
        return self._descriptor.dq_dP(self.ao_density())

    def dq_dR_explicit(self):
        return self._descriptor.dq_dR_explicit(self.ao_density())

    def correction_energy(self):
        if self.model is None:
            return 0.0
        tensor_density = torch.from_numpy(self.ao_density()).double()
        tensor_energy = evaluate_correction(
            self.model,
            tensor_density,
            self._descriptor.overlap_shells,
            with_potential=False,
        )
        energy = float(tensor_energy.detach().cpu().reshape(-1)[0])
        get_element_constant = getattr(self.model, "get_elem_const", None)
        if get_element_constant is not None:
            energy += get_element_constant(
                [int(charge) for charge in self.mol.atom_charges()]
            )
        return energy

    def validate_force_compatibility(self, **tolerances):
        """Validate ordered-eigenvalue and model-sensitivity force semantics."""
        validate_reference(self.reference)
        values = self._descriptor.torch_descriptor(self.ao_density())
        sensitivity = (
            torch.zeros_like(values)
            if self.model is None
            else descriptor_sensitivity(self.model, values)
        )
        n_occupied = int(np.count_nonzero(self.reference.mo_occ > 0))
        return validate_differentiability(
            values.detach().cpu().numpy(),
            self._descriptor.shell_sizes,
            n_occupied,
            sensitivity.detach().cpu().numpy(),
            **tolerances,
        )

    def kernel(self):
        """Evaluate E_base + E_corr while leaving the reference unchanged."""
        validate_reference(self.reference)
        self.e_base = float(self.reference.e_tot)
        self.e_corr = self.correction_energy()
        self.e_tot = self.e_base + self.e_corr
        return self.e_tot
