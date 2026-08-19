"""Strict direct-oracle nuclear gradients for RHF DeePHF."""

import operator

import numpy as np

from .pyscf_rhf import RHFResponseError


class RHFDeePHFGradients:
    """Contract the complete relaxed descriptor response with one correction model."""

    def __init__(self, method, response_options=None):
        self.base = method
        self.mol = method.mol
        self.response_options = dict(response_options or {})
        self.response_result = None
        self.descriptor_diagnostics = None
        self.reference_gradient = None
        self.dq_dR_explicit = None
        self.dq_dR_response = None
        self.dq_dR_relaxed = None
        self.correction_gradient_explicit = None
        self.correction_gradient_response = None
        self.correction_gradient = None
        self.de_full = None
        self.de = None

    @property
    def response_diagnostics(self):
        if self.response_result is None:
            return None
        return self.response_result.diagnostics

    def kernel(self, atmlst=None) -> np.ndarray:
        """Evaluate d(E_base + E_corr)/dR for all or selected atoms."""
        atom_indices = None
        if atmlst is not None:
            try:
                requested_indices = tuple(atmlst)
            except TypeError as error:
                raise TypeError("gradient atmlst must be an iterable of integers") from error
            validated_indices = []
            for index in requested_indices:
                if isinstance(index, (bool, np.bool_)):
                    raise TypeError("gradient atom indices must be integers")
                try:
                    atom_index = operator.index(index)
                except TypeError as error:
                    raise TypeError("gradient atom indices must be integers") from error
                if atom_index < 0 or atom_index >= self.mol.natm:
                    raise IndexError("gradient atom index is outside the molecule")
                validated_indices.append(atom_index)
            atom_indices = tuple(validated_indices)
        self.descriptor_diagnostics = self.base.validate_force_compatibility()
        self.response_result = self.base.response(**self.response_options)
        self.reference_gradient = np.asarray(
            self.base.reference.nuc_grad_method().kernel()
        )
        self.dq_dR_explicit = self.base.dq_dR_explicit()
        self.dq_dR_response = self.base.dq_dR_response(
            response=self.response_result
        )
        self.dq_dR_relaxed = self.dq_dR_explicit + self.dq_dR_response
        sensitivity = self.base.correction_sensitivity()
        self.correction_gradient_explicit = np.einsum(
            "bxap,ap->bx",
            self.dq_dR_explicit,
            sensitivity,
        )
        self.correction_gradient_response = np.einsum(
            "bxap,ap->bx",
            self.dq_dR_response,
            sensitivity,
        )
        self.correction_gradient = (
            self.correction_gradient_explicit
            + self.correction_gradient_response
        )
        self.de_full = self.reference_gradient + self.correction_gradient
        if not np.isfinite(self.de_full).all():
            raise RHFResponseError("the RHF DeePHF analytic gradient is nonfinite")
        if atom_indices is None:
            self.de = self.de_full
        else:
            self.de = self.de_full[list(atom_indices)]
        return self.de

    def run(self, atmlst=None):
        """Evaluate the gradient and return this result object."""
        self.kernel(atmlst=atmlst)
        return self

    def forces(self, atmlst=None) -> np.ndarray:
        """Evaluate nuclear forces as minus the energy gradient."""
        return -self.kernel(atmlst=atmlst)
