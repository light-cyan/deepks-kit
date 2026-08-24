"""Strict coupled scalar-adjoint nuclear gradients for UKS DeePHF."""

from types import MappingProxyType

import numpy as np

from .pyscf_uks import (
    UKSAdjointError,
    native_uks_gradient,
)
from .gradient import _validate_retain_details
from .uhf_zvector import UHFDeePHFZVectorGradients


class UKSDeePHFZVectorGradients(UHFDeePHFZVectorGradients):
    """Evaluate one finite-grid UKS correction through one coupled adjoint."""

    def __init__(self, method, adjoint_options=None, retain_details=True):
        from .uks_method import UKSDeePHF

        if type(method) is not UKSDeePHF:
            raise TypeError("the UKS Z-vector gradient driver requires an exact UKSDeePHF method")
        self._base = method
        self._bound_base = method
        self._mol = method.mol
        self._bound_mol = method.mol
        self._backend = "zvector"
        self.retain_details = _validate_retain_details(retain_details)
        self._adjoint_options = MappingProxyType(dict(adjoint_options or {}))
        self._bound_adjoint_options = self._adjoint_options
        self._reset_results()

    def _validate_driver_binding(self) -> None:
        from .uks_method import UKSDeePHF

        if (
            type(self._base) is not UKSDeePHF
            or self._base is not self._bound_base
            or self._mol is not self._bound_mol
            or self._mol is not self._base.mol
            or self._backend != "zvector"
            or self._adjoint_options is not self._bound_adjoint_options
            or not isinstance(self._adjoint_options, MappingProxyType)
        ):
            raise UKSAdjointError("the UKS DeePHF Z-vector driver binding is invalid")

    def _reset_results(self) -> None:
        super()._reset_results()

    def _kernel(self, atom_indices) -> dict:
        diagnostics, sensitivity, adjoint = self.base._zvector_inputs(
            self.adjoint_options,
            atom_indices=atom_indices,
        )
        self.base._validate_science_state("UKS Z-vector native gradient evaluation")
        native = native_uks_gradient(
            self.base.reference,
            atom_indices=atom_indices,
        )
        self.base._validate_science_state("UKS Z-vector native gradient evaluation")
        dq_explicit_spin = self.base.dq_dR_explicit_spin(
            atom_indices=atom_indices
        )
        self.base._validate_science_state("UKS Z-vector explicit descriptor gradient evaluation")
        dq_explicit = dq_explicit_spin.sum(axis=0)
        correction_explicit_spin = np.einsum("sbxap,ap->sbx", dq_explicit_spin, sensitivity)
        metric_spin = np.asarray(adjoint.correction_gradient_metric_spin)
        nuclear_spin = np.asarray(adjoint.correction_gradient_adjoint_nuclear_spin)
        adjoint_metric_spin = np.asarray(adjoint.correction_gradient_adjoint_metric_spin)
        ov_spin = np.asarray(adjoint.correction_gradient_occupied_virtual_spin)
        response_spin = metric_spin + ov_spin
        correction_spin = correction_explicit_spin + response_spin
        correction_explicit = correction_explicit_spin.sum(axis=0)
        metric = np.asarray(adjoint.correction_gradient_metric)
        nuclear = np.asarray(adjoint.correction_gradient_adjoint_nuclear)
        adjoint_metric = np.asarray(adjoint.correction_gradient_adjoint_metric)
        ov = np.asarray(adjoint.correction_gradient_occupied_virtual)
        correction_response = np.asarray(adjoint.correction_gradient_response)
        correction = correction_explicit + correction_response
        total = native + correction
        fixed_spin = np.asarray(adjoint.correction_gradient_adjoint_fixed_grid_spin)
        coordinate_spin = np.asarray(adjoint.correction_gradient_adjoint_grid_coordinate_spin)
        weight_spin = np.asarray(adjoint.correction_gradient_adjoint_grid_weight_spin)
        shape = (len(adjoint.core.atom_indices), 3)
        if total.shape != shape or not np.isfinite(total).all():
            raise UKSAdjointError("the UKS DeePHF Z-vector gradient is invalid")
        self.base._validate_science_state("UKS Z-vector gradient assembly")
        return {
            "adjoint_result": adjoint,
            "descriptor_diagnostics": diagnostics,
            "reference_gradient": native,
            "dq_dR_explicit_spin": dq_explicit_spin,
            "dq_dR_explicit": dq_explicit,
            "correction_gradient_explicit_spin": correction_explicit_spin,
            "correction_gradient_metric_spin": metric_spin,
            "correction_gradient_adjoint_nuclear_spin": nuclear_spin,
            "correction_gradient_adjoint_fixed_grid_spin": fixed_spin,
            "correction_gradient_adjoint_grid_coordinate_spin": coordinate_spin,
            "correction_gradient_adjoint_grid_weight_spin": weight_spin,
            "correction_gradient_adjoint_metric_spin": adjoint_metric_spin,
            "correction_gradient_occupied_virtual_spin": ov_spin,
            "correction_gradient_response_spin": response_spin,
            "correction_gradient_spin": correction_spin,
            "correction_gradient_explicit": correction_explicit,
            "correction_gradient_metric": metric,
            "correction_gradient_adjoint_nuclear": nuclear,
            "correction_gradient_adjoint_fixed_grid": np.asarray(adjoint.correction_gradient_adjoint_fixed_grid),
            "correction_gradient_adjoint_grid_coordinate": np.asarray(adjoint.correction_gradient_adjoint_grid_coordinate),
            "correction_gradient_adjoint_grid_weight": np.asarray(adjoint.correction_gradient_adjoint_grid_weight),
            "correction_gradient_adjoint_metric": adjoint_metric,
            "correction_gradient_occupied_virtual": ov,
            "correction_gradient_response": correction_response,
            "correction_gradient": correction,
            "de_full": total,
        }

    def _compact_kernel(self, atom_indices) -> dict:
        diagnostics, explicit, adjoint_diagnostics, response_gradient = self.base._zvector_inputs(
            self.adjoint_options,
            atom_indices=atom_indices,
            compact=True,
        )
        self.base._validate_science_state("UKS Z-vector native gradient evaluation")
        reference = native_uks_gradient(self.base.reference, atom_indices)
        self.base._validate_science_state("UKS Z-vector native gradient evaluation")
        total = reference + explicit + response_gradient
        if total.shape != reference.shape or not np.isfinite(total).all():
            raise UKSAdjointError("the compact UKS Z-vector gradient is invalid")
        self.base._validate_science_state("UKS Z-vector gradient assembly")
        return {
            "descriptor_diagnostics": diagnostics,
            "response_diagnostics": adjoint_diagnostics,
            "de": total,
        }

    def as_scanner(self, **scanner_options):
        raise UKSAdjointError("UKS DeePHF does not provide a gradient scanner")


__all__ = ["UKSDeePHFZVectorGradients"]
