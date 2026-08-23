"""Strict direct-oracle nuclear gradients for finite-grid UKS DeePHF."""

import numpy as np

from .pyscf_uks import UKSNativeGradient, UKSResponseError, native_uks_gradient
from .gradient import _validate_retain_details
from .uhf_gradient import UHFDeePHFGradients


class UKSDeePHFGradients(UHFDeePHFGradients):
    """Contract the complete coupled UKS response with one correction model."""

    def __init__(self, method, response_options=None, retain_details=True):
        from .uks_method import UKSDeePHF

        if type(method) is not UKSDeePHF:
            raise TypeError("the UKS direct gradient driver requires an exact UKSDeePHF method")
        self._base = method
        self._bound_base = method
        self._mol = method.mol
        self._bound_mol = method.mol
        self._backend = "direct"
        self.retain_details = _validate_retain_details(retain_details)
        self.response_options = dict(response_options or {})
        self._reset_results()

    def _validate_driver_binding(self) -> None:
        from .uks_method import UKSDeePHF

        if (
            type(self._base) is not UKSDeePHF
            or self._base is not self._bound_base
            or self._mol is not self._bound_mol
            or self._mol is not self._base.mol
            or self._backend != "direct"
        ):
            raise UKSResponseError("the UKS direct gradient driver binding is invalid")

    def _reset_results(self) -> None:
        super()._reset_results()

    def _validated_native_gradient(self, atom_indices) -> UKSNativeGradient:
        native = native_uks_gradient(
            self.base.reference,
            atom_indices=atom_indices,
        )
        if type(native) is not UKSNativeGradient:
            raise UKSResponseError("the native UKS gradient adapter returned an invalid result type")
        return native

    def _kernel(self, atom_indices) -> dict:
        descriptor_diagnostics, sensitivity = self.base._force_inputs()
        response = self.base._solve_response(
            self.response_options,
            atom_indices=atom_indices,
        )
        self.base._validate_science_state("UKS native gradient evaluation")
        native = self._validated_native_gradient(atom_indices)
        self.base._validate_science_state("UKS native gradient evaluation")
        dq_explicit_spin = self.base.dq_dR_explicit_spin(
            atom_indices=atom_indices
        )
        dq_dP = self.base.dq_dP()
        spin_density_response, metric_density, ov_density = (
            response.density_partitions()
        )
        dq_response_spin = np.einsum(
            "apij,sbxij->sbxap",
            dq_dP,
            spin_density_response,
        )
        dq_relaxed_spin = dq_explicit_spin + dq_response_spin
        dq_explicit = dq_explicit_spin.sum(axis=0)
        dq_response = dq_response_spin.sum(axis=0)
        dq_relaxed = dq_relaxed_spin.sum(axis=0)
        objective = self.base._correction_ao_potential(sensitivity, dq_dP)
        correction_explicit_spin = np.einsum("sbxap,ap->sbx", dq_explicit_spin, sensitivity)
        correction_metric_spin = np.einsum("ij,sbxij->sbx", objective, metric_density)
        correction_ov_spin = np.einsum("ij,sbxij->sbx", objective, ov_density)
        correction_response_spin = np.einsum("sbxap,ap->sbx", dq_response_spin, sensitivity)
        if not np.allclose(correction_response_spin, correction_metric_spin + correction_ov_spin, rtol=0.0, atol=1.0e-12):
            raise UKSResponseError("the UKS direct spin-response gradient partitions are inconsistent")
        correction_spin = correction_explicit_spin + correction_response_spin
        correction_explicit = correction_explicit_spin.sum(axis=0)
        correction_metric = correction_metric_spin.sum(axis=0)
        correction_ov = correction_ov_spin.sum(axis=0)
        correction_response = correction_response_spin.sum(axis=0)
        correction = correction_spin.sum(axis=0)
        if not np.allclose(correction, correction_explicit + correction_response, rtol=0.0, atol=1.0e-12):
            raise UKSResponseError("the UKS direct correction-gradient partitions are inconsistent")
        total = native.gradient + correction
        arrays = (dq_explicit_spin, dq_response_spin, dq_relaxed_spin, correction_spin, total)
        if any(not np.isfinite(value).all() for value in arrays):
            raise UKSResponseError("nonfinite UKS DeePHF direct-gradient quantity")
        self.base._validate_science_state("UKS direct gradient assembly")
        return {
            "response_result": response,
            "descriptor_diagnostics": descriptor_diagnostics,
            "native_gradient_result": native,
            "reference_gradient": native.gradient,
            "reference_gradient_without_grid_response": native.gradient_without_grid_response,
            "reference_gradient_xc_grid_coordinate": native.xc_grid_coordinate,
            "reference_gradient_xc_grid_weight": native.xc_grid_weight,
            "reference_gradient_reconstruction_residual": native.reconstruction_residual,
            "dq_dR_explicit_spin": dq_explicit_spin,
            "dq_dR_response_spin": dq_response_spin,
            "dq_dR_relaxed_spin": dq_relaxed_spin,
            "dq_dR_explicit": dq_explicit,
            "dq_dR_response": dq_response,
            "dq_dR_relaxed": dq_relaxed,
            "correction_gradient_explicit_spin": correction_explicit_spin,
            "correction_gradient_metric_spin": correction_metric_spin,
            "correction_gradient_occupied_virtual_spin": correction_ov_spin,
            "correction_gradient_response_spin": correction_response_spin,
            "correction_gradient_spin": correction_spin,
            "correction_gradient_explicit": correction_explicit,
            "correction_gradient_metric": correction_metric,
            "correction_gradient_occupied_virtual": correction_ov,
            "correction_gradient_response": correction_response,
            "correction_gradient": correction,
            "de_full": total,
        }

    def as_scanner(self, **scanner_options):
        raise UKSResponseError("UKS DeePHF does not provide a gradient scanner")


__all__ = ["UKSDeePHFGradients"]
