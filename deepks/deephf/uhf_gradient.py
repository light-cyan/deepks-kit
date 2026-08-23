"""Strict direct-oracle nuclear gradients for UHF DeePHF."""

import numpy as np

from .capabilities import science_state_transaction
from .gradient import _validate_atom_indices
from .pyscf_uhf import UHFResponseError


class UHFDeePHFGradients:
    """Contract the complete coupled UHF response with one correction model."""

    def __init__(self, method, response_options=None):
        from .uhf_method import UHFDeePHF

        if type(method) is not UHFDeePHF:
            raise TypeError(
                "the UHF direct gradient driver requires an exact UHFDeePHF method"
            )
        self._base = method
        self._bound_base = method
        self._mol = method.mol
        self._bound_mol = method.mol
        self._backend = "direct"
        self.response_options = dict(response_options or {})
        self._reset_results()

    @property
    def base(self):
        return self._base

    @property
    def mol(self):
        return self._mol

    @property
    def backend(self) -> str:
        return self._backend

    def _validate_driver_binding(self) -> None:
        from .uhf_method import UHFDeePHF

        if (
            type(self._base) is not UHFDeePHF
            or self._base is not self._bound_base
            or self._mol is not self._bound_mol
            or self._mol is not self._base.mol
            or self._backend != "direct"
        ):
            raise UHFResponseError(
                "the UHF direct gradient driver binding is invalid"
            )

    def _reset_results(self) -> None:
        self.response_result = None
        self.descriptor_diagnostics = None
        self.reference_gradient = None
        self.dq_dR_explicit_spin = None
        self.dq_dR_response_spin = None
        self.dq_dR_relaxed_spin = None
        self.dq_dR_explicit = None
        self.dq_dR_response = None
        self.dq_dR_relaxed = None
        self.correction_gradient_explicit_spin = None
        self.correction_gradient_metric_spin = None
        self.correction_gradient_occupied_virtual_spin = None
        self.correction_gradient_response_spin = None
        self.correction_gradient_spin = None
        self.correction_gradient_explicit = None
        self.correction_gradient_metric = None
        self.correction_gradient_occupied_virtual = None
        self.correction_gradient_response = None
        self.correction_gradient = None
        self.de_full = None
        self.de = None

    @property
    def response_diagnostics(self):
        if self.response_result is None:
            return None
        return self.response_result.diagnostics

    def _kernel(self) -> dict:
        descriptor_diagnostics, sensitivity = self.base._force_inputs()
        response_result = self.base._solve_response(self.response_options)
        self.base._validate_science_state("UHF native gradient evaluation")
        reference_gradient = np.asarray(
            self.base.reference.nuc_grad_method().kernel()
        )
        self.base._validate_science_state("UHF native gradient evaluation")
        dq_explicit_spin = self.base.dq_dR_explicit_spin()
        dq_dP = self.base.dq_dP()
        spin_density_response = np.stack(
            (
                response_result.alpha_density_response,
                response_result.beta_density_response,
            )
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
        objective_ao_potential = self.base._correction_ao_potential(
            sensitivity,
            dq_dP,
        )
        correction_explicit_spin = np.einsum(
            "sbxap,ap->sbx",
            dq_explicit_spin,
            sensitivity,
        )
        metric_density = np.stack(
            (
                response_result.alpha_density_response_metric,
                response_result.beta_density_response_metric,
            )
        )
        occupied_virtual_density = np.stack(
            (
                response_result.alpha_density_response_occupied_virtual,
                response_result.beta_density_response_occupied_virtual,
            )
        )
        correction_metric_spin = np.einsum(
            "ij,sbxij->sbx",
            objective_ao_potential,
            metric_density,
        )
        correction_occupied_virtual_spin = np.einsum(
            "ij,sbxij->sbx",
            objective_ao_potential,
            occupied_virtual_density,
        )
        correction_response_spin = np.einsum(
            "sbxap,ap->sbx",
            dq_response_spin,
            sensitivity,
        )
        if not np.allclose(
            correction_response_spin,
            correction_metric_spin + correction_occupied_virtual_spin,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise UHFResponseError(
                "the UHF direct spin-response gradient partitions are inconsistent"
            )
        correction_spin = correction_explicit_spin + correction_response_spin
        correction_explicit = correction_explicit_spin.sum(axis=0)
        correction_metric = correction_metric_spin.sum(axis=0)
        correction_occupied_virtual = correction_occupied_virtual_spin.sum(axis=0)
        correction_response = correction_response_spin.sum(axis=0)
        correction = correction_spin.sum(axis=0)
        if not np.allclose(
            correction,
            correction_explicit + correction_response,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise UHFResponseError(
                "the UHF direct correction-gradient partitions are inconsistent"
            )
        de_full = reference_gradient + correction
        arrays = {
            "reference gradient": reference_gradient,
            "explicit descriptor derivative": dq_explicit_spin,
            "response descriptor derivative": dq_response_spin,
            "relaxed descriptor derivative": dq_relaxed_spin,
            "correction gradient": correction_spin,
            "total gradient": de_full,
        }
        nonfinite = [
            name for name, value in arrays.items() if not np.isfinite(value).all()
        ]
        if nonfinite:
            raise UHFResponseError(
                "nonfinite UHF DeePHF gradient quantities: "
                + ", ".join(nonfinite)
            )
        self.base._validate_science_state("UHF gradient assembly")
        return {
            "response_result": response_result,
            "descriptor_diagnostics": descriptor_diagnostics,
            "reference_gradient": reference_gradient,
            "dq_dR_explicit_spin": dq_explicit_spin,
            "dq_dR_response_spin": dq_response_spin,
            "dq_dR_relaxed_spin": dq_relaxed_spin,
            "dq_dR_explicit": dq_explicit,
            "dq_dR_response": dq_response,
            "dq_dR_relaxed": dq_relaxed,
            "correction_gradient_explicit_spin": correction_explicit_spin,
            "correction_gradient_metric_spin": correction_metric_spin,
            "correction_gradient_occupied_virtual_spin": (
                correction_occupied_virtual_spin
            ),
            "correction_gradient_response_spin": correction_response_spin,
            "correction_gradient_spin": correction_spin,
            "correction_gradient_explicit": correction_explicit,
            "correction_gradient_metric": correction_metric,
            "correction_gradient_occupied_virtual": (
                correction_occupied_virtual
            ),
            "correction_gradient_response": correction_response,
            "correction_gradient": correction,
            "de_full": de_full,
        }

    @science_state_transaction
    def kernel(self, atmlst=None) -> np.ndarray:
        """Evaluate d(E_base + E_corr)/dR for all or selected atoms."""
        self._reset_results()
        try:
            self._validate_driver_binding()
            atom_indices = _validate_atom_indices(self.mol, atmlst)
            results = self._kernel()
            for name, value in results.items():
                setattr(self, name, value)
            if atom_indices is None:
                self.de = self.de_full
            else:
                self.de = self.de_full[list(atom_indices)]
            return self.de
        except Exception:
            self._reset_results()
            raise

    def run(self, atmlst=None):
        """Evaluate the gradient and return this result object."""
        self.kernel(atmlst=atmlst)
        return self

    def forces(self, atmlst=None) -> np.ndarray:
        """Evaluate nuclear forces as minus the energy gradient."""
        return -self.kernel(atmlst=atmlst)

    def as_scanner(self, **scanner_options):
        """Reject unavailable unrestricted scanner construction."""
        raise UHFResponseError(
            "UHF DeePHF does not provide a gradient scanner"
        )


__all__ = ["UHFDeePHFGradients"]
