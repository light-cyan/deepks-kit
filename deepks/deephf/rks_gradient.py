"""Strict direct-oracle nuclear gradients for finite-grid RKS DeePHF."""

import numpy as np

from .capabilities import science_state_transaction
from .gradient import _validate_atom_indices
from .pyscf_rks import (
    RKSNativeGradient,
    RKSResponseError,
    native_rks_gradient,
)


class RKSDeePHFGradients:
    """Contract the complete pure-LDA RKS response with one correction model."""

    def __init__(self, method, response_options=None):
        from .rks_method import RKSDeePHF

        if type(method) is not RKSDeePHF:
            raise TypeError(
                "the RKS direct gradient driver requires an exact RKSDeePHF method"
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
        from .rks_method import RKSDeePHF

        if (
            type(self._base) is not RKSDeePHF
            or self._base is not self._bound_base
            or self._mol is not self._bound_mol
            or self._mol is not self._base.mol
            or self._backend != "direct"
        ):
            raise RKSResponseError(
                "the RKS direct gradient driver binding is invalid"
            )

    def _reset_results(self) -> None:
        self.response_result = None
        self.descriptor_diagnostics = None
        self.native_gradient_result = None
        self.reference_gradient = None
        self.reference_gradient_without_grid_response = None
        self.reference_gradient_xc_grid_coordinate = None
        self.reference_gradient_xc_grid_weight = None
        self.reference_gradient_reconstruction_residual = None
        self.dq_dR_explicit = None
        self.dq_dR_response = None
        self.dq_dR_relaxed = None
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

    def _validated_native_gradient(self) -> RKSNativeGradient:
        native = native_rks_gradient(self.base.reference)
        if type(native) is not RKSNativeGradient:
            raise RKSResponseError(
                "the native RKS gradient adapter returned an invalid result type"
            )
        expected_shape = (self.mol.natm, 3)
        partitions = {
            "complete native RKS gradient": native.gradient,
            "native RKS gradient without grid response": (
                native.gradient_without_grid_response
            ),
            "native RKS XC grid-coordinate gradient": native.xc_grid_coordinate,
            "native RKS XC grid-weight gradient": native.xc_grid_weight,
        }
        for name, value in partitions.items():
            if not isinstance(value, np.ndarray) or value.shape != expected_shape:
                raise RKSResponseError(
                    f"the {name} has shape {getattr(value, 'shape', None)}; "
                    f"expected {expected_shape}"
                )
            if value.dtype != np.dtype(np.float64) or np.iscomplexobj(value):
                raise RKSResponseError(f"the {name} must use real numpy.float64")
            if not np.isfinite(value).all():
                raise RKSResponseError(f"the {name} must be finite")
            if value.flags.writeable:
                raise RKSResponseError(f"the {name} must be immutable")
        measured_residual = float(
            np.max(
                np.abs(
                    native.gradient
                    - native.gradient_without_grid_response
                    - native.xc_grid_coordinate
                    - native.xc_grid_weight
                ),
                initial=0.0,
            )
        )
        if (
            isinstance(native.reconstruction_residual, (bool, np.bool_))
            or not isinstance(
                native.reconstruction_residual,
                (int, float, np.integer, np.floating),
            )
            or not np.isfinite(native.reconstruction_residual)
            or native.reconstruction_residual < 0.0
            or not np.isclose(
                native.reconstruction_residual,
                measured_residual,
                rtol=1.0e-12,
                atol=np.finfo(float).eps,
            )
        ):
            raise RKSResponseError(
                "the native RKS gradient reconstruction diagnostic is invalid"
            )
        return native

    def _kernel(self) -> dict:
        descriptor_diagnostics, sensitivity = self.base._force_inputs()
        response_result = self.base._solve_response(self.response_options)
        self.base._validate_science_state("RKS native gradient evaluation")
        native_gradient_result = self._validated_native_gradient()
        self.base._validate_science_state("RKS native gradient evaluation")
        dq_explicit = self.base.dq_dR_explicit()
        dq_dP = self.base.dq_dP()
        dq_response = np.einsum(
            "apij,bxij->bxap",
            dq_dP,
            response_result.density_response,
        )
        dq_relaxed = dq_explicit + dq_response
        objective_ao_potential = self.base._correction_ao_potential(
            sensitivity,
            dq_dP,
        )
        correction_explicit = np.einsum(
            "bxap,ap->bx",
            dq_explicit,
            sensitivity,
        )
        correction_metric = np.einsum(
            "ij,bxij->bx",
            objective_ao_potential,
            response_result.density_response_metric,
        )
        correction_occupied_virtual = np.einsum(
            "ij,bxij->bx",
            objective_ao_potential,
            response_result.density_response_occupied_virtual,
        )
        correction_response = np.einsum(
            "bxap,ap->bx",
            dq_response,
            sensitivity,
        )
        if not np.allclose(
            correction_response,
            correction_metric + correction_occupied_virtual,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise RKSResponseError(
                "the RKS direct response-gradient partitions are inconsistent"
            )
        correction = correction_explicit + correction_response
        de_full = native_gradient_result.gradient + correction
        arrays = {
            "explicit descriptor derivative": dq_explicit,
            "response descriptor derivative": dq_response,
            "relaxed descriptor derivative": dq_relaxed,
            "explicit correction gradient": correction_explicit,
            "metric correction gradient": correction_metric,
            "occupied-virtual correction gradient": (
                correction_occupied_virtual
            ),
            "response correction gradient": correction_response,
            "correction gradient": correction,
            "total gradient": de_full,
        }
        nonfinite = [
            name for name, value in arrays.items() if not np.isfinite(value).all()
        ]
        if nonfinite:
            raise RKSResponseError(
                "nonfinite RKS DeePHF gradient quantities: "
                + ", ".join(nonfinite)
            )
        self.base._validate_science_state("RKS gradient assembly")
        return {
            "response_result": response_result,
            "descriptor_diagnostics": descriptor_diagnostics,
            "native_gradient_result": native_gradient_result,
            "reference_gradient": native_gradient_result.gradient,
            "reference_gradient_without_grid_response": (
                native_gradient_result.gradient_without_grid_response
            ),
            "reference_gradient_xc_grid_coordinate": (
                native_gradient_result.xc_grid_coordinate
            ),
            "reference_gradient_xc_grid_weight": (
                native_gradient_result.xc_grid_weight
            ),
            "reference_gradient_reconstruction_residual": (
                native_gradient_result.reconstruction_residual
            ),
            "dq_dR_explicit": dq_explicit,
            "dq_dR_response": dq_response,
            "dq_dR_relaxed": dq_relaxed,
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
        """Reject unavailable RKS scanner construction."""
        raise RKSResponseError(
            "RKS DeePHF does not provide a gradient scanner"
        )


__all__ = ["RKSDeePHFGradients"]
