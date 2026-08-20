"""Strict scalar-adjoint nuclear gradients for finite-grid RKS DeePHF."""

from types import MappingProxyType

import numpy as np

from .gradient import _validate_atom_indices
from .pyscf_rks import (
    RKSAdjointError,
    RKSNativeGradient,
    native_rks_gradient,
)


class RKSDeePHFZVectorGradients:
    """Evaluate one pure-LDA RKS correction through a scalar adjoint."""

    def __init__(self, method, adjoint_options=None):
        from .rks_method import RKSDeePHF

        if type(method) is not RKSDeePHF:
            raise TypeError(
                "the RKS Z-vector gradient driver requires an exact RKSDeePHF method"
            )
        self._base = method
        self._bound_base = method
        self._mol = method.mol
        self._bound_mol = method.mol
        self._backend = "zvector"
        self._adjoint_options = MappingProxyType(dict(adjoint_options or {}))
        self._bound_adjoint_options = self._adjoint_options
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

    @property
    def adjoint_options(self):
        return self._adjoint_options

    def _validate_driver_binding(self) -> None:
        from .rks_method import RKSDeePHF

        if (
            type(self._base) is not RKSDeePHF
            or self._base is not self._bound_base
            or self._mol is not self._bound_mol
            or self._mol is not self._base.mol
            or self._backend != "zvector"
            or self._adjoint_options is not self._bound_adjoint_options
            or not isinstance(self._adjoint_options, MappingProxyType)
        ):
            raise RKSAdjointError(
                "the RKS DeePHF Z-vector driver binding is invalid"
            )

    def _reset_results(self) -> None:
        self.adjoint_result = None
        self.descriptor_diagnostics = None
        self.native_gradient_result = None
        self.reference_gradient = None
        self.reference_gradient_without_grid_response = None
        self.reference_gradient_xc_grid_coordinate = None
        self.reference_gradient_xc_grid_weight = None
        self.reference_gradient_reconstruction_residual = None
        self.dq_dR_explicit = None
        self.correction_gradient_explicit = None
        self.correction_gradient_metric = None
        self.correction_gradient_adjoint_fixed_grid = None
        self.correction_gradient_adjoint_grid_coordinate = None
        self.correction_gradient_adjoint_grid_weight = None
        self.correction_gradient_adjoint_nuclear = None
        self.correction_gradient_adjoint_metric = None
        self.correction_gradient_grid_coordinate = None
        self.correction_gradient_grid_weight = None
        self.correction_gradient_occupied_virtual = None
        self.correction_gradient_response = None
        self.correction_gradient = None
        self.de_full = None
        self.de = None

    @property
    def adjoint_diagnostics(self):
        if self.adjoint_result is None:
            return None
        return self.adjoint_result.diagnostics

    @property
    def response_diagnostics(self):
        """Return scalar-adjoint diagnostics under the common driver name."""
        return self.adjoint_diagnostics

    def _validated_native_gradient(self) -> RKSNativeGradient:
        native = native_rks_gradient(self.base.reference)
        if type(native) is not RKSNativeGradient:
            raise RKSAdjointError(
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
                raise RKSAdjointError(
                    f"the {name} has shape {getattr(value, 'shape', None)}; "
                    f"expected {expected_shape}"
                )
            if value.dtype != np.dtype(np.float64) or np.iscomplexobj(value):
                raise RKSAdjointError(
                    f"the {name} must use real numpy.float64"
                )
            if not np.isfinite(value).all():
                raise RKSAdjointError(f"the {name} must be finite")
            if value.flags.writeable:
                raise RKSAdjointError(f"the {name} must be immutable")
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
        diagnostic = native.reconstruction_residual
        if (
            isinstance(diagnostic, (bool, np.bool_))
            or not isinstance(
                diagnostic,
                (int, float, np.integer, np.floating),
            )
            or not np.isfinite(diagnostic)
            or diagnostic < 0.0
            or not np.isclose(
                diagnostic,
                measured_residual,
                rtol=1.0e-12,
                atol=np.finfo(float).eps,
            )
        ):
            raise RKSAdjointError(
                "the native RKS gradient reconstruction diagnostic is invalid"
            )
        return native

    @staticmethod
    def _require_partition_close(actual, expected, label: str) -> None:
        if not np.allclose(actual, expected, rtol=0.0, atol=1.0e-12):
            raise RKSAdjointError(
                f"the RKS Z-vector {label} partitions are inconsistent"
            )

    def _kernel(self) -> dict:
        descriptor_diagnostics, sensitivity, adjoint = self.base._zvector_inputs(
            self.adjoint_options
        )
        descriptor_diagnostics, sensitivity, adjoint = (
            self.base._validate_zvector_inputs(
                descriptor_diagnostics,
                sensitivity,
                adjoint,
            )
        )
        self.base._validate_science_state(
            "RKS Z-vector native gradient evaluation"
        )
        self.base._assert_trusted_rks_force_model_state(
            "RKS Z-vector native gradient evaluation"
        )
        native = self._validated_native_gradient()
        self.base._validate_science_state(
            "RKS Z-vector native gradient evaluation"
        )
        self.base._assert_trusted_rks_force_model_state(
            "RKS Z-vector native gradient evaluation"
        )
        dq_explicit = self.base.dq_dR_explicit()
        self.base._validate_science_state(
            "RKS Z-vector explicit descriptor gradient evaluation"
        )
        self.base._assert_trusted_rks_force_model_state(
            "RKS Z-vector explicit descriptor gradient evaluation"
        )
        correction_explicit = np.einsum(
            "bxap,ap->bx",
            dq_explicit,
            sensitivity,
        )
        correction_metric = np.asarray(adjoint.correction_gradient_metric)
        correction_adjoint_fixed_grid = np.asarray(
            adjoint.correction_gradient_adjoint_fixed_grid
        )
        correction_adjoint_grid_coordinate = np.asarray(
            adjoint.correction_gradient_adjoint_grid_coordinate
        )
        correction_adjoint_grid_weight = np.asarray(
            adjoint.correction_gradient_adjoint_grid_weight
        )
        correction_adjoint_nuclear = np.asarray(
            adjoint.correction_gradient_adjoint_nuclear
        )
        correction_adjoint_metric = np.asarray(
            adjoint.correction_gradient_adjoint_metric
        )
        correction_occupied_virtual = np.asarray(
            adjoint.correction_gradient_occupied_virtual
        )
        correction_response = np.asarray(
            adjoint.correction_gradient_response
        )
        correction = correction_explicit + correction_response
        total = native.gradient + correction
        expected_shape = (self.mol.natm, 3)
        arrays = {
            "explicit descriptor derivative": dq_explicit,
            "explicit correction gradient": correction_explicit,
            "metric correction gradient": correction_metric,
            "fixed-grid adjoint correction gradient": (
                correction_adjoint_fixed_grid
            ),
            "grid-coordinate adjoint correction gradient": (
                correction_adjoint_grid_coordinate
            ),
            "grid-weight adjoint correction gradient": (
                correction_adjoint_grid_weight
            ),
            "nuclear adjoint correction gradient": correction_adjoint_nuclear,
            "metric adjoint correction gradient": correction_adjoint_metric,
            "occupied-virtual correction gradient": (
                correction_occupied_virtual
            ),
            "response correction gradient": correction_response,
            "complete correction gradient": correction,
            "RKS DeePHF Z-vector gradient": total,
        }
        expected_descriptor_shape = (
            self.mol.natm,
            3,
            self.base.n_descriptor_atoms,
            self.base.n_descriptor_features,
        )
        for name, value in arrays.items():
            expected = (
                expected_descriptor_shape
                if name == "explicit descriptor derivative"
                else expected_shape
            )
            if not isinstance(value, np.ndarray) or value.shape != expected:
                raise RKSAdjointError(
                    f"the {name} has shape {getattr(value, 'shape', None)}; "
                    f"expected {expected}"
                )
            if value.dtype != np.dtype(np.float64) or np.iscomplexobj(value):
                raise RKSAdjointError(
                    f"the {name} must use real numpy.float64"
                )
            if not np.isfinite(value).all():
                raise RKSAdjointError(f"the {name} must be finite")
        self._require_partition_close(
            correction_adjoint_nuclear,
            correction_adjoint_fixed_grid
            + correction_adjoint_grid_coordinate
            + correction_adjoint_grid_weight,
            "nuclear adjoint",
        )
        self._require_partition_close(
            correction_occupied_virtual,
            correction_adjoint_nuclear + correction_adjoint_metric,
            "occupied-virtual adjoint",
        )
        self._require_partition_close(
            correction_response,
            correction_metric + correction_occupied_virtual,
            "scalar-adjoint response",
        )
        self._require_partition_close(
            correction,
            correction_explicit + correction_response,
            "correction gradient",
        )
        self._require_partition_close(
            total,
            native.gradient + correction,
            "total gradient",
        )
        self.base._validate_science_state("RKS Z-vector gradient assembly")
        self.base._assert_trusted_rks_force_model_state(
            "RKS Z-vector gradient assembly"
        )
        return {
            "adjoint_result": adjoint,
            "descriptor_diagnostics": descriptor_diagnostics,
            "native_gradient_result": native,
            "reference_gradient": native.gradient,
            "reference_gradient_without_grid_response": (
                native.gradient_without_grid_response
            ),
            "reference_gradient_xc_grid_coordinate": native.xc_grid_coordinate,
            "reference_gradient_xc_grid_weight": native.xc_grid_weight,
            "reference_gradient_reconstruction_residual": (
                native.reconstruction_residual
            ),
            "dq_dR_explicit": dq_explicit,
            "correction_gradient_explicit": correction_explicit,
            "correction_gradient_metric": correction_metric,
            "correction_gradient_adjoint_fixed_grid": (
                correction_adjoint_fixed_grid
            ),
            "correction_gradient_adjoint_grid_coordinate": (
                correction_adjoint_grid_coordinate
            ),
            "correction_gradient_adjoint_grid_weight": (
                correction_adjoint_grid_weight
            ),
            "correction_gradient_adjoint_nuclear": correction_adjoint_nuclear,
            "correction_gradient_adjoint_metric": correction_adjoint_metric,
            "correction_gradient_grid_coordinate": (
                correction_adjoint_grid_coordinate
            ),
            "correction_gradient_grid_weight": correction_adjoint_grid_weight,
            "correction_gradient_occupied_virtual": (
                correction_occupied_virtual
            ),
            "correction_gradient_response": correction_response,
            "correction_gradient": correction,
            "de_full": total,
        }

    def kernel(self, atmlst=None) -> np.ndarray:
        """Evaluate d(E_base + E_corr)/dR without constructing dP/dR."""
        self._reset_results()
        self._bound_base._clear_trusted_adjoint()
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
            self._bound_base._clear_trusted_adjoint()
            raise

    def run(self, atmlst=None):
        """Evaluate the gradient and return this populated driver."""
        self.kernel(atmlst=atmlst)
        return self

    def forces(self, atmlst=None) -> np.ndarray:
        """Evaluate nuclear forces as minus the energy gradient."""
        return -self.kernel(atmlst=atmlst)

    def as_scanner(self, **scanner_options):
        """Reject unavailable RKS scanner construction."""
        raise RKSAdjointError(
            "RKS DeePHF does not provide a gradient scanner"
        )


__all__ = ["RKSDeePHFZVectorGradients"]
