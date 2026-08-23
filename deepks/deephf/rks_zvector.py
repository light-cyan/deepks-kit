"""Strict scalar-adjoint nuclear gradients for finite-grid RKS DeePHF."""

from types import MappingProxyType

import numpy as np

from .capabilities import science_state_transaction
from .gradient import (
    _compact_driver_results,
    _reset_driver_results,
    _validate_atom_indices,
    _validate_retain_details,
)
from .pyscf_rks import (
    RKSAdjointError,
    RKSNativeGradient,
    native_rks_gradient,
)


class RKSDeePHFZVectorGradients:
    """Evaluate one pure-LDA RKS correction through a scalar adjoint."""

    def __init__(self, method, adjoint_options=None, retain_details=True):
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
        self.retain_details = _validate_retain_details(retain_details)
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
        _reset_driver_results(self)

    @property
    def adjoint_diagnostics(self):
        return (
            self._response_diagnostics
            if getattr(self, "adjoint_result", None) is None
            else self.adjoint_result.diagnostics
        )

    @property
    def response_diagnostics(self):
        """Return scalar-adjoint diagnostics under the common driver name."""
        return self.adjoint_diagnostics

    def _validated_native_gradient(self, atom_indices) -> RKSNativeGradient:
        native = native_rks_gradient(self.base.reference, atom_indices=atom_indices)
        if type(native) is not RKSNativeGradient:
            raise RKSAdjointError(
                "the native RKS gradient adapter returned an invalid result type"
            )
        expected_shape = (
            self.mol.natm if atom_indices is None else len(atom_indices),
            3,
        )
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

    def _kernel(self, atom_indices) -> dict:
        descriptor_diagnostics, sensitivity, adjoint = self.base._zvector_inputs(
            self.adjoint_options,
            atom_indices=atom_indices,
        )
        self.base._validate_science_state(
            "RKS Z-vector native gradient evaluation"
        )
        native = self._validated_native_gradient(atom_indices)
        self.base._validate_science_state(
            "RKS Z-vector native gradient evaluation"
        )
        dq_explicit = self.base.dq_dR_explicit(atom_indices=atom_indices)
        self.base._validate_science_state(
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
        expected_shape = (len(adjoint.atom_indices), 3)
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
            len(adjoint.atom_indices),
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

    @science_state_transaction
    def kernel(self, atmlst=None) -> np.ndarray:
        """Evaluate d(E_base + E_corr)/dR without constructing dP/dR."""
        self._reset_results()
        try:
            self._validate_driver_binding()
            atom_indices = _validate_atom_indices(self.mol, atmlst)
            results = self._kernel(atom_indices)
            for name, value in results.items():
                setattr(self, name, value)
            self.de = self.de_full
            if not self.retain_details:
                _compact_driver_results(self)
            return self.de
        except Exception:
            self._reset_results()
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
