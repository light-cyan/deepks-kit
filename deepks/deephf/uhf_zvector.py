"""Strict coupled scalar-adjoint nuclear gradients for UHF DeePHF."""

from types import MappingProxyType

import numpy as np

from .capabilities import science_state_transaction
from .gradient import (
    _compact_driver_results,
    _reset_driver_results,
    _validate_atom_indices,
    _validate_retain_details,
)
from .pyscf_uhf import UHFAdjointError, _native_unrestricted_gradient


class UHFDeePHFZVectorGradients:
    """Evaluate one unrestricted correction through a coupled scalar adjoint."""

    def __init__(self, method, adjoint_options=None, retain_details=True):
        from .uhf_method import UHFDeePHF

        if type(method) is not UHFDeePHF:
            raise TypeError(
                "the UHF Z-vector gradient driver requires an exact UHFDeePHF method"
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
        from .uhf_method import UHFDeePHF

        if (
            type(self._base) is not UHFDeePHF
            or self._base is not self._bound_base
            or self._mol is not self._bound_mol
            or self._mol is not self._base.mol
            or self._backend != "zvector"
            or self._adjoint_options is not self._bound_adjoint_options
            or not isinstance(self._adjoint_options, MappingProxyType)
        ):
            raise UHFAdjointError(
                "the UHF DeePHF Z-vector driver binding is invalid"
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

    @staticmethod
    def _require_partition_close(actual, expected, label: str) -> None:
        if not np.allclose(actual, expected, rtol=0.0, atol=1.0e-12):
            raise UHFAdjointError(
                f"the UHF Z-vector {label} partitions are inconsistent"
            )

    def _validated_native_gradient(self, atom_indices) -> np.ndarray:
        self.base._validate_science_state(
            "UHF Z-vector native gradient evaluation"
        )
        try:
            gradient = _native_unrestricted_gradient(
                self.base.reference,
                self.base.reference.nuc_grad_method(),
                range(self.mol.natm) if atom_indices is None else atom_indices,
            )
        except Exception as error:
            raise UHFAdjointError(
                f"native UHF gradient evaluation failed: {error}"
            ) from error
        self.base._validate_science_state(
            "UHF Z-vector native gradient evaluation"
        )
        expected_shape = (
            self.mol.natm if atom_indices is None else len(atom_indices),
            3,
        )
        if gradient.shape != expected_shape:
            raise UHFAdjointError(
                f"the native UHF gradient has shape {gradient.shape}; "
                f"expected {expected_shape}"
            )
        if gradient.dtype != np.dtype(np.float64) or np.iscomplexobj(gradient):
            raise UHFAdjointError(
                "the native UHF gradient must use real numpy.float64"
            )
        if not np.isfinite(gradient).all():
            raise UHFAdjointError("the native UHF gradient must be finite")
        return gradient

    def _kernel(self, atom_indices) -> dict:
        descriptor_diagnostics, sensitivity, adjoint = self.base._zvector_inputs(
            self.adjoint_options,
            atom_indices=atom_indices,
        )
        reference_gradient = self._validated_native_gradient(atom_indices)
        dq_explicit_spin = self.base.dq_dR_explicit_spin(
            atom_indices=atom_indices
        )
        self.base._validate_science_state(
            "UHF Z-vector explicit descriptor gradient evaluation"
        )
        dq_explicit = dq_explicit_spin.sum(axis=0)
        correction_explicit_spin = np.einsum(
            "sbxap,ap->sbx",
            dq_explicit_spin,
            sensitivity,
        )
        correction_metric_spin = np.asarray(
            adjoint.correction_gradient_metric_spin
        )
        correction_adjoint_nuclear_spin = np.asarray(
            adjoint.correction_gradient_adjoint_nuclear_spin
        )
        correction_adjoint_metric_spin = np.asarray(
            adjoint.correction_gradient_adjoint_metric_spin
        )
        correction_occupied_virtual_spin = np.asarray(
            adjoint.correction_gradient_occupied_virtual_spin
        )
        correction_response_spin = (
            correction_metric_spin + correction_occupied_virtual_spin
        )
        correction_spin = correction_explicit_spin + correction_response_spin
        correction_explicit = correction_explicit_spin.sum(axis=0)
        correction_metric = np.asarray(adjoint.correction_gradient_metric)
        correction_adjoint_nuclear = np.asarray(
            adjoint.correction_gradient_adjoint_nuclear
        )
        correction_adjoint_metric = np.asarray(
            adjoint.correction_gradient_adjoint_metric
        )
        correction_occupied_virtual = np.asarray(
            adjoint.correction_gradient_occupied_virtual
        )
        correction_response = np.asarray(adjoint.correction_gradient_response)
        correction = correction_explicit + correction_response
        total = reference_gradient + correction
        natm = len(adjoint.atom_indices)
        expected_shape = (natm, 3)
        expected_spin_shape = (2, natm, 3)
        expected_descriptor_shape = (
            2,
            natm,
            3,
            self.base.n_descriptor_atoms,
            self.base.n_descriptor_features,
        )
        arrays = {
            "explicit descriptor spin derivative": (
                dq_explicit_spin,
                expected_descriptor_shape,
            ),
            "explicit correction spin gradient": (
                correction_explicit_spin,
                expected_spin_shape,
            ),
            "metric correction spin gradient": (
                correction_metric_spin,
                expected_spin_shape,
            ),
            "nuclear adjoint spin gradient": (
                correction_adjoint_nuclear_spin,
                expected_spin_shape,
            ),
            "metric adjoint spin gradient": (
                correction_adjoint_metric_spin,
                expected_spin_shape,
            ),
            "occupied-virtual adjoint spin gradient": (
                correction_occupied_virtual_spin,
                expected_spin_shape,
            ),
            "response correction spin gradient": (
                correction_response_spin,
                expected_spin_shape,
            ),
            "complete correction spin gradient": (
                correction_spin,
                expected_spin_shape,
            ),
            "native UHF gradient": (reference_gradient, expected_shape),
            "explicit correction gradient": (correction_explicit, expected_shape),
            "metric correction gradient": (correction_metric, expected_shape),
            "nuclear adjoint gradient": (
                correction_adjoint_nuclear,
                expected_shape,
            ),
            "metric adjoint gradient": (
                correction_adjoint_metric,
                expected_shape,
            ),
            "occupied-virtual correction gradient": (
                correction_occupied_virtual,
                expected_shape,
            ),
            "response correction gradient": (
                correction_response,
                expected_shape,
            ),
            "complete correction gradient": (correction, expected_shape),
            "UHF DeePHF Z-vector gradient": (total, expected_shape),
        }
        for name, (value, expected) in arrays.items():
            if not isinstance(value, np.ndarray) or value.shape != expected:
                raise UHFAdjointError(
                    f"the {name} has shape {getattr(value, 'shape', None)}; "
                    f"expected {expected}"
                )
            if value.dtype != np.dtype(np.float64) or np.iscomplexobj(value):
                raise UHFAdjointError(
                    f"the {name} must use real numpy.float64"
                )
            if not np.isfinite(value).all():
                raise UHFAdjointError(f"the {name} must be finite")
        self._require_partition_close(
            correction_adjoint_nuclear,
            correction_adjoint_nuclear_spin.sum(axis=0),
            "nuclear adjoint spin sum",
        )
        self._require_partition_close(
            correction_adjoint_metric,
            correction_adjoint_metric_spin.sum(axis=0),
            "metric adjoint spin sum",
        )
        self._require_partition_close(
            correction_occupied_virtual_spin,
            correction_adjoint_nuclear_spin + correction_adjoint_metric_spin,
            "occupied-virtual spin",
        )
        self._require_partition_close(
            correction_occupied_virtual,
            correction_occupied_virtual_spin.sum(axis=0),
            "occupied-virtual spin sum",
        )
        self._require_partition_close(
            correction_response,
            correction_metric + correction_occupied_virtual,
            "scalar-adjoint response",
        )
        self._require_partition_close(
            correction,
            correction_explicit + correction_response,
            "correction",
        )
        self._require_partition_close(
            correction,
            correction_spin.sum(axis=0),
            "correction spin sum",
        )
        self._require_partition_close(
            total,
            reference_gradient + correction,
            "total gradient",
        )
        self.base._validate_science_state("UHF Z-vector gradient assembly")
        return {
            "adjoint_result": adjoint,
            "descriptor_diagnostics": descriptor_diagnostics,
            "reference_gradient": reference_gradient,
            "dq_dR_explicit_spin": dq_explicit_spin,
            "dq_dR_explicit": dq_explicit,
            "correction_gradient_explicit_spin": correction_explicit_spin,
            "correction_gradient_metric_spin": correction_metric_spin,
            "correction_gradient_adjoint_nuclear_spin": (
                correction_adjoint_nuclear_spin
            ),
            "correction_gradient_adjoint_metric_spin": (
                correction_adjoint_metric_spin
            ),
            "correction_gradient_occupied_virtual_spin": (
                correction_occupied_virtual_spin
            ),
            "correction_gradient_response_spin": correction_response_spin,
            "correction_gradient_spin": correction_spin,
            "correction_gradient_explicit": correction_explicit,
            "correction_gradient_metric": correction_metric,
            "correction_gradient_adjoint_nuclear": correction_adjoint_nuclear,
            "correction_gradient_adjoint_metric": correction_adjoint_metric,
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
        """Reject unavailable unrestricted scanner construction."""
        raise UHFAdjointError(
            "UHF DeePHF does not provide a gradient scanner"
        )


__all__ = ["UHFDeePHFZVectorGradients"]
