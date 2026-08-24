"""Strict scalar-adjoint nuclear gradients for RHF DeePHF."""

import numpy as np

from .capabilities import science_state_transaction
from .gradient import (
    _reset_driver_results,
    _validate_atom_indices,
    _validate_retain_details,
)
from .pyscf_rhf import RHFAdjointError


class RHFDeePHFZVectorGradients:
    """Evaluate the correction response through one RHF scalar adjoint."""

    def __init__(self, method, adjoint_options=None, retain_details=True):
        from .method import DeePHF

        if type(method) is not DeePHF:
            raise TypeError("the Z-vector driver requires an exact DeePHF method")
        self._base = method
        self._bound_base = method
        self._mol = method.mol
        self._bound_mol = method.mol
        self._backend = "zvector"
        self.retain_details = _validate_retain_details(retain_details)
        self.response_options = dict(adjoint_options or {})
        self._reset_results()

    @property
    def base(self):
        return self._base

    @property
    def mol(self):
        return self._mol

    @property
    def backend(self):
        return self._backend

    def _validate_driver_binding(self) -> None:
        from .method import DeePHF

        if (
            type(self._base) is not DeePHF
            or self._base is not self._bound_base
            or self._mol is not self._bound_mol
            or self._mol is not self._base.mol
            or self._backend != "zvector"
        ):
            raise RHFAdjointError(
                "the RHF DeePHF Z-vector driver binding is invalid"
            )

    def _reset_results(self):
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
        """Return the scalar-adjoint diagnostics under the common driver name."""
        return self.adjoint_diagnostics

    def _compact_kernel(self, atom_indices):
        descriptor_diagnostics, sensitivity, adjoint = self.base._zvector_inputs(
            self.response_options,
            atom_indices=atom_indices,
        )
        self.base._assert_science_state("native RHF gradient evaluation")
        reference_gradient = np.asarray(
            self.base.reference.nuc_grad_method().kernel(
                atmlst=None if atom_indices is None else list(atom_indices)
            )
        )
        self.base._validate_science_state("native RHF gradient evaluation")
        explicit = self.base._correction_gradient_explicit(
            sensitivity,
            atom_indices,
        )
        total = reference_gradient + explicit + adjoint.correction_gradient_response
        if total.shape != (len(adjoint.atom_indices), 3) or not np.isfinite(total).all():
            raise RHFAdjointError("the compact RHF Z-vector gradient is invalid")
        return descriptor_diagnostics, adjoint.diagnostics, total

    @science_state_transaction
    def kernel(self, atmlst=None) -> np.ndarray:
        """Evaluate d(E_base + E_corr)/dR without a coordinate-wise density response."""
        self._reset_results()
        self._validate_driver_binding()
        atom_indices = _validate_atom_indices(self.mol, atmlst)
        if not self.retain_details:
            (
                self.descriptor_diagnostics,
                self._response_diagnostics,
                self.de,
            ) = self._compact_kernel(atom_indices)
            return self.de
        descriptor_diagnostics, sensitivity, adjoint = self.base._zvector_inputs(
            self.response_options,
            atom_indices=atom_indices,
        )
        self.base._assert_science_state("native RHF gradient evaluation")
        reference_gradient = np.asarray(
            self.base.reference.nuc_grad_method().kernel(
                atmlst=None if atom_indices is None else list(atom_indices)
            )
        )
        self.base._validate_science_state("native RHF gradient evaluation")
        self.base._assert_science_state("explicit descriptor gradient evaluation")
        dq_dR_explicit = self.base.dq_dR_explicit(atom_indices=atom_indices)
        self.base._assert_science_state("explicit descriptor gradient evaluation")
        correction_gradient_explicit = np.einsum(
            "bxap,ap->bx",
            dq_dR_explicit,
            sensitivity,
        )
        correction_gradient_metric = np.asarray(
            adjoint.correction_gradient_metric
        )
        correction_gradient_adjoint_nuclear = np.asarray(
            adjoint.correction_gradient_adjoint_nuclear
        )
        correction_gradient_adjoint_metric = np.asarray(
            adjoint.correction_gradient_adjoint_metric
        )
        correction_gradient_occupied_virtual = np.asarray(
            adjoint.correction_gradient_occupied_virtual
        )
        correction_gradient_response = np.asarray(
            adjoint.correction_gradient_response
        )
        correction_gradient = (
            correction_gradient_explicit + correction_gradient_response
        )
        de_full = reference_gradient + correction_gradient
        expected_shape = (len(adjoint.atom_indices), 3)
        result_fields = {
            "reference gradient": reference_gradient,
            "explicit correction gradient": correction_gradient_explicit,
            "metric correction gradient": correction_gradient_metric,
            "adjoint nuclear correction gradient": (
                correction_gradient_adjoint_nuclear
            ),
            "adjoint metric correction gradient": (
                correction_gradient_adjoint_metric
            ),
            "occupied-virtual correction gradient": (
                correction_gradient_occupied_virtual
            ),
            "response correction gradient": correction_gradient_response,
            "complete correction gradient": correction_gradient,
            "RHF DeePHF Z-vector gradient": de_full,
        }
        for name, value in result_fields.items():
            if value.shape != expected_shape:
                raise RHFAdjointError(
                    f"the {name} has shape {value.shape}; expected {expected_shape}"
                )
            if value.dtype != np.dtype(np.float64) or np.iscomplexobj(value):
                raise RHFAdjointError(
                    f"the {name} must be a real numpy.float64 array"
                )
            if not np.isfinite(value).all():
                raise RHFAdjointError(f"the {name} must be finite")
        if not np.allclose(
            correction_gradient_occupied_virtual,
            correction_gradient_adjoint_nuclear
            + correction_gradient_adjoint_metric,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise RHFAdjointError(
                "the RHF occupied-virtual adjoint partitions are inconsistent"
            )
        if not np.allclose(
            correction_gradient_response,
            correction_gradient_metric
            + correction_gradient_occupied_virtual,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise RHFAdjointError(
                "the RHF scalar-adjoint response partitions are inconsistent"
            )
        self.base._assert_science_state("Z-vector gradient assembly")
        self.adjoint_result = adjoint
        self.descriptor_diagnostics = descriptor_diagnostics
        self.reference_gradient = reference_gradient
        self.dq_dR_explicit = dq_dR_explicit
        self.correction_gradient_explicit = correction_gradient_explicit
        self.correction_gradient_metric = correction_gradient_metric
        self.correction_gradient_adjoint_nuclear = (
            correction_gradient_adjoint_nuclear
        )
        self.correction_gradient_adjoint_metric = (
            correction_gradient_adjoint_metric
        )
        self.correction_gradient_occupied_virtual = (
            correction_gradient_occupied_virtual
        )
        self.correction_gradient_response = correction_gradient_response
        self.correction_gradient = correction_gradient
        self.de_full = de_full
        self.de = de_full
        return self.de

    def run(self, atmlst=None):
        """Evaluate the gradient and return this populated driver."""
        self.kernel(atmlst=atmlst)
        return self

    def forces(self, atmlst=None) -> np.ndarray:
        """Evaluate nuclear forces as minus the energy gradient."""
        return -self.kernel(atmlst=atmlst)

    def as_scanner(self, **scanner_options):
        """Build a strict fresh-reference RHF DeePHF gradient scanner."""
        from .scanner import RHFDeePHFGradientScanner

        return RHFDeePHFGradientScanner(self, **scanner_options)
