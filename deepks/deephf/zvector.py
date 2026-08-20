"""Strict scalar-adjoint nuclear gradients for RHF DeePHF."""

import numpy as np

from .gradient import _validate_atom_indices
from .pyscf_rhf import RHFAdjointError


class RHFDeePHFZVectorGradients:
    """Evaluate the correction response through one RHF scalar adjoint."""

    def __init__(self, method, adjoint_options=None):
        self.base = method
        self.mol = method.mol
        self.backend = "zvector"
        self.response_options = dict(adjoint_options or {})
        self._reset_results()

    def _reset_results(self):
        self.adjoint_result = None
        self.descriptor_diagnostics = None
        self.reference_gradient = None
        self.dq_dR_explicit = None
        self.correction_gradient_explicit = None
        self.correction_gradient_metric = None
        self.correction_gradient_adjoint_nuclear = None
        self.correction_gradient_adjoint_metric = None
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
        """Return the scalar-adjoint diagnostics under the common driver name."""
        return self.adjoint_diagnostics

    def kernel(self, atmlst=None) -> np.ndarray:
        """Evaluate d(E_base + E_corr)/dR without a coordinate-wise density response."""
        self._reset_results()
        atom_indices = _validate_atom_indices(self.mol, atmlst)
        descriptor_diagnostics, sensitivity, adjoint = self.base._zvector_inputs(
            self.response_options
        )
        reference_gradient = np.asarray(
            self.base.reference.nuc_grad_method().kernel()
        )
        dq_dR_explicit = self.base.dq_dR_explicit()
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
        expected_shape = (self.mol.natm, 3)
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
        selected_gradient = (
            de_full
            if atom_indices is None
            else de_full[list(atom_indices)]
        )
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
        self.de = selected_gradient
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
