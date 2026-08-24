"""UKS validation audit separated from production composition."""

from __future__ import annotations

from numbers import Real
from ..pyscf_uhf_reference import UHFAdjointError
from ..pyscf_uks_reference import UKSAdjoint
from ..pyscf_uks_reference import UKSAdjointDiagnostics
from ..pyscf_uks_reference import UKSAdjointError
from ..pyscf_dft_provenance import _grid_provenance
from ..pyscf_uks_reference import _uks_functional_provenance
import numpy as np
from ..pyscf_uks_reference import uks_adjoint_integrity_fingerprint
from ..pyscf_uks_reference import validate_uks_reference
from ..pyscf_uks_response import (
    _require_wrapper_close,
)


def audit_adjoint(self, adjoint: UKSAdjoint, expected_objective_ao_potential: np.ndarray) -> None:
    """Rebuild one UKS adjoint without another transpose solve."""
    validate_uks_reference(self.reference)
    if type(adjoint) is not UKSAdjoint or type(adjoint.diagnostics) is not UKSAdjointDiagnostics:
        raise UKSAdjointError("the supplied UKS adjoint has an invalid type")
    if adjoint.integrity_fingerprint != uks_adjoint_integrity_fingerprint(adjoint):
        raise UKSAdjointError("the supplied UKS adjoint failed its integrity check")
    if (
        adjoint.functional != _uks_functional_provenance(self.reference)
        or adjoint.grid != _grid_provenance(self.reference)
    ):
        raise UKSAdjointError("the supplied UKS adjoint provenance is inconsistent")
    if adjoint.diagnostics.functional != adjoint.functional or adjoint.diagnostics.grid != adjoint.grid:
        raise UKSAdjointError("the supplied UKS adjoint diagnostics provenance is inconsistent")
    try:
        self._core.audit_adjoint(adjoint.core, expected_objective_ao_potential)
    except UHFAdjointError as error:
        raise UKSAdjointError(f"UKS adjoint audit failed: {error}") from error
    fixed_spin, coordinate_spin, weight_spin, partition_residual = self._nuclear_partitions(adjoint.core)
    expected_shape = (2, len(adjoint.core.atom_indices), 3)
    spin_arrays = {
        "fixed-grid adjoint gradient": (adjoint.correction_gradient_adjoint_fixed_grid_spin, fixed_spin),
        "grid-coordinate adjoint gradient": (adjoint.correction_gradient_adjoint_grid_coordinate_spin, coordinate_spin),
        "grid-weight adjoint gradient": (adjoint.correction_gradient_adjoint_grid_weight_spin, weight_spin),
    }
    for name, (actual, expected) in spin_arrays.items():
        if (
            type(actual) is not np.ndarray
            or actual.shape != expected_shape
            or actual.dtype != np.dtype(np.float64)
            or actual.flags.writeable
            or not np.isfinite(actual).all()
        ):
            raise UKSAdjointError(f"the supplied UKS {name} is invalid")
        _require_wrapper_close(actual, expected, name, UKSAdjointError)
    for name, spin_name in (
        ("correction_gradient_adjoint_fixed_grid", "correction_gradient_adjoint_fixed_grid_spin"),
        ("correction_gradient_adjoint_grid_coordinate", "correction_gradient_adjoint_grid_coordinate_spin"),
        ("correction_gradient_adjoint_grid_weight", "correction_gradient_adjoint_grid_weight_spin"),
    ):
        _require_wrapper_close(getattr(adjoint, name), getattr(adjoint, spin_name).sum(axis=0), name, UKSAdjointError)
    measured = {
        "nuclear_partition_residual": partition_residual,
    }
    for name, expected in measured.items():
        stored = getattr(adjoint.diagnostics, name)
        if (
            isinstance(stored, (bool, np.bool_))
            or not isinstance(stored, Real)
            or not np.isfinite(stored)
            or not np.isclose(stored, expected, rtol=1.0e-10, atol=1.0e-12)
        ):
            raise UKSAdjointError(f"the supplied UKS {name} diagnostic is inconsistent")
    if max(measured.values()) > adjoint.diagnostics.invariant_tolerance:
        raise UKSAdjointError("the supplied UKS adjoint invariant exceeds tolerance")


__all__ = ['audit_adjoint']
