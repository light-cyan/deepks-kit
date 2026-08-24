"""UKS validation audit separated from production composition."""

from __future__ import annotations

from numbers import Real
from ..pyscf_uhf_reference import UHFResponseError
from ..pyscf_uks_reference import UKSResponse
from ..pyscf_uks_reference import UKSResponseDiagnostics
from ..pyscf_uks_reference import UKSResponseError
from ..pyscf_dft_provenance import _grid_provenance
from ..pyscf_uks_reference import _uks_functional_provenance
import numpy as np
from ..pyscf_uks_reference import uks_response_integrity_fingerprint
from ..pyscf_uks_reference import validate_uks_reference
from ..pyscf_uks_response import (
    _require_wrapper_close,
)


def audit_response_equations(self, response: UKSResponse) -> None:
    """Rebuild one supplied UKS response without another CPHF solve."""
    validate_uks_reference(self.reference)
    if type(response) is not UKSResponse or type(response.diagnostics) is not UKSResponseDiagnostics:
        raise UKSResponseError("the supplied UKS response has an invalid type")
    if response.integrity_fingerprint != uks_response_integrity_fingerprint(response):
        raise UKSResponseError("the supplied UKS response failed its integrity check")
    if (
        response.functional != _uks_functional_provenance(self.reference)
        or response.grid != _grid_provenance(self.reference)
    ):
        raise UKSResponseError("the supplied UKS response provenance is inconsistent")
    if response.diagnostics.functional != response.functional or response.diagnostics.grid != response.grid:
        raise UKSResponseError("the supplied UKS response diagnostics provenance is inconsistent")
    try:
        self._core.audit_response_equations(response.core)
    except UHFResponseError as error:
        raise UKSResponseError(f"UKS response audit failed: {error}") from error
    full, fixed, coordinate, weight = self._components(self._core)
    expected_shape = (2, len(response.core.atom_indices), 3, self.reference.mol.nao, self.reference.mol.nao)
    arrays = {
        "fixed-grid Hamiltonian derivative": (response.hamiltonian_derivative_fixed_grid_spin, fixed),
        "grid-coordinate XC derivative": (response.xc_hamiltonian_derivative_grid_coordinate_spin, coordinate),
        "grid-weight XC derivative": (response.xc_hamiltonian_derivative_grid_weight_spin, weight),
    }
    for name, (actual, expected) in arrays.items():
        if (
            type(actual) is not np.ndarray
            or actual.shape != expected_shape
            or actual.dtype != np.dtype(np.float64)
            or actual.flags.writeable
            or not np.isfinite(actual).all()
        ):
            raise UKSResponseError(f"the supplied UKS {name} is invalid")
        _require_wrapper_close(actual, expected, name, UKSResponseError)
    reconstruction = float(np.max(np.abs(full - fixed - coordinate - weight), initial=0.0))
    measured = {
        "hamiltonian_reconstruction_residual": reconstruction,
    }
    for name, expected in measured.items():
        stored = getattr(response.diagnostics, name)
        if (
            isinstance(stored, (bool, np.bool_))
            or not isinstance(stored, Real)
            or not np.isfinite(stored)
            or not np.isclose(stored, expected, rtol=1.0e-10, atol=1.0e-12)
        ):
            raise UKSResponseError(f"the supplied UKS {name} diagnostic is inconsistent")
    if max(measured.values()) > response.diagnostics.invariant_tolerance:
        raise UKSResponseError("the supplied UKS response invariant exceeds tolerance")


__all__ = ['audit_response_equations']
