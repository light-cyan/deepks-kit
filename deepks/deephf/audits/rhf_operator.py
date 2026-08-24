"""Bounded dense response-operator audits."""

from __future__ import annotations

from ..capabilities import DeePHFCapabilityError
from ..pyscf_rhf_reference import RHFResponseError
import numpy as np


def _response_operator_matrix_and_diagnostics(
    self,
    coefficient: np.ndarray,
    energy: np.ndarray,
    occupation: np.ndarray,
    occupied: np.ndarray,
    virtual: np.ndarray,
) -> tuple[np.ndarray, int, float, float, float, float]:
    """Build and audit a small unshifted occupied-virtual operator."""
    nocc = int(np.count_nonzero(occupied))
    nvir = int(np.count_nonzero(virtual))
    dimension = nocc * nvir
    if dimension > self.operator_dimension_limit:
        raise DeePHFCapabilityError(
            "the explicit RHF operator validation dimension exceeds its "
            f"debug limit: {dimension} > {self.operator_dimension_limit}"
        )
    matrix = np.empty((dimension, dimension), dtype=np.float64)
    batch_size = min(64, dimension)
    for start in range(0, dimension, batch_size):
        stop = min(start + batch_size, dimension)
        flat_roots = np.zeros((stop - start, dimension), dtype=np.float64)
        flat_roots[np.arange(stop - start), np.arange(start, stop)] = 1.0
        roots = flat_roots.reshape(-1, nvir, nocc)
        images = self._apply_occupied_virtual_operator(
            roots,
            coefficient,
            energy,
            occupation,
            occupied,
            virtual,
        )
        matrix[:, start:stop] = images.reshape(stop - start, dimension).T
    if not np.isfinite(matrix).all():
        raise RHFResponseError(
            "the RHF occupied-virtual response operator is nonfinite"
        )
    symmetry_residual = float(
        np.max(np.abs(matrix - matrix.T), initial=0.0)
    )
    if symmetry_residual > self.operator_symmetry_tolerance:
        raise RHFResponseError(
            "the RHF occupied-virtual response operator violates symmetry: "
            f"{symmetry_residual:.3e} > {self.operator_symmetry_tolerance:.3e}"
        )
    try:
        eigenvalues = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
    except np.linalg.LinAlgError as error:
        raise RHFResponseError(
            f"the RHF response-operator eigensolve failed: {error}"
        ) from error
    minimum_eigenvalue = float(eigenvalues[0])
    maximum_eigenvalue = float(eigenvalues[-1])
    if minimum_eigenvalue <= self.operator_stability_tolerance:
        raise DeePHFCapabilityError(
            "the RHF occupied-virtual response operator is unstable or singular: "
            f"minimum eigenvalue {minimum_eigenvalue:.3e} <= "
            f"{self.operator_stability_tolerance:.3e}"
        )
    condition_number = maximum_eigenvalue / minimum_eigenvalue
    if (
        not np.isfinite(condition_number)
        or condition_number > self.operator_condition_tolerance
    ):
        raise DeePHFCapabilityError(
            "the RHF occupied-virtual response operator is ill conditioned: "
            f"{condition_number:.3e} > {self.operator_condition_tolerance:.3e}"
        )
    return (
        matrix,
        dimension,
        minimum_eigenvalue,
        maximum_eigenvalue,
        float(condition_number),
        symmetry_residual,
    )


def validate_response_operator_exact(self) -> tuple[int, float, float, float, float]:
    """Run an explicit dense stability audit for a bounded debug problem."""
    coefficient, energy, occupation, occupied, virtual, _gap = self._state()
    return self._response_operator_matrix_and_diagnostics(
        coefficient,
        energy,
        occupation,
        occupied,
        virtual,
    )[1:]


__all__ = ['_response_operator_matrix_and_diagnostics', 'validate_response_operator_exact']
