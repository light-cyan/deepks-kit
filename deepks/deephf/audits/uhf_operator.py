"""Bounded dense response-operator audits."""

from __future__ import annotations

from ..capabilities import DeePHFCapabilityError
from ..pyscf_uhf_reference import UHFResponseError
import numpy as np


def _response_operator_matrix_and_diagnostics(
    self,
    coefficient: np.ndarray,
    energy: np.ndarray,
    occupied: np.ndarray,
    virtual: np.ndarray,
) -> tuple[np.ndarray, int, int, int, float, float, float, float]:
    dimensions = self._dimensions(occupied, virtual)
    alpha_dimension, beta_dimension = dimensions[-2:]
    dimension = alpha_dimension + beta_dimension
    if dimension > self.operator_dimension_limit:
        raise DeePHFCapabilityError(
            "UHF coupled occupied-virtual response dimension exceeds the "
            f"condition-audit limit: {dimension} > {self.operator_dimension_limit}"
        )
    identity = np.eye(dimension, dtype=np.float64)
    matrix = np.empty((dimension, dimension), dtype=np.float64)
    batch_size = min(64, dimension)
    for start in range(0, dimension, batch_size):
        stop = min(start + batch_size, dimension)
        images = self._apply_occupied_virtual_operator(
            identity[start:stop],
            coefficient,
            energy,
            occupied,
            virtual,
        )
        matrix[:, start:stop] = images.T
    if not np.isfinite(matrix).all():
        raise UHFResponseError(
            "the coupled UHF occupied-virtual response operator is nonfinite"
        )
    symmetry_residual = float(
        np.max(np.abs(matrix - matrix.T), initial=0.0)
    )
    if symmetry_residual > self.operator_symmetry_tolerance:
        raise UHFResponseError(
            "the coupled UHF occupied-virtual response operator violates symmetry: "
            f"{symmetry_residual:.3e} > {self.operator_symmetry_tolerance:.3e}"
        )
    try:
        eigenvalues = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
    except np.linalg.LinAlgError as error:
        raise UHFResponseError(
            f"the coupled UHF response-operator eigensolve failed: {error}"
        ) from error
    minimum_eigenvalue = float(eigenvalues[0])
    maximum_eigenvalue = float(eigenvalues[-1])
    if minimum_eigenvalue <= self.operator_stability_tolerance:
        raise DeePHFCapabilityError(
            "the coupled UHF occupied-virtual response operator is unstable or "
            f"singular: minimum eigenvalue {minimum_eigenvalue:.3e} <= "
            f"{self.operator_stability_tolerance:.3e}"
        )
    condition_number = maximum_eigenvalue / minimum_eigenvalue
    if (
        not np.isfinite(condition_number)
        or condition_number > self.operator_condition_tolerance
    ):
        raise DeePHFCapabilityError(
            "the coupled UHF occupied-virtual response operator is ill conditioned: "
            f"{condition_number:.3e} > {self.operator_condition_tolerance:.3e}"
        )
    return (
        matrix,
        dimension,
        alpha_dimension,
        beta_dimension,
        minimum_eigenvalue,
        maximum_eigenvalue,
        float(condition_number),
        symmetry_residual,
    )


def validate_response_operator_exact(
    self,
) -> tuple[int, int, int, float, float, float, float]:
    """Run an explicit dense stability audit for a bounded debug problem."""
    coefficient, energy, _occupation, occupied, virtual, _gaps = self._state()
    return self._response_operator_matrix_and_diagnostics(
        coefficient,
        energy,
        occupied,
        virtual,
    )[1:]


__all__ = ['_response_operator_matrix_and_diagnostics', 'validate_response_operator_exact']
