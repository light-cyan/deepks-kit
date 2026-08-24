"""Bounded dense response-operator audits."""

from __future__ import annotations

from ..capabilities import DeePHFCapabilityError
from ..pyscf_dft_provenance import RKSResponseError
from ..pyscf_dft_provenance import _validated_float64_array
import numpy as np


def _response_operator_matrix_and_diagnostics(
    self,
    coefficient: np.ndarray,
    energy: np.ndarray,
    occupation: np.ndarray,
    occupied: np.ndarray,
    virtual: np.ndarray,
) -> tuple[np.ndarray, int, float, float, float, float, float]:
    nocc = int(np.count_nonzero(occupied))
    nvir = int(np.count_nonzero(virtual))
    dimension = nocc * nvir
    if dimension > self.operator_dimension_limit:
        raise DeePHFCapabilityError(
            "RKS occupied-virtual response dimension exceeds the explicit "
            f"condition-audit limit: {dimension} > {self.operator_dimension_limit}"
        )
    identity = np.eye(dimension, dtype=np.float64)
    matrix = np.empty((dimension, dimension), dtype=np.float64)
    reconstruction_residual = 0.0
    reference_response = self.reference.gen_response(
        coefficient,
        occupation,
        hermi=1,
    )
    batch_size = min(32, dimension)
    for start in range(0, dimension, batch_size):
        stop = min(start + batch_size, dimension)
        roots = identity[start:stop].reshape(-1, nvir, nocc)
        images = self._apply_occupied_virtual_operator(
            roots,
            coefficient,
            energy,
            occupation,
            occupied,
            virtual,
        )
        matrix[:, start:stop] = images.reshape(stop - start, dimension).T
        full_roots = np.zeros(
            (stop - start, coefficient.shape[1], nocc),
            dtype=np.float64,
        )
        full_roots[:, virtual] = roots
        density_roots = self._density_from_mo_response(
            full_roots,
            coefficient,
            occupation,
            occupied,
        )
        independent = self._induced_potential(density_roots)
        try:
            pyscf_response = np.asarray(reference_response(density_roots))
        except Exception as error:
            raise RKSResponseError(
                f"PySCF RKS induced-response reconstruction failed: {error}"
            ) from error
        pyscf_response = _validated_float64_array(
            pyscf_response,
            density_roots.shape,
            "PySCF induced RKS response",
        )
        reconstruction_residual = max(
            reconstruction_residual,
            float(
                np.max(
                    np.abs(independent - pyscf_response),
                    initial=0.0,
                )
            ),
        )
    if not np.isfinite(matrix).all():
        raise RKSResponseError("the RKS occupied-virtual response operator is nonfinite")
    if reconstruction_residual > self.invariant_tolerance:
        raise RKSResponseError(
            "the independent direct-J plus dense-LDA response does not match "
            f"PySCF: residual {reconstruction_residual:.3e}"
        )
    symmetry_residual = float(
        np.max(np.abs(matrix - matrix.T), initial=0.0)
    )
    if symmetry_residual > self.operator_symmetry_tolerance:
        raise RKSResponseError(
            "the RKS occupied-virtual response operator violates symmetry: "
            f"{symmetry_residual:.3e} > {self.operator_symmetry_tolerance:.3e}"
        )
    try:
        eigenvalues = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
    except np.linalg.LinAlgError as error:
        raise RKSResponseError(
            f"the RKS response-operator eigensolve failed: {error}"
        ) from error
    minimum_eigenvalue = float(eigenvalues[0])
    maximum_eigenvalue = float(eigenvalues[-1])
    if minimum_eigenvalue <= self.operator_stability_tolerance:
        raise DeePHFCapabilityError(
            "the RKS occupied-virtual response operator is unstable or singular: "
            f"minimum eigenvalue {minimum_eigenvalue:.3e} <= "
            f"{self.operator_stability_tolerance:.3e}"
        )
    condition_number = maximum_eigenvalue / minimum_eigenvalue
    if (
        not np.isfinite(condition_number)
        or condition_number > self.operator_condition_tolerance
    ):
        raise DeePHFCapabilityError(
            "the RKS occupied-virtual response operator is ill conditioned: "
            f"{condition_number:.3e} > {self.operator_condition_tolerance:.3e}"
        )
    return (
        matrix,
        dimension,
        minimum_eigenvalue,
        maximum_eigenvalue,
        float(condition_number),
        symmetry_residual,
        reconstruction_residual,
    )


def validate_response_operator_exact(
    self,
) -> tuple[int, float, float, float, float, float]:
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
