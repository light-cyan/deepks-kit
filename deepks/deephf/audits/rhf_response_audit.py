"""Bounded dense audits separated from production solver assembly."""

from __future__ import annotations

from ..pyscf_rhf_reference import RHFResponseError
import numpy as np
import pyscf
from ..pyscf_rhf_reference import validate_reference


def _rebuild_response_equations(self, response: RHFResponse):
    """Rebuild derivative inputs, equations, and invariants for a supplied response."""
    validate_reference(self.reference)
    if response.diagnostics.operator_is_self_adjoint is not True:
        raise RHFResponseError("the supplied RHF response operator contract is invalid")
    if response.diagnostics.pyscf_version != pyscf.__version__:
        raise RHFResponseError(
            "the supplied RHF response PySCF version does not match the runtime"
        )
    (
        coefficient,
        energy,
        occupation,
        occupied,
        virtual,
        minimum_gap,
    ) = self._state()
    response_dimension = int(np.count_nonzero(occupied)) * int(
        np.count_nonzero(virtual)
    )
    atom_indices = self._response_atom_indices(response.atom_indices)
    expected_overlap_derivative = self._overlap_derivative(atom_indices)
    expected_hamiltonian_derivative = self._hamiltonian_derivative(
        coefficient,
        occupation,
        atom_indices,
    )
    derivative_fields = (
        (
            response.overlap_derivative,
            expected_overlap_derivative,
            "overlap derivative",
        ),
        (
            response.hamiltonian_derivative,
            expected_hamiltonian_derivative,
            "Hamiltonian derivative",
        ),
    )
    for stored, expected, name in derivative_fields:
        if not np.allclose(stored, expected, rtol=0.0, atol=1.0e-12):
            raise RHFResponseError(
                f"the supplied RHF response {name} does not match the reference"
            )
    occupied_coefficients = coefficient[:, occupied]
    hamiltonian_mo = np.einsum(
        "mp,...mn,ni->...pi",
        coefficient,
        expected_hamiltonian_derivative,
        occupied_coefficients,
    )
    overlap_mo = np.einsum(
        "mp,...mn,ni->...pi",
        coefficient,
        expected_overlap_derivative,
        occupied_coefficients,
    )
    residual = self._orbital_residual(
        response.mo_response,
        hamiltonian_mo,
        overlap_mo,
        coefficient,
        energy,
        occupation,
        occupied,
        virtual,
    )
    if not np.allclose(
        response.orbital_response_residual,
        residual,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RHFResponseError(
            "the supplied RHF response orbital residual is not independently reproducible"
        )
    return (
        minimum_gap, response_dimension, occupied, residual,
        expected_overlap_derivative, overlap_mo,
    )


def _audit_response_invariants(self, response, state):
    (
        minimum_gap, response_dimension, occupied, residual,
        expected_overlap_derivative, overlap_mo,
    ) = state
    overlap = np.asarray(self.reference.get_ovlp())
    density_ground = np.asarray(self.reference.make_rdm1())
    density_response = response.density_response
    overlap_occupied = overlap_mo[..., occupied, :]
    metric_residual = float(
        np.max(
            np.abs(
                response.mo_response[..., occupied, :]
                + response.mo_response[..., occupied, :].swapaxes(-1, -2)
                + overlap_occupied
            ),
            initial=0.0,
        )
    )
    idempotency = (
        np.einsum(
            "...ij,jk,kl->...il",
            density_response,
            overlap,
            density_ground,
        )
        + np.einsum(
            "ij,...jk,kl->...il",
            density_ground,
            expected_overlap_derivative,
            density_ground,
        )
        + np.einsum(
            "ij,jk,...kl->...il",
            density_ground,
            overlap,
            density_response,
        )
        - 2.0 * density_response
    )
    particle_number = (
        np.einsum("...ij,ji->...", density_response, overlap)
        + np.einsum(
            "ij,...ji->...",
            density_ground,
            expected_overlap_derivative,
        )
    )
    measured = {
        "minimum_orbital_gap": minimum_gap,
        "response_dimension": response_dimension,
        "maximum_residual": float(np.max(np.abs(residual), initial=0.0)),
        "residual_rms": float(np.sqrt(np.mean(np.square(residual)))),
        "metric_residual": metric_residual,
        "idempotency_residual": float(
            np.max(np.abs(idempotency), initial=0.0)
        ),
        "particle_number_residual": float(
            np.max(np.abs(particle_number), initial=0.0)
        ),
    }
    for name, value in measured.items():
        recorded = getattr(response.diagnostics, name)
        if isinstance(value, int):
            consistent = recorded == value
        else:
            consistent = np.isclose(
                recorded,
                value,
                rtol=1.0e-10,
                atol=1.0e-12,
            )
        if not consistent:
            raise RHFResponseError(
                f"the supplied RHF response {name} diagnostic is inconsistent"
            )
    if measured["maximum_residual"] > self.residual_tolerance:
        raise RHFResponseError(
            "the supplied RHF response residual exceeds its tolerance"
        )
    invariant_values = (
        measured["metric_residual"],
        measured["idempotency_residual"],
        measured["particle_number_residual"],
    )
    if max(invariant_values) > self.invariant_tolerance:
        raise RHFResponseError(
            "the supplied RHF response invariant exceeds its tolerance"
        )


def audit_response_equations(self, response: RHFResponse) -> None:
    """Rebuild derivative inputs, equations, and invariants for a supplied response."""
    state = _rebuild_response_equations(self, response)
    _audit_response_invariants(self, response, state)


__all__ = ['audit_response_equations']

"""Bounded dense response-operator audits."""

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


__all__ = [
    "_response_operator_matrix_and_diagnostics",
    "audit_response_equations",
    "validate_response_operator_exact",
]
