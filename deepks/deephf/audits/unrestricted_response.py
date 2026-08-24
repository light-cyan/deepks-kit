"""Bounded dense audits separated from production solver assembly."""

from __future__ import annotations

from numbers import Real
from ..unrestricted_reference import UHFResponse
from ..unrestricted_reference import UHFResponseDiagnostics
from ..unrestricted_reference import UHFResponseError
from ..unrestricted_reference import _validated_response_array
from dataclasses import fields
import numpy as np
import pyscf
from ..unrestricted_reference import uhf_response_integrity_fingerprint


def _validate_response_arrays(
    self,
    response: UHFResponse,
    occupied: np.ndarray,
    virtual: np.ndarray,
) -> None:
    atom_indices = self._response_atom_indices(response.atom_indices)
    if atom_indices != response.atom_indices:
        raise UHFResponseError("the supplied UHF response atom selection is invalid")
    coordinate_shape = (len(atom_indices), 3)
    nao = self.molecule.nao
    alpha_nocc = int(np.count_nonzero(occupied[0]))
    beta_nocc = int(np.count_nonzero(occupied[1]))
    alpha_nvir = int(np.count_nonzero(virtual[0]))
    beta_nvir = int(np.count_nonzero(virtual[1]))
    alpha_response_shape = (*coordinate_shape, nao, alpha_nocc)
    beta_response_shape = (*coordinate_shape, nao, beta_nocc)
    density_shape = (*coordinate_shape, nao, nao)
    expected_shapes = {
        "alpha_mo_response": alpha_response_shape,
        "beta_mo_response": beta_response_shape,
        "alpha_mo_response_occupied_virtual": alpha_response_shape,
        "beta_mo_response_occupied_virtual": beta_response_shape,
        "alpha_mo_response_metric": alpha_response_shape,
        "beta_mo_response_metric": beta_response_shape,
        "alpha_coefficient_response": alpha_response_shape,
        "beta_coefficient_response": beta_response_shape,
        "alpha_coefficient_response_occupied_virtual": alpha_response_shape,
        "beta_coefficient_response_occupied_virtual": beta_response_shape,
        "alpha_coefficient_response_metric": alpha_response_shape,
        "beta_coefficient_response_metric": beta_response_shape,
        "alpha_density_response": density_shape,
        "beta_density_response": density_shape,
        "total_density_response": density_shape,
        "alpha_density_response_occupied_virtual": density_shape,
        "beta_density_response_occupied_virtual": density_shape,
        "total_density_response_occupied_virtual": density_shape,
        "alpha_density_response_metric": density_shape,
        "beta_density_response_metric": density_shape,
        "total_density_response_metric": density_shape,
        "overlap_derivative": density_shape,
        "alpha_hamiltonian_derivative": density_shape,
        "beta_hamiltonian_derivative": density_shape,
        "alpha_orbital_response_residual": (
            *coordinate_shape,
            alpha_nvir,
            alpha_nocc,
        ),
        "beta_orbital_response_residual": (
            *coordinate_shape,
            beta_nvir,
            beta_nocc,
        ),
    }
    for name, expected_shape in expected_shapes.items():
        _validated_response_array(
            getattr(response, name),
            expected_shape,
            name.replace("_", " "),
        )
    if type(response.reference_identity) is not int:
        raise UHFResponseError(
            "the supplied UHF response reference identity must be an integer"
        )
    for name in ("state_fingerprint", "integrity_fingerprint"):
        value = getattr(response, name)
        if type(value) is not str or not value:
            raise UHFResponseError(
                f"the supplied UHF response {name.replace('_', ' ')} is invalid"
            )
    return atom_indices, alpha_nocc, beta_nocc, alpha_nvir, beta_nvir


def _validate_response_diagnostics(self, response, state):
    atom_indices, alpha_nocc, beta_nocc, alpha_nvir, beta_nvir = state
    diagnostics = response.diagnostics
    if type(diagnostics.pyscf_version) is not str:
        raise UHFResponseError(
            "the supplied UHF response PySCF version is invalid"
        )
    integer_fields = {
        "max_cycle",
        "max_refinement_cycles",
        "response_dimension",
        "alpha_response_dimension",
        "beta_response_dimension",
        "operator_dimension_limit",
        "refinement_cycles",
    }
    for diagnostic_field in fields(diagnostics):
        name = diagnostic_field.name
        if name in {"pyscf_version", "residual_history"}:
            continue
        value = getattr(diagnostics, name)
        if name in {
            "alpha_translation_residual",
            "beta_translation_residual",
            "translation_residual",
        } and len(atom_indices) != self.molecule.natm:
            if value is not None:
                raise UHFResponseError(
                    "selected UHF responses cannot publish translation residuals"
                )
            continue
        if name == "operator_is_self_adjoint":
            if value is not True:
                raise UHFResponseError(
                    "the supplied UHF operator contract is invalid"
                )
        elif name in integer_fields:
            if type(value) is not int:
                raise UHFResponseError(
                    f"the supplied UHF response {name.replace('_', ' ')} must be an integer"
                )
        elif (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, Real)
            or not np.isfinite(float(value))
        ):
            raise UHFResponseError(
                f"the supplied UHF response {name.replace('_', ' ')} must be finite and real"
            )
    history = diagnostics.residual_history
    if type(history) is not tuple or not history:
        raise UHFResponseError(
            "the supplied UHF response residual history must be a nonempty tuple"
        )
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
        or not np.isfinite(float(value))
        or float(value) < 0
        for value in history
    ):
        raise UHFResponseError(
            "the supplied UHF response residual history is invalid"
        )
    expected_controls = {
        "cphf_tolerance": self.cphf_tolerance,
        "residual_tolerance": self.residual_tolerance,
        "invariant_tolerance": self.invariant_tolerance,
        "orbital_gap_tolerance": self.orbital_gap_tolerance,
        "max_cycle": self.max_cycle,
        "max_refinement_cycles": self.max_refinement_cycles,
        "level_shift": self.level_shift,
    }
    for name, expected in expected_controls.items():
        if getattr(diagnostics, name) != expected:
            raise UHFResponseError(
                f"the supplied UHF response {name.replace('_', ' ')} does not match the adapter"
            )
    if diagnostics.refinement_cycles != len(history) - 1:
        raise UHFResponseError(
            "the supplied UHF response refinement history is inconsistent"
        )
    if not 0 <= diagnostics.refinement_cycles <= self.max_refinement_cycles:
        raise UHFResponseError(
            "the supplied UHF response refinement cycle count is invalid"
        )
    if diagnostics.response_dimension != (
        diagnostics.alpha_response_dimension
        + diagnostics.beta_response_dimension
    ):
        raise UHFResponseError(
            "the supplied UHF response dimensions are inconsistent"
        )
    expected_dimensions = (
        alpha_nocc * alpha_nvir,
        beta_nocc * beta_nvir,
    )
    if (
        diagnostics.alpha_response_dimension,
        diagnostics.beta_response_dimension,
    ) != expected_dimensions:
        raise UHFResponseError(
            "the supplied UHF response spin dimensions are inconsistent"
        )
    nonnegative_fields = {
        "maximum_residual",
        "alpha_maximum_residual",
        "beta_maximum_residual",
        "residual_rms",
        "alpha_metric_residual",
        "beta_metric_residual",
        "alpha_idempotency_residual",
        "beta_idempotency_residual",
        "alpha_particle_number_residual",
        "beta_particle_number_residual",
        "alpha_translation_residual",
        "beta_translation_residual",
        "translation_residual",
    }
    if any(
        float(getattr(diagnostics, name)) < 0
        for name in nonnegative_fields
        if getattr(diagnostics, name) is not None
    ):
        raise UHFResponseError(
            "the supplied UHF response contains a negative residual diagnostic"
        )
    if diagnostics.minimum_alpha_orbital_gap <= self.orbital_gap_tolerance:
        raise UHFResponseError(
            "the supplied UHF response alpha gap is outside the adapter domain"
        )
    if diagnostics.minimum_beta_orbital_gap <= self.orbital_gap_tolerance:
        raise UHFResponseError(
            "the supplied UHF response beta gap is outside the adapter domain"
        )
    expected_maximum = max(
        diagnostics.alpha_maximum_residual,
        diagnostics.beta_maximum_residual,
    )
    if not np.isclose(
        diagnostics.maximum_residual,
        expected_maximum,
        rtol=1.0e-12,
        atol=1.0e-15,
    ):
        raise UHFResponseError(
            "the supplied UHF response spin residual diagnostics are inconsistent"
        )
    if not np.isclose(
        history[-1],
        diagnostics.maximum_residual,
        rtol=1.0e-12,
        atol=1.0e-15,
    ):
        raise UHFResponseError(
            "the supplied UHF response final residual history is inconsistent"
        )


def _validate_supplied_structure(
    self,
    response: UHFResponse,
    occupied: np.ndarray,
    virtual: np.ndarray,
) -> None:
    state = _validate_response_arrays(self, response, occupied, virtual)
    _validate_response_diagnostics(self, response, state)


def _prepare_response_audit(self, response):
    """Rebuild coupled equations and invariants for a supplied response."""
    self._validate_reference(self.reference)
    if type(response) is not UHFResponse:
        raise UHFResponseError("the supplied UHF response has an invalid type")
    if type(response.diagnostics) is not UHFResponseDiagnostics:
        raise UHFResponseError(
            "the supplied UHF response diagnostics have an invalid type"
        )
    coefficient, energy, occupation, occupied, virtual, minimum_gaps = self._state()
    self._validate_supplied_structure(response, occupied, virtual)
    if response.reference_identity != id(self.reference):
        raise UHFResponseError("the supplied UHF response belongs to another reference")
    if response.state_fingerprint != self._reference_fingerprint(self.reference):
        raise UHFResponseError("the supplied UHF response does not match the current state")
    if response.integrity_fingerprint != uhf_response_integrity_fingerprint(response):
        raise UHFResponseError("the supplied UHF response failed its integrity check")
    if response.diagnostics.pyscf_version != pyscf.__version__:
        raise UHFResponseError(
            "the supplied UHF response PySCF version does not match the runtime"
        )
    *_, alpha_dimension, beta_dimension = self._dimensions(occupied, virtual)
    overlap = np.asarray(self.reference.get_ovlp())
    overlap_derivative = self._overlap_derivative(response.atom_indices)
    hamiltonian_derivative = self._hamiltonian_derivative(
        coefficient,
        occupation,
        response.atom_indices,
    )
    expected_derivatives = (
        (response.overlap_derivative, overlap_derivative, "overlap derivative"),
        (
            response.alpha_hamiltonian_derivative,
            hamiltonian_derivative[0],
            "alpha Hamiltonian derivative",
        ),
        (
            response.beta_hamiltonian_derivative,
            hamiltonian_derivative[1],
            "beta Hamiltonian derivative",
        ),
    )
    for stored, expected, name in expected_derivatives:
        if not np.allclose(stored, expected, rtol=0.0, atol=1.0e-12):
            raise UHFResponseError(
                f"the supplied UHF response {name} does not match the reference"
            )
    return (
        coefficient, energy, occupation, occupied, virtual, minimum_gaps,
        alpha_dimension, beta_dimension, overlap, overlap_derivative,
        hamiltonian_derivative,
    )


def _audit_spin_response_partitions(self, response, state):
    (
        coefficient, _energy, _occupation, occupied, virtual, _minimum_gaps,
        _alpha_dimension, _beta_dimension, overlap, overlap_derivative,
        _hamiltonian_derivative,
    ) = state
    responses = (response.alpha_mo_response, response.beta_mo_response)
    response_parts = (
        (
            response.alpha_mo_response_occupied_virtual,
            response.alpha_mo_response_metric,
        ),
        (
            response.beta_mo_response_occupied_virtual,
            response.beta_mo_response_metric,
        ),
    )
    density_responses = (
        response.alpha_density_response,
        response.beta_density_response,
    )
    density_parts = (
        (
            response.alpha_density_response_occupied_virtual,
            response.alpha_density_response_metric,
        ),
        (
            response.beta_density_response_occupied_virtual,
            response.beta_density_response_metric,
        ),
    )
    coefficient_responses = (
        response.alpha_coefficient_response,
        response.beta_coefficient_response,
    )
    coefficient_parts = (
        (
            response.alpha_coefficient_response_occupied_virtual,
            response.alpha_coefficient_response_metric,
        ),
        (
            response.beta_coefficient_response_occupied_virtual,
            response.beta_coefficient_response_metric,
        ),
    )
    metric_residuals = []
    invariant_values = []
    ground_density = np.asarray(self.reference.make_rdm1())
    for spin_index, spin_name in enumerate(("alpha", "beta")):
        expected_mo_occupied_virtual = np.zeros_like(responses[spin_index])
        expected_mo_occupied_virtual[..., virtual[spin_index], :] = responses[
            spin_index
        ][..., virtual[spin_index], :]
        expected_mo_metric = np.zeros_like(responses[spin_index])
        expected_mo_metric[..., occupied[spin_index], :] = responses[spin_index][
            ..., occupied[spin_index], :
        ]
        expected_coefficient = np.einsum(
            "mp,...pi->...mi",
            coefficient[spin_index],
            responses[spin_index],
        )
        expected_coefficient_occupied_virtual = np.einsum(
            "mp,...pi->...mi",
            coefficient[spin_index],
            expected_mo_occupied_virtual,
        )
        expected_coefficient_metric = np.einsum(
            "mp,...pi->...mi",
            coefficient[spin_index],
            expected_mo_metric,
        )
        expected_density = self._density_from_mo_response(
            responses[spin_index],
            coefficient[spin_index],
            occupied[spin_index],
        )
        expected_density_occupied_virtual = self._density_from_mo_response(
            expected_mo_occupied_virtual,
            coefficient[spin_index],
            occupied[spin_index],
        )
        expected_density_metric = self._density_from_mo_response(
            expected_mo_metric,
            coefficient[spin_index],
            occupied[spin_index],
        )
        comparisons = (
            (response_parts[spin_index][0], expected_mo_occupied_virtual, "MO OV"),
            (response_parts[spin_index][1], expected_mo_metric, "MO metric"),
            (coefficient_responses[spin_index], expected_coefficient, "coefficient"),
            (
                coefficient_parts[spin_index][0],
                expected_coefficient_occupied_virtual,
                "coefficient OV",
            ),
            (
                coefficient_parts[spin_index][1],
                expected_coefficient_metric,
                "coefficient metric",
            ),
            (density_responses[spin_index], expected_density, "density"),
            (
                density_parts[spin_index][0],
                expected_density_occupied_virtual,
                "density OV",
            ),
            (
                density_parts[spin_index][1],
                expected_density_metric,
                "density metric",
            ),
        )
        for stored, expected, name in comparisons:
            if not np.allclose(stored, expected, rtol=0.0, atol=1.0e-12):
                raise UHFResponseError(
                    f"the supplied UHF {spin_name} {name} response is inconsistent"
                )
        overlap_mo = np.einsum(
            "mp,...mn,ni->...pi",
            coefficient[spin_index],
            overlap_derivative,
            coefficient[spin_index][:, occupied[spin_index]],
        )
        overlap_occupied = overlap_mo[..., occupied[spin_index], :]
        metric_residuals.append(
            float(
                np.max(
                    np.abs(
                        responses[spin_index][..., occupied[spin_index], :]
                        + responses[spin_index][..., occupied[spin_index], :].swapaxes(-1, -2)
                        + overlap_occupied
                    ),
                    initial=0.0,
                )
            )
        )
        invariant_values.append(
            self._invariants(
                density_responses[spin_index],
                ground_density[spin_index],
                overlap,
                overlap_derivative,
            )
        )
    expected_total = density_responses[0] + density_responses[1]
    expected_total_occupied_virtual = density_parts[0][0] + density_parts[1][0]
    expected_total_metric = density_parts[0][1] + density_parts[1][1]
    total_comparisons = (
        (response.total_density_response, expected_total, "total density"),
        (
            response.total_density_response_occupied_virtual,
            expected_total_occupied_virtual,
            "total density OV",
        ),
        (
            response.total_density_response_metric,
            expected_total_metric,
            "total density metric",
        ),
    )
    for stored, expected, name in total_comparisons:
        if not np.allclose(stored, expected, rtol=0.0, atol=1.0e-12):
            raise UHFResponseError(
                f"the supplied UHF response {name} is inconsistent"
            )
    return responses, density_responses, metric_residuals, invariant_values, expected_total


def _rebuild_response_residuals(self, response, state, response_views):
    (
        coefficient, energy, _occupation, occupied, virtual, _minimum_gaps,
        _alpha_dimension, _beta_dimension, _overlap, overlap_derivative,
        hamiltonian_derivative,
    ) = state
    responses, _density_responses, _metric_residuals, _invariants, _total = response_views
    hamiltonian_mo = tuple(
        np.einsum(
            "mp,...mn,ni->...pi",
            coefficient[spin_index],
            hamiltonian_derivative[spin_index],
            coefficient[spin_index][:, occupied[spin_index]],
        )
        for spin_index in range(2)
    )
    overlap_mo = tuple(
        np.einsum(
            "mp,...mn,ni->...pi",
            coefficient[spin_index],
            overlap_derivative,
            coefficient[spin_index][:, occupied[spin_index]],
        )
        for spin_index in range(2)
    )
    residuals = self._orbital_residual(
        responses,
        hamiltonian_mo,
        overlap_mo,
        coefficient,
        energy,
        occupied,
        virtual,
    )
    stored_residuals = (
        response.alpha_orbital_response_residual,
        response.beta_orbital_response_residual,
    )
    for stored, expected, spin_name in zip(
        stored_residuals,
        residuals,
        ("alpha", "beta"),
        strict=True,
    ):
        if not np.allclose(stored, expected, rtol=0.0, atol=1.0e-12):
            raise UHFResponseError(
                f"the supplied UHF {spin_name} residual is not reproducible"
            )
    return residuals


def _audit_response_diagnostics(self, response, state, response_views, residuals):
    (
        _coefficient, _energy, _occupation, _occupied, _virtual, minimum_gaps,
        alpha_dimension, beta_dimension, _overlap, _overlap_derivative,
        _hamiltonian_derivative,
    ) = state
    (
        _responses, density_responses, metric_residuals,
        invariant_values, expected_total,
    ) = response_views
    alpha_maximum = float(np.max(np.abs(residuals[0]), initial=0.0))
    beta_maximum = float(np.max(np.abs(residuals[1]), initial=0.0))
    squared_sum = sum(float(np.sum(np.square(value))) for value in residuals)
    residual_size = sum(value.size for value in residuals)
    if len(response.atom_indices) == self.molecule.natm:
        alpha_translation = float(
            np.max(np.abs(np.sum(density_responses[0], axis=0)), initial=0.0)
        )
        beta_translation = float(
            np.max(np.abs(np.sum(density_responses[1], axis=0)), initial=0.0)
        )
        translation = float(
            np.max(np.abs(np.sum(expected_total, axis=0)), initial=0.0)
        )
    else:
        alpha_translation = beta_translation = translation = None
    measured = {
        "minimum_alpha_orbital_gap": minimum_gaps[0],
        "minimum_beta_orbital_gap": minimum_gaps[1],
        "response_dimension": alpha_dimension + beta_dimension,
        "alpha_response_dimension": alpha_dimension,
        "beta_response_dimension": beta_dimension,
        "maximum_residual": max(alpha_maximum, beta_maximum),
        "alpha_maximum_residual": alpha_maximum,
        "beta_maximum_residual": beta_maximum,
        "residual_rms": float(np.sqrt(squared_sum / residual_size)),
        "alpha_metric_residual": metric_residuals[0],
        "beta_metric_residual": metric_residuals[1],
        "alpha_idempotency_residual": invariant_values[0][0],
        "beta_idempotency_residual": invariant_values[1][0],
        "alpha_particle_number_residual": invariant_values[0][1],
        "beta_particle_number_residual": invariant_values[1][1],
        "alpha_translation_residual": alpha_translation,
        "beta_translation_residual": beta_translation,
        "translation_residual": translation,
    }
    for name, value in measured.items():
        recorded = getattr(response.diagnostics, name)
        if value is None:
            consistent = recorded is None
        elif isinstance(value, int):
            consistent = recorded == value
        else:
            consistent = np.isclose(recorded, value, rtol=1.0e-10, atol=1.0e-12)
        if not consistent:
            raise UHFResponseError(
                f"the supplied UHF response {name} diagnostic is inconsistent"
            )
    if measured["maximum_residual"] > self.residual_tolerance:
        raise UHFResponseError(
            "the supplied UHF response residual exceeds its tolerance"
        )
    invariant_maximum = max(
        measured[name]
        for name in (
            "alpha_metric_residual",
            "beta_metric_residual",
            "alpha_idempotency_residual",
            "beta_idempotency_residual",
            "alpha_particle_number_residual",
            "beta_particle_number_residual",
            "alpha_translation_residual",
            "beta_translation_residual",
            "translation_residual",
        )
        if measured[name] is not None
    )
    if invariant_maximum > self.invariant_tolerance:
        raise UHFResponseError(
            "the supplied UHF response invariant exceeds its tolerance"
        )


def audit_response_equations(self, response: UHFResponse) -> None:
    """Rebuild coupled equations and invariants for a supplied response."""
    state = _prepare_response_audit(self, response)
    response_views = _audit_spin_response_partitions(self, response, state)
    residuals = _rebuild_response_residuals(
        self, response, state, response_views
    )
    _audit_response_diagnostics(
        self, response, state, response_views, residuals
    )


__all__ = ['_validate_supplied_structure', 'audit_response_equations']

"""Bounded dense response-operator audits."""

from ..capabilities import DeePHFCapabilityError
from ..unrestricted_reference import UHFResponseError
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

"""UKS validation audit separated from production composition."""

from numbers import Real
from ..unrestricted_reference import UHFResponseError
from ..unrestricted_reference import UKSResponse
from ..unrestricted_reference import UKSResponseDiagnostics
from ..unrestricted_reference import UKSResponseError
from ..pyscf_dft_provenance import _grid_provenance
from ..unrestricted_reference import _uks_functional_provenance
import numpy as np
from ..unrestricted_reference import uks_response_integrity_fingerprint
from ..unrestricted_reference import validate_uks_reference
from ..pyscf_uks_response import (
    _require_wrapper_close,
)


def audit_uks_response_equations(self, response: UKSResponse) -> None:
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


__all__ = [
    "_response_operator_matrix_and_diagnostics",
    "_validate_supplied_structure",
    "audit_response_equations",
    "audit_uks_response_equations",
    "validate_response_operator_exact",
]
