"""Bounded dense audits separated from production solver assembly."""

from __future__ import annotations

from ..pyscf_dft_provenance import RKSFunctionalProvenance
from ..pyscf_dft_provenance import RKSGridProvenance
from ..pyscf_dft_provenance import RKSResponse
from ..pyscf_dft_provenance import RKSResponseDiagnostics
from ..pyscf_dft_provenance import RKSResponseError
from ..pyscf_dft_provenance import _functional_provenance
from ..pyscf_dft_provenance import _grid_provenance
from ..pyscf_dft_provenance import _validated_float64_array
from pyscf.dft import libxc
import numpy as np
import pyscf
from ..pyscf_rks_reference import rks_reference_fingerprint
from ..pyscf_rks_reference import rks_response_integrity_fingerprint
from ..pyscf_rks_reference import validate_rks_reference


def audit_response_equations(self, response: RKSResponse) -> None:
    """Independently rebuild every supplied equation without another solve."""
    validate_rks_reference(self.reference)
    if type(response) is not RKSResponse:
        raise RKSResponseError("the supplied RKS response has an invalid type")
    if type(response.diagnostics) is not RKSResponseDiagnostics:
        raise RKSResponseError(
            "the supplied RKS response diagnostics have an invalid type"
        )
    if response.reference_identity != id(self.reference):
        raise RKSResponseError("the supplied RKS response belongs to another reference")
    if response.state_fingerprint != rks_reference_fingerprint(self.reference):
        raise RKSResponseError("the supplied RKS response state is stale")
    if response.integrity_fingerprint != rks_response_integrity_fingerprint(response):
        raise RKSResponseError("the supplied RKS response integrity check failed")
    functional_provenance = _functional_provenance(self.reference)
    grid_provenance = _grid_provenance(self.reference)
    if (
        type(response.functional_provenance) is not RKSFunctionalProvenance
        or response.functional_provenance != functional_provenance
    ):
        raise RKSResponseError(
            "the supplied RKS response functional provenance is invalid"
        )
    if (
        type(response.grid_provenance) is not RKSGridProvenance
        or response.grid_provenance != grid_provenance
    ):
        raise RKSResponseError("the supplied RKS response grid provenance is invalid")
    coefficient, energy, occupation, occupied, virtual, minimum_gap = self._state()
    nmo = coefficient.shape[1]
    nocc = int(np.count_nonzero(occupied))
    nvir = int(np.count_nonzero(virtual))
    atom_indices = self._response_atom_indices(response.atom_indices)
    if atom_indices != response.atom_indices:
        raise RKSResponseError("the supplied RKS response atom selection is invalid")
    perturbation_shape = (len(atom_indices), 3)
    mo_shape = (*perturbation_shape, nmo, nocc)
    coefficient_shape = (*perturbation_shape, self.molecule.nao, nocc)
    density_shape = (
        *perturbation_shape,
        self.molecule.nao,
        self.molecule.nao,
    )
    residual_shape = (*perturbation_shape, nvir, nocc)
    expected_shapes = {
        "mo_response": mo_shape,
        "mo_response_occupied_virtual": mo_shape,
        "mo_response_metric": mo_shape,
        "coefficient_response": coefficient_shape,
        "coefficient_response_occupied_virtual": coefficient_shape,
        "coefficient_response_metric": coefficient_shape,
        "density_response": density_shape,
        "density_response_occupied_virtual": density_shape,
        "density_response_metric": density_shape,
        "overlap_derivative": density_shape,
        "hamiltonian_derivative": density_shape,
        "hamiltonian_derivative_fixed_grid": density_shape,
        "xc_hamiltonian_derivative_grid_coordinate": density_shape,
        "xc_hamiltonian_derivative_grid_weight": density_shape,
        "orbital_response_residual": residual_shape,
    }
    for name, expected_shape in expected_shapes.items():
        value = getattr(response, name)
        if type(value) is not np.ndarray or value.flags.writeable:
            raise RKSResponseError(
                f"the supplied RKS response {name} must be an immutable ndarray"
            )
        _validated_float64_array(value, expected_shape, f"supplied {name}")
    expected_overlap_derivative = self._overlap_derivative(atom_indices)
    (
        expected_hamiltonian_derivative,
        expected_hamiltonian_fixed_grid,
        expected_xc_grid_coordinate,
        expected_xc_grid_weight,
    ) = self._hamiltonian_derivative(coefficient, occupation, atom_indices)
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
        (
            response.hamiltonian_derivative_fixed_grid,
            expected_hamiltonian_fixed_grid,
            "fixed-grid Hamiltonian derivative",
        ),
        (
            response.xc_hamiltonian_derivative_grid_coordinate,
            expected_xc_grid_coordinate,
            "grid-coordinate XC Hamiltonian derivative",
        ),
        (
            response.xc_hamiltonian_derivative_grid_weight,
            expected_xc_grid_weight,
            "grid-weight XC Hamiltonian derivative",
        ),
    )
    for stored, expected, name in derivative_fields:
        if not np.allclose(stored, expected, rtol=0.0, atol=1.0e-11):
            raise RKSResponseError(
                f"the supplied RKS response {name} is not independently reproducible"
            )
    mo_partition_residual = float(
        np.max(
            np.abs(
                response.mo_response
                - response.mo_response_metric
                - response.mo_response_occupied_virtual
            ),
            initial=0.0,
        )
    )
    if (
        np.max(
            np.abs(response.mo_response_metric[..., virtual, :]),
            initial=0.0,
        )
        > 1.0e-12
        or np.max(
            np.abs(response.mo_response_occupied_virtual[..., occupied, :]),
            initial=0.0,
        )
        > 1.0e-12
        or mo_partition_residual > 1.0e-12
    ):
        raise RKSResponseError("the supplied RKS MO response partition is invalid")
    rebuilt_coefficients = {
        "coefficient_response": np.einsum(
            "mp,...pi->...mi",
            coefficient,
            response.mo_response,
        ),
        "coefficient_response_occupied_virtual": np.einsum(
            "mp,...pi->...mi",
            coefficient,
            response.mo_response_occupied_virtual,
        ),
        "coefficient_response_metric": np.einsum(
            "mp,...pi->...mi",
            coefficient,
            response.mo_response_metric,
        ),
    }
    for name, rebuilt in rebuilt_coefficients.items():
        if not np.allclose(getattr(response, name), rebuilt, rtol=0.0, atol=1.0e-11):
            raise RKSResponseError(
                f"the supplied RKS response {name} does not follow from its MO response"
            )
    rebuilt_densities = {
        "density_response": self._density_from_mo_response(
            response.mo_response,
            coefficient,
            occupation,
            occupied,
        ),
        "density_response_occupied_virtual": self._density_from_mo_response(
            response.mo_response_occupied_virtual,
            coefficient,
            occupation,
            occupied,
        ),
        "density_response_metric": self._density_from_mo_response(
            response.mo_response_metric,
            coefficient,
            occupation,
            occupied,
        ),
    }
    for name, rebuilt in rebuilt_densities.items():
        if not np.allclose(getattr(response, name), rebuilt, rtol=0.0, atol=1.0e-11):
            raise RKSResponseError(
                f"the supplied RKS response {name} does not follow from its MO response"
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
    physical_residual = self._orbital_residual(
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
        physical_residual,
        rtol=0.0,
        atol=1.0e-11,
    ):
        raise RKSResponseError(
            "the supplied RKS orbital residual is not independently reproducible"
        )
    response_dimension = int(np.count_nonzero(occupied)) * int(
        np.count_nonzero(virtual)
    )
    overlap = np.asarray(self.reference.get_ovlp())
    density_ground = np.asarray(self.reference.make_rdm1())
    overlap_occupied = overlap_mo[..., occupied, :]
    occupied_occupied_response = response.mo_response[..., occupied, :]
    metric_residual = max(
        float(
            np.max(
                np.abs(
                    occupied_occupied_response
                    + occupied_occupied_response.swapaxes(-1, -2)
                    + overlap_occupied
                ),
                initial=0.0,
            )
        ),
        float(
            np.max(
                np.abs(occupied_occupied_response + 0.5 * overlap_occupied),
                initial=0.0,
            )
        ),
    )
    idempotency = (
        np.einsum(
            "...ij,jk,kl->...il",
            response.density_response,
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
            response.density_response,
        )
        - 2.0 * response.density_response
    )
    particle_number = (
        np.einsum("...ij,ji->...", response.density_response, overlap)
        + np.einsum(
            "ij,...ji->...",
            density_ground,
            expected_overlap_derivative,
        )
    )
    translation_residual = (
        float(
            np.max(
                np.abs(np.sum(response.density_response, axis=0)),
                initial=0.0,
            )
        )
        if len(atom_indices) == self.molecule.natm
        else None
    )
    hamiltonian_reconstruction_residual = float(
        np.max(
            np.abs(
                expected_hamiltonian_derivative
                - expected_hamiltonian_fixed_grid
                - expected_xc_grid_coordinate
                - expected_xc_grid_weight
            ),
            initial=0.0,
        )
    )
    try:
        ao = self.reference._numint.eval_ao(
            self.molecule,
            self.reference.grids.coords,
            deriv=0,
        )
        rho = np.einsum(
            "gp,pq,gq->g",
            ao,
            density_ground,
            ao,
            optimize=True,
        )
        quadrature_electron_count = float(
            np.dot(self.reference.grids.weights, rho)
        )
    except Exception as error:
        raise RKSResponseError(
            f"RKS supplied-response quadrature audit failed: {error}"
        ) from error
    measured = {
        "minimum_orbital_gap": minimum_gap,
        "response_dimension": response_dimension,
        "hamiltonian_reconstruction_residual": (
            hamiltonian_reconstruction_residual
        ),
        "metric_residual": metric_residual,
        "idempotency_residual": float(
            np.max(np.abs(idempotency), initial=0.0)
        ),
        "particle_number_residual": float(
            np.max(np.abs(particle_number), initial=0.0)
        ),
        "translation_residual": translation_residual,
        "maximum_residual": float(
            np.max(np.abs(physical_residual), initial=0.0)
        ),
        "residual_rms": float(np.sqrt(np.mean(np.square(physical_residual)))),
        "quadrature_electron_count": quadrature_electron_count,
    }
    diagnostics = response.diagnostics
    if diagnostics.operator_is_self_adjoint is not True:
        raise RKSResponseError("the supplied RKS response operator contract is invalid")
    exact_diagnostics = {
        "pyscf_version": pyscf.__version__,
        "libxc_version": str(libxc.__version__),
        "functional_components": functional_provenance.components,
        "grid_point_count": grid_provenance.point_count,
        "grid_coordinates_fingerprint": grid_provenance.coordinates_fingerprint,
        "grid_weights_fingerprint": grid_provenance.weights_fingerprint,
        "cphf_tolerance": self.cphf_tolerance,
        "residual_tolerance": self.residual_tolerance,
        "invariant_tolerance": self.invariant_tolerance,
        "orbital_gap_tolerance": self.orbital_gap_tolerance,
        "max_cycle": self.max_cycle,
        "max_refinement_cycles": self.max_refinement_cycles,
        "level_shift": self.level_shift,
    }
    for name, expected in exact_diagnostics.items():
        if getattr(diagnostics, name) != expected:
            raise RKSResponseError(
                f"the supplied RKS response diagnostic {name} is invalid"
            )
    for name, expected in measured.items():
        if expected is None:
            if getattr(diagnostics, name) is not None:
                raise RKSResponseError(
                    f"the supplied RKS response diagnostic {name} is not reproducible"
                )
            continue
        if not np.isclose(
            getattr(diagnostics, name),
            expected,
            rtol=0.0,
            atol=1.0e-11,
        ):
            raise RKSResponseError(
                f"the supplied RKS response diagnostic {name} is not reproducible"
            )
    history = diagnostics.residual_history
    if (
        type(history) is not tuple
        or not history
        or diagnostics.refinement_cycles != len(history) - 1
        or not np.isfinite(history).all()
        or any(value < 0.0 for value in history)
        or any(
            later > earlier + self.residual_tolerance
            for earlier, later in zip(history, history[1:])
        )
        or not np.isclose(
            history[-1],
            measured["maximum_residual"],
            rtol=0.0,
            atol=1.0e-11,
        )
    ):
        raise RKSResponseError(
            "the supplied RKS response residual-refinement history is invalid"
        )
    if measured["maximum_residual"] > self.residual_tolerance:
        raise RKSResponseError(
            "the supplied RKS response physical residual exceeds tolerance"
        )
    invariant_names = (
        "hamiltonian_reconstruction_residual",
        "metric_residual",
        "idempotency_residual",
        "particle_number_residual",
        "translation_residual",
    )
    if max(
        measured[name]
        for name in invariant_names
        if measured[name] is not None
    ) > self.invariant_tolerance:
        raise RKSResponseError(
            "the supplied RKS response invariant exceeds tolerance"
        )


__all__ = ['audit_response_equations']
