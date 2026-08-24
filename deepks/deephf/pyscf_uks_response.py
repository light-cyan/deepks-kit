"""Internal implementation extracted from pyscf_uks.py."""

from dataclasses import replace
import numpy as np
from pyscf.grad import uks as uks_grad
from pyscf.hessian import uks as uks_hessian
from pyscf.scf import hf as scf_hf
from .capabilities import DeePHFCapabilityError
from .contracts import immutable_array as _immutable_array
from .pyscf_dft_provenance import (
    _grid_provenance,
    _normalized_atom_grid,
    _validated_grid_response_blocks,
)
from .pyscf_uhf_adjoint import UHFAdjointAdapter
from .pyscf_uhf_reference import (
    UHFAdjoint,
    UHFAdjointError,
    UHFResponseError,
    _native_unrestricted_gradient,
    _validated_float64_array,
)
from .pyscf_uhf_response import UHFResponseAdapter
from .pyscf_uks_reference import (
    UKSAdjoint,
    UKSAdjointDiagnostics,
    UKSAdjointError,
    UKSResponse,
    UKSResponseDiagnostics,
    UKSResponseError,
    _uks_functional_provenance,
    _validate_native_uks_gradient,
    uks_adjoint_integrity_fingerprint,
    uks_reference_fingerprint,
    uks_response_integrity_fingerprint,
    validate_uks_reference,
)

class _UKSLinearResponseMixin:
    """Replace the UHF J/K response by strict finite-grid UKS J plus LDA f_xc."""

    @staticmethod
    def _validate_reference(reference):
        return validate_uks_reference(reference)

    @staticmethod
    def _reference_fingerprint(reference) -> str:
        return uks_reference_fingerprint(reference)

    def _xc_nuclear_derivative_components(self, density: np.ndarray, atom_indices=None):
        molecule = self.molecule
        atom_indices = self._response_atom_indices(atom_indices)
        result_positions = {
            atom_index: result_index
            for result_index, atom_index in enumerate(atom_indices)
        }
        integration = self.reference._numint
        shape = (2, len(atom_indices), 3, molecule.nao, molecule.nao)
        grid_coordinate = np.zeros(shape)
        grid_weight = np.zeros(shape)
        blocks = _validated_grid_response_blocks(
            self.reference,
            _normalized_atom_grid(molecule, self.reference.grids.atom_grid),
            audit_weight_derivative=False,
        )
        for host_atom, (coordinates, weights, weight_derivative) in enumerate(blocks):
            try:
                ao = integration.eval_ao(molecule, coordinates, deriv=1)
                values = ao[0]
                gradients = ao[1:4]
                rho = np.stack(
                    [
                        np.einsum("gp,pq,gq->g", values, spin_density, values, optimize=True)
                        for spin_density in density
                    ]
                )
                xc_values = integration.eval_xc_eff(
                    self.reference.xc,
                    rho,
                    deriv=2,
                    xctype="LDA",
                    spin=1,
                )
                potential = np.asarray(xc_values[1])[:, 0]
                kernel = np.asarray(xc_values[2])[:, 0, :, 0]
            except Exception as error:
                raise UKSResponseError(
                    f"UKS LDA nuclear quadrature failed: {error}"
                ) from error
            values_to_check = (
                np.asarray(coordinates),
                np.asarray(weights),
                np.asarray(weight_derivative),
                values,
                gradients,
                rho,
                potential,
                kernel,
            )
            if not all(np.isfinite(value).all() for value in values_to_check):
                raise UKSResponseError("the UKS LDA nuclear quadrature is nonfinite")
            grid_weight += np.einsum(
                "axg,sg,gp,gq->saxpq",
                weight_derivative[list(atom_indices)],
                potential,
                values,
                values,
                optimize=True,
            )

            def accumulate(target, atom_index, axis, derivative_values):
                density_derivative = np.stack(
                    [
                        np.einsum(
                            "gp,pq,gq->g",
                            derivative_values,
                            spin_density,
                            values,
                            optimize=True,
                        )
                        + np.einsum(
                            "gp,pq,gq->g",
                            values,
                            spin_density,
                            derivative_values,
                            optimize=True,
                        )
                        for spin_density in density
                    ]
                )
                potential_derivative = np.einsum(
                    "tsg,tg->sg",
                    kernel,
                    density_derivative,
                    optimize=True,
                )
                target[:, atom_index, axis] += np.einsum(
                    "g,sg,gp,gq->spq",
                    weights,
                    potential_derivative,
                    values,
                    values,
                    optimize=True,
                )
                target[:, atom_index, axis] += np.einsum(
                    "g,sg,gp,gq->spq",
                    weights,
                    potential,
                    derivative_values,
                    values,
                    optimize=True,
                )
                target[:, atom_index, axis] += np.einsum(
                    "g,sg,gp,gq->spq",
                    weights,
                    potential,
                    values,
                    derivative_values,
                    optimize=True,
                )

            if host_atom in result_positions:
                result_index = result_positions[host_atom]
                for axis in range(3):
                    accumulate(grid_coordinate, result_index, axis, gradients[axis])
        return grid_coordinate, grid_weight

    def _hamiltonian_derivative(self, coefficient, occupation, atom_indices=None):
        atom_indices = self._response_atom_indices(atom_indices)
        density = np.asarray(self.reference.make_rdm1(coefficient, occupation))
        expected = (2, len(atom_indices), 3, self.molecule.nao, self.molecule.nao)
        try:
            hessian = uks_hessian.Hessian(self.reference)
            fixed_grid = hessian.make_h1(
                coefficient,
                occupation,
                atmlst=atom_indices,
            )
            fixed_grid = np.stack(
                [
                    [fixed_grid[spin][atom_index] for atom_index in atom_indices]
                    for spin in range(2)
                ]
            )
        except Exception as error:
            raise UKSResponseError(
                f"PySCF UKS Hamiltonian derivative construction failed: {error}"
            ) from error
        fixed_grid = _validated_float64_array(
            fixed_grid,
            expected,
            "fixed-grid UKS Hamiltonian derivative",
        )
        grid_coordinate, grid_weight = self._xc_nuclear_derivative_components(
            density,
            atom_indices,
        )
        full = fixed_grid + grid_coordinate + grid_weight
        if not all(
            np.isfinite(value).all()
            for value in (full, fixed_grid, grid_coordinate, grid_weight)
        ):
            raise UKSResponseError("the complete UKS Hamiltonian derivative is nonfinite")
        self._last_hamiltonian_components = (
            full,
            fixed_grid,
            grid_coordinate,
            grid_weight,
        )
        return full[0], full[1]

    def _induced_potential(self, alpha_density, beta_density):
        perturbation_shape = alpha_density.shape[:-2]
        flat_alpha = np.asarray(alpha_density).reshape(-1, self.molecule.nao, self.molecule.nao)
        flat_beta = np.asarray(beta_density).reshape(flat_alpha.shape)
        if not np.isfinite(flat_alpha).all() or not np.isfinite(flat_beta).all():
            raise UKSResponseError("the UKS trial density response is nonfinite")
        if max(
            float(np.max(np.abs(flat_alpha - flat_alpha.swapaxes(-1, -2)), initial=0.0)),
            float(np.max(np.abs(flat_beta - flat_beta.swapaxes(-1, -2)), initial=0.0)),
        ) > 1.0e-10:
            raise UKSResponseError("the UKS trial density response is not symmetric")
        try:
            coulomb, _exchange = scf_hf.get_jk(
                self.molecule,
                np.stack((flat_alpha, flat_beta)),
                hermi=1,
            )
            total_coulomb = np.asarray(coulomb[0] + coulomb[1])
            coordinates = np.asarray(self.reference.grids.coords)
            weights = np.asarray(self.reference.grids.weights)
            integration = self.reference._numint
            ao = integration.eval_ao(self.molecule, coordinates, deriv=0)
            ground_density = np.asarray(self.reference.make_rdm1())
            rho = np.stack(
                [
                    np.einsum("gp,pq,gq->g", ao, spin_density, ao, optimize=True)
                    for spin_density in ground_density
                ]
            )
            kernel = np.asarray(
                integration.eval_xc_eff(
                    self.reference.xc,
                    rho,
                    deriv=2,
                    xctype="LDA",
                    spin=1,
                )[2]
            )[:, 0, :, 0]
            density_response = np.stack(
                [
                    np.einsum("gp,xpq,gq->xg", ao, spin_density, ao, optimize=True)
                    for spin_density in (flat_alpha, flat_beta)
                ]
            )
            potential_response = np.einsum(
                "tsg,txg->sxg",
                kernel,
                density_response,
                optimize=True,
            )
            xc_response = np.einsum(
                "g,sxg,gp,gq->sxpq",
                weights,
                potential_response,
                ao,
                ao,
                optimize=True,
            )
        except Exception as error:
            raise UKSResponseError(
                f"the independent UKS J plus LDA f_xc action failed: {error}"
            ) from error
        expected = (*perturbation_shape, self.molecule.nao, self.molecule.nao)
        alpha = total_coulomb + xc_response[0]
        beta = total_coulomb + xc_response[1]
        return alpha.reshape(expected), beta.reshape(expected)


class _UKSInternalResponseAdapter(_UKSLinearResponseMixin, UHFResponseAdapter):
    pass


class _UKSInternalAdjointAdapter(_UKSLinearResponseMixin, UHFAdjointAdapter):
    pass


def _require_wrapper_close(actual, expected, name: str, error_type) -> None:
    if not np.allclose(actual, expected, rtol=1.0e-11, atol=1.0e-12):
        residual = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
        raise error_type(f"the UKS {name} is inconsistent: residual {residual:.3e}")


class UKSResponseAdapter:
    """Solve and independently audit complete finite-grid UKS response."""

    def __init__(self, reference, **controls):
        try:
            self._core = _UKSInternalResponseAdapter(reference, **controls)
        except (DeePHFCapabilityError, UKSResponseError):
            raise
        except UHFResponseError as error:
            raise UKSResponseError(f"UKS response setup failed: {error}") from error
        self.reference = self._core.reference
        for name in (
            "cphf_tolerance",
            "residual_tolerance",
            "invariant_tolerance",
            "orbital_gap_tolerance",
            "max_cycle",
            "max_refinement_cycles",
            "level_shift",
            "operator_stability_tolerance",
            "operator_condition_tolerance",
            "operator_symmetry_tolerance",
            "operator_dimension_limit",
        ):
            setattr(self, name, getattr(self._core, name))

    @staticmethod
    def _components(core) -> tuple[np.ndarray, ...]:
        components = getattr(core, "_last_hamiltonian_components", None)
        if type(components) is not tuple or len(components) != 4:
            raise UKSResponseError("the UKS Hamiltonian derivative partitions are unavailable")
        return tuple(np.stack(value) if isinstance(value, tuple) else value for value in components)

    def solve(self, atom_indices=None) -> UKSResponse:
        """Return one immutable UKS response for selected atoms."""
        return self._solve(atom_indices, "response")

    def _solve_with_density_partitions(self, atom_indices=None):
        """Return a response and its transient spin-density work arrays."""
        return self._solve(atom_indices, "partitions")

    def _solve_for_gradient(self, objective, atom_indices=None):
        """Return compact diagnostics and the final density contraction."""
        return self._solve(atom_indices, "gradient", objective)

    def _solve(self, atom_indices, result_mode, objective=None):
        try:
            if result_mode == "gradient":
                core_diagnostics, density_partitions = self._core._solve_for_gradient(
                    objective,
                    atom_indices=atom_indices
                )
                core_response = None
            elif result_mode == "partitions":
                core_response, density_partitions = (
                    self._core._solve_with_density_partitions(
                        atom_indices=atom_indices
                    )
                )
                core_diagnostics = core_response.diagnostics
            else:
                core_response = self._core.solve(atom_indices=atom_indices)
                core_diagnostics = core_response.diagnostics
            if result_mode == "gradient":
                components = self._core._last_hamiltonian_components
                reconstruction = max(
                    float(np.max(np.abs(a - b - c - d), initial=0.0))
                    for a, b, c, d in zip(*components, strict=True)
                )
            else:
                full, fixed, coordinate, weight = self._components(self._core)
                _require_wrapper_close(
                    full,
                    fixed + coordinate + weight,
                    "Hamiltonian derivative partition",
                    UKSResponseError,
                )
                reconstruction = float(
                    np.max(np.abs(full - fixed - coordinate - weight), initial=0.0)
                )
            functional = _uks_functional_provenance(self.reference)
            grid = _grid_provenance(self.reference)
            diagnostics = UKSResponseDiagnostics(
                core=core_diagnostics,
                functional=functional,
                grid=grid,
                hamiltonian_reconstruction_residual=reconstruction,
            )
            if result_mode == "gradient":
                return diagnostics, density_partitions
            response = UKSResponse(
                core=core_response,
                functional=functional,
                grid=grid,
                hamiltonian_derivative_fixed_grid_spin=_immutable_array(fixed),
                xc_hamiltonian_derivative_grid_coordinate_spin=_immutable_array(coordinate),
                xc_hamiltonian_derivative_grid_weight_spin=_immutable_array(weight),
                diagnostics=diagnostics,
                integrity_fingerprint="",
            )
            response = replace(
                response,
                integrity_fingerprint=uks_response_integrity_fingerprint(response),
            )
            return (
                (response, density_partitions)
                if result_mode == "partitions"
                else response
            )
        except DeePHFCapabilityError:
            raise
        except UKSResponseError:
            raise
        except UHFResponseError as error:
            raise UKSResponseError(f"UKS response evaluation failed: {error}") from error

    def validate_response_operator_exact(self):
        """Run the bounded explicit debug audit of the internal UKS operator."""
        return self._core.validate_response_operator_exact()

    def audit_response_equations(self, response: UKSResponse) -> None:
        from .audits.uks_response import audit_response_equations as audit
        return audit(self, response)


class UKSAdjointAdapter:
    """Solve one finite-grid UKS correction-specific coupled scalar adjoint."""

    def __init__(self, reference, **controls):
        try:
            self._core = _UKSInternalAdjointAdapter(reference, **controls)
        except (DeePHFCapabilityError, UKSAdjointError):
            raise
        except UHFAdjointError as error:
            raise UKSAdjointError(f"UKS adjoint setup failed: {error}") from error
        self.reference = self._core.reference
        for name in (
            "residual_tolerance",
            "invariant_tolerance",
            "orbital_gap_tolerance",
            "operator_stability_tolerance",
            "operator_condition_tolerance",
            "operator_symmetry_tolerance",
            "operator_dimension_limit",
            "objective_symmetry_tolerance",
            "max_cycle",
            "krylov_restart",
        ):
            setattr(self, name, getattr(self._core, name))

    def _nuclear_partitions(self, core_adjoint: UHFAdjoint):
        atom_indices = core_adjoint.atom_indices
        coefficient, energy, occupation, occupied, virtual, _ = self._core._state()
        overlap = self._core._overlap_derivative(atom_indices)
        full, fixed, coordinate, weight = UKSResponseAdapter._components(self._core)
        zvector = (core_adjoint.alpha_zvector, core_adjoint.beta_zvector)
        fixed_spin = []
        coordinate_spin = []
        weight_spin = []
        for spin in range(2):
            occupied_coefficients = coefficient[spin][:, occupied[spin]]
            overlap_mo = np.einsum("mp,...mn,ni->...pi", coefficient[spin], overlap, occupied_coefficients)
            fixed_mo = np.einsum("mp,...mn,ni->...pi", coefficient[spin], fixed[spin], occupied_coefficients)
            coordinate_mo = np.einsum("mp,...mn,ni->...pi", coefficient[spin], coordinate[spin], occupied_coefficients)
            weight_mo = np.einsum("mp,...mn,ni->...pi", coefficient[spin], weight[spin], occupied_coefficients)
            fixed_rhs = (
                fixed_mo[..., virtual[spin], :]
                - overlap_mo[..., virtual[spin], :]
                * energy[spin, occupied[spin]]
            )
            fixed_spin.append(-np.einsum("ai,...ai->...", zvector[spin], fixed_rhs))
            coordinate_spin.append(-np.einsum("ai,...ai->...", zvector[spin], coordinate_mo[..., virtual[spin], :]))
            weight_spin.append(-np.einsum("ai,...ai->...", zvector[spin], weight_mo[..., virtual[spin], :]))
        fixed_spin = np.stack(fixed_spin)
        coordinate_spin = np.stack(coordinate_spin)
        weight_spin = np.stack(weight_spin)
        residual = float(
            np.max(
                np.abs(
                    core_adjoint.correction_gradient_adjoint_nuclear_spin
                    - fixed_spin
                    - coordinate_spin
                    - weight_spin
                ),
                initial=0.0,
            )
        )
        return fixed_spin, coordinate_spin, weight_spin, residual

    def validate_response_operator_exact(self):
        """Run the bounded explicit debug audit of the internal UKS operator."""
        return self._core.validate_response_operator_exact()

    def solve(self, objective_ao_potential: np.ndarray, atom_indices=None, compact=False):
        """Return one immutable UKS adjoint from exactly one transpose solve."""
        try:
            if compact:
                core_diagnostics, response = self._core.solve(
                    objective_ao_potential,
                    atom_indices=atom_indices,
                    compact=True,
                )
                return UKSAdjointDiagnostics(
                    core=core_diagnostics,
                    functional=_uks_functional_provenance(self.reference),
                    grid=_grid_provenance(self.reference),
                    nuclear_partition_residual=None,
                ), response
            core_adjoint = self._core.solve(
                objective_ao_potential,
                atom_indices=atom_indices,
            )
            fixed_spin, coordinate_spin, weight_spin, partition_residual = self._nuclear_partitions(core_adjoint)
            if partition_residual > self.invariant_tolerance:
                raise UKSAdjointError("the UKS adjoint nuclear partitions are inconsistent")
            functional = _uks_functional_provenance(self.reference)
            grid = _grid_provenance(self.reference)
            diagnostics = UKSAdjointDiagnostics(
                core=core_adjoint.diagnostics,
                functional=functional,
                grid=grid,
                nuclear_partition_residual=partition_residual,
            )
            adjoint = UKSAdjoint(
                core=core_adjoint,
                functional=functional,
                grid=grid,
                correction_gradient_adjoint_fixed_grid_spin=_immutable_array(fixed_spin),
                correction_gradient_adjoint_grid_coordinate_spin=_immutable_array(coordinate_spin),
                correction_gradient_adjoint_grid_weight_spin=_immutable_array(weight_spin),
                correction_gradient_adjoint_fixed_grid=_immutable_array(fixed_spin.sum(axis=0)),
                correction_gradient_adjoint_grid_coordinate=_immutable_array(coordinate_spin.sum(axis=0)),
                correction_gradient_adjoint_grid_weight=_immutable_array(weight_spin.sum(axis=0)),
                diagnostics=diagnostics,
                integrity_fingerprint="",
            )
            return replace(adjoint, integrity_fingerprint=uks_adjoint_integrity_fingerprint(adjoint))
        except DeePHFCapabilityError:
            raise
        except UKSAdjointError:
            raise
        except UHFAdjointError as error:
            raise UKSAdjointError(f"UKS adjoint evaluation failed: {error}") from error

    def audit_adjoint(self, adjoint: UKSAdjoint, expected_objective_ao_potential: np.ndarray) -> None:
        from .audits.uks_adjoint import audit_adjoint as audit
        return audit(self, adjoint, expected_objective_ao_potential)


def native_uks_gradient(reference, atom_indices=None) -> np.ndarray:
    """Evaluate one selected native UKS gradient with grid response."""
    validate_uks_reference(reference)
    _validate_native_uks_gradient()
    atom_indices = (
        tuple(range(reference.mol.natm))
        if atom_indices is None
        else tuple(atom_indices)
    )
    initial_fingerprint = uks_reference_fingerprint(reference)
    try:
        driver = uks_grad.Gradients(reference)
        if type(driver) is not uks_grad.Gradients:
            raise UKSResponseError("the native UKS gradient driver type is invalid")
        driver.grids = reference.grids
        driver.grid_response = True
        gradient = _native_unrestricted_gradient(reference, driver, atom_indices)
    except UKSResponseError:
        raise
    except Exception as error:
        raise UKSResponseError(f"PySCF native UKS gradient failed: {error}") from error
    gradient = _validated_float64_array(
        gradient,
        (len(atom_indices), 3),
        "native UKS gradient",
    )
    _validate_native_uks_gradient()
    validate_uks_reference(reference)
    if uks_reference_fingerprint(reference) != initial_fingerprint:
        raise UKSResponseError("the UKS reference changed during native gradient evaluation")
    return gradient


__all__ = [
    "UKSAdjoint",
    "UKSAdjointAdapter",
    "UKSAdjointDiagnostics",
    "UKSAdjointError",
    "UKSResponse",
    "UKSResponseAdapter",
    "UKSResponseDiagnostics",
    "UKSResponseError",
    "native_uks_gradient",
    "uks_adjoint_integrity_fingerprint",
    "uks_reference_fingerprint",
    "uks_response_integrity_fingerprint",
    "validate_uks_reference",
]
