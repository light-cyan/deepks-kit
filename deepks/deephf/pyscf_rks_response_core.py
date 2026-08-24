"""Internal implementation extracted from pyscf_rks.py."""

import hashlib
import numpy as np
from pyscf.hessian import rks as rks_hessian
from pyscf.scf import hf as scf_hf
from .adjoint import ScalarAdjointProblem
from .capabilities import DeePHFCapabilityError
from .pyscf_dft_provenance import (
    RKSResponseError,
    _cycle_limit,
    _normalized_atom_grid,
    _response_real_control,
    _validated_float64_array,
    _validated_grid_response_blocks,
)
from .pyscf_rks_reference import validate_rks_reference, rks_reference_fingerprint

class _RKSLinearResponseProblem:
    """Bind one action-only RKS operator to the reference-neutral protocol."""

    is_self_adjoint = True

    def __init__(self, adapter: "_RKSLinearResponseCore"):
        self._adapter = adapter
        self._state_fingerprint = rks_reference_fingerprint(adapter.reference)
        coefficient, _energy, occupation, occupied, virtual, _gap = adapter._state()
        self._dimension = int(np.count_nonzero(occupied)) * int(
            np.count_nonzero(virtual)
        )

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def operator_fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"pyscf-2.14-rks-occupied-virtual-operator-v1")
        digest.update(self._state_fingerprint.encode("ascii"))
        return digest.hexdigest()

    def _validate_state(self) -> None:
        validate_rks_reference(self._adapter.reference)
        if rks_reference_fingerprint(self._adapter.reference) != self._state_fingerprint:
            raise RKSResponseError(
                "the RKS reference changed after the linear-response problem was built"
            )

    def apply(self, vector: np.ndarray) -> np.ndarray:
        self._validate_state()
        vector = _validated_float64_array(
            vector,
            (self.dimension,),
            "RKS linear-response vector",
        )
        coefficient, energy, occupation, occupied, virtual, _gap = (
            self._adapter._state()
        )
        response = vector.reshape(
            int(np.count_nonzero(virtual)),
            int(np.count_nonzero(occupied)),
        )
        return self._adapter._apply_occupied_virtual_operator(
            response,
            coefficient,
            energy,
            occupation,
            occupied,
            virtual,
        ).reshape(-1)

    def apply_transpose(self, vector: np.ndarray) -> np.ndarray:
        self._validate_state()
        vector = _validated_float64_array(
            vector,
            (self.dimension,),
            "RKS transpose linear-response vector",
        )
        return self.apply(vector)

    def precondition(self, vector: np.ndarray) -> np.ndarray:
        self._adapter._count_operation("preconditioner_actions")
        self._validate_state()
        _coefficient, energy, _occupation, occupied, virtual, _gap = (
            self._adapter._state()
        )
        gaps = energy[virtual, None] - energy[occupied]
        return np.asarray(vector).reshape(self.dimension) / gaps.reshape(-1)


class _RKSLinearResponseCore:
    """Provide shared pure-LDA RKS operator and nuclear perturbation primitives."""

    def __init__(
        self,
        reference,
        *,
        cphf_tolerance: float = 1.0e-11,
        residual_tolerance: float = 1.0e-9,
        invariant_tolerance: float = 1.0e-9,
        orbital_gap_tolerance: float = 1.0e-7,
        max_cycle: int = 100,
        max_refinement_cycles: int = 3,
        level_shift: float = 0.0,
        operator_stability_tolerance: float = 1.0e-6,
        operator_condition_tolerance: float = 1.0e8,
        operator_symmetry_tolerance: float = 1.0e-10,
        operator_dimension_limit: int = 512,
    ):
        self._operation_hook = None
        self.reference = validate_rks_reference(reference)
        self.cphf_tolerance = _response_real_control(
            cphf_tolerance,
            "cphf_tolerance",
        )
        self.residual_tolerance = _response_real_control(
            residual_tolerance,
            "residual_tolerance",
        )
        self.invariant_tolerance = _response_real_control(
            invariant_tolerance,
            "invariant_tolerance",
        )
        self.orbital_gap_tolerance = _response_real_control(
            orbital_gap_tolerance,
            "orbital_gap_tolerance",
        )
        self.max_cycle = _cycle_limit(max_cycle, "max_cycle")
        self.max_refinement_cycles = _cycle_limit(
            max_refinement_cycles,
            "max_refinement_cycles",
        )
        self.level_shift = _response_real_control(level_shift, "level_shift")
        self.operator_stability_tolerance = _response_real_control(
            operator_stability_tolerance,
            "operator_stability_tolerance",
        )
        self.operator_condition_tolerance = _response_real_control(
            operator_condition_tolerance,
            "operator_condition_tolerance",
        )
        self.operator_symmetry_tolerance = _response_real_control(
            operator_symmetry_tolerance,
            "operator_symmetry_tolerance",
        )
        self.operator_dimension_limit = _cycle_limit(
            operator_dimension_limit,
            "operator_dimension_limit",
        )
        positive = (
            self.cphf_tolerance,
            self.residual_tolerance,
            self.invariant_tolerance,
            self.orbital_gap_tolerance,
            self.operator_stability_tolerance,
            self.operator_symmetry_tolerance,
        )
        if any(value <= 0.0 for value in positive):
            raise ValueError("response tolerances must be positive")
        if self.operator_condition_tolerance <= 1.0:
            raise ValueError("operator_condition_tolerance must exceed one")
        if self.max_cycle <= 0 or self.max_refinement_cycles < 0:
            raise ValueError("response cycle limits are invalid")
        if self.operator_dimension_limit <= 0:
            raise ValueError("operator_dimension_limit must be positive")

    def _count_operation(self, name: str) -> None:
        if self._operation_hook is not None:
            self._operation_hook(name)

    @property
    def molecule(self):
        return self.reference.mol

    def _response_atom_indices(self, atom_indices) -> tuple[int, ...]:
        from .driver import validate_atom_indices

        selected = validate_atom_indices(self.molecule, atom_indices)
        return tuple(range(self.molecule.natm)) if selected is None else selected

    def _state(self):
        coefficient = np.asarray(self.reference.mo_coeff)
        energy = np.asarray(self.reference.mo_energy)
        occupation = np.asarray(self.reference.mo_occ)
        occupied = occupation > 0
        virtual = occupation == 0
        gaps = energy[virtual, None] - energy[occupied]
        minimum_gap = float(np.min(gaps))
        if not np.isfinite(minimum_gap) or minimum_gap <= self.orbital_gap_tolerance:
            raise DeePHFCapabilityError(
                "RKS occupied-virtual gap is outside the strict response domain: "
                f"{minimum_gap:.3e} <= {self.orbital_gap_tolerance:.3e}"
            )
        return coefficient, energy, occupation, occupied, virtual, minimum_gap

    def _overlap_derivative(self, atom_indices=None) -> np.ndarray:
        molecule = self.molecule
        atom_indices = self._response_atom_indices(atom_indices)
        try:
            integral = -molecule.intor("int1e_ipovlp", comp=3)
        except Exception as error:
            raise RKSResponseError(
                f"PySCF overlap-derivative construction failed: {error}"
            ) from error
        integral = _validated_float64_array(
            integral,
            (3, molecule.nao, molecule.nao),
            "overlap-derivative integral",
        )
        result = np.zeros((len(atom_indices), 3, molecule.nao, molecule.nao))
        atom_slices = molecule.aoslice_by_atom()
        for result_index, atom_index in enumerate(atom_indices):
            atom_slice = atom_slices[atom_index]
            ao_start, ao_stop = atom_slice[2:]
            result[result_index, :, ao_start:ao_stop] += integral[:, ao_start:ao_stop]
            result[result_index, :, :, ao_start:ao_stop] += integral[
                :, ao_start:ao_stop
            ].transpose(0, 2, 1)
        return result

    @staticmethod
    def _density_from_mo_response(
        mo_response: np.ndarray,
        coefficient: np.ndarray,
        occupation: np.ndarray,
        occupied: np.ndarray,
    ) -> np.ndarray:
        occupied_coefficients = coefficient[:, occupied]
        coefficient_response = np.einsum(
            "mp,...pi->...mi",
            coefficient,
            mo_response,
        )
        one_sided = np.einsum(
            "...pi,qi,i->...pq",
            coefficient_response,
            occupied_coefficients,
            occupation[occupied],
        )
        return one_sided + one_sided.swapaxes(-1, -2)

    def _xc_nuclear_derivative_components(
        self,
        density: np.ndarray,
        atom_indices=None,
    ) -> tuple[np.ndarray, np.ndarray]:
        molecule = self.molecule
        atom_indices = self._response_atom_indices(atom_indices)
        result_positions = {
            atom_index: result_index
            for result_index, atom_index in enumerate(atom_indices)
        }
        integration = self.reference._numint
        shape = (len(atom_indices), 3, molecule.nao, molecule.nao)
        grid_coordinate = np.zeros(shape)
        grid_weight = np.zeros(shape)
        atom_grid = _normalized_atom_grid(
            molecule,
            self.reference.grids.atom_grid,
        )
        blocks = _validated_grid_response_blocks(
            self.reference,
            atom_grid,
            audit_weight_derivative=False,
        )
        for host_atom, (coordinates, weights, weight_derivative) in enumerate(blocks):
            coordinates = np.asarray(coordinates)
            weights = np.asarray(weights)
            weight_derivative = np.asarray(weight_derivative)
            if weight_derivative.shape != (molecule.natm, 3, weights.size):
                raise RKSResponseError("the RKS grid-weight derivative shape is invalid")
            try:
                ao = integration.eval_ao(molecule, coordinates, deriv=1)
                values = ao[0]
                gradients = ao[1:4]
                rho = np.einsum(
                    "gp,pq,gq->g",
                    values,
                    density,
                    values,
                    optimize=True,
                )
                xc_values = integration.eval_xc_eff(
                    self.reference.xc,
                    rho,
                    deriv=2,
                    xctype="LDA",
                    spin=0,
                )
                potential = np.asarray(xc_values[1])[0]
                kernel = np.asarray(xc_values[2])[0, 0]
            except Exception as error:
                raise RKSResponseError(
                    f"RKS LDA nuclear quadrature failed: {error}"
                ) from error
            quadrature_values = (
                coordinates,
                weights,
                weight_derivative,
                values,
                gradients,
                rho,
                potential,
                kernel,
            )
            if not all(np.isfinite(value).all() for value in quadrature_values):
                raise RKSResponseError("RKS LDA nuclear quadrature is nonfinite")
            grid_weight += np.einsum(
                "axg,g,gp,gq->axpq",
                weight_derivative[list(atom_indices)],
                potential,
                values,
                values,
                optimize=True,
            )

            def accumulate(target, atom_index, axis, derivative_values):
                density_derivative = np.einsum(
                    "gp,pq,gq->g",
                    derivative_values,
                    density,
                    values,
                    optimize=True,
                )
                density_derivative += np.einsum(
                    "gp,pq,gq->g",
                    values,
                    density,
                    derivative_values,
                    optimize=True,
                )
                target[atom_index, axis] += np.einsum(
                    "g,g,g,gp,gq->pq",
                    weights,
                    kernel,
                    density_derivative,
                    values,
                    values,
                    optimize=True,
                )
                target[atom_index, axis] += np.einsum(
                    "g,g,gp,gq->pq",
                    weights,
                    potential,
                    derivative_values,
                    values,
                    optimize=True,
                )
                target[atom_index, axis] += np.einsum(
                    "g,g,gp,gq->pq",
                    weights,
                    potential,
                    values,
                    derivative_values,
                    optimize=True,
                )

            if host_atom in result_positions:
                result_index = result_positions[host_atom]
                for axis in range(3):
                    accumulate(
                        grid_coordinate,
                        result_index,
                        axis,
                        gradients[axis],
                    )
        return grid_coordinate, grid_weight

    def _hamiltonian_derivative(
        self,
        coefficient: np.ndarray,
        occupation: np.ndarray,
        atom_indices=None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        atom_indices = self._response_atom_indices(atom_indices)
        density = np.asarray(self.reference.make_rdm1(coefficient, occupation))
        expected_shape = (
            len(atom_indices),
            3,
            self.molecule.nao,
            self.molecule.nao,
        )
        try:
            hessian = rks_hessian.Hessian(self.reference)
            fixed_grid_hamiltonian = hessian.make_h1(
                coefficient,
                occupation,
                atmlst=atom_indices,
            )
            fixed_grid_hamiltonian = [
                fixed_grid_hamiltonian[index] for index in atom_indices
            ]
        except Exception as error:
            raise RKSResponseError(
                f"PySCF RKS Hamiltonian derivative construction failed: {error}"
            ) from error
        fixed_grid_hamiltonian = _validated_float64_array(
            fixed_grid_hamiltonian,
            expected_shape,
            "fixed-grid Hamiltonian derivative",
        )
        grid_coordinate, grid_weight = (
            self._xc_nuclear_derivative_components(density, atom_indices)
        )
        full_hamiltonian = fixed_grid_hamiltonian + grid_coordinate + grid_weight
        arrays = (
            full_hamiltonian,
            fixed_grid_hamiltonian,
            grid_coordinate,
            grid_weight,
        )
        if not all(np.isfinite(value).all() for value in arrays):
            raise RKSResponseError("the complete RKS Hamiltonian derivative is nonfinite")
        return arrays

    def _induced_potential_components(
        self,
        density_response: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        perturbation_shape = density_response.shape[:-2]
        flat_density = np.asarray(density_response).reshape(
            -1,
            self.molecule.nao,
            self.molecule.nao,
        )
        if not np.isfinite(flat_density).all():
            raise RKSResponseError("the RKS trial density response is nonfinite")
        symmetry_residual = float(
            np.max(
                np.abs(flat_density - flat_density.swapaxes(-1, -2)),
                initial=0.0,
            )
        )
        if symmetry_residual > 1.0e-10:
            raise RKSResponseError("the RKS trial density response is not symmetric")
        try:
            coulomb, _exchange = scf_hf.get_jk(
                self.molecule,
                flat_density,
                hermi=1,
            )
        except Exception as error:
            raise RKSResponseError(
                f"PySCF direct Coulomb response failed: {error}"
            ) from error
        coulomb = _validated_float64_array(
            coulomb,
            flat_density.shape,
            "induced Coulomb response",
        )
        xc_response = np.zeros_like(flat_density)
        ground_density = np.asarray(self.reference.make_rdm1())
        coordinates = np.asarray(self.reference.grids.coords)
        weights = np.asarray(self.reference.grids.weights)
        integration = self.reference._numint
        block_size = 2048
        try:
            for start in range(0, weights.size, block_size):
                stop = min(start + block_size, weights.size)
                block_coordinates = coordinates[start:stop]
                block_weights = weights[start:stop]
                ao = integration.eval_ao(
                    self.molecule,
                    block_coordinates,
                    deriv=0,
                )
                rho = np.einsum(
                    "gp,pq,gq->g",
                    ao,
                    ground_density,
                    ao,
                    optimize=True,
                )
                kernel = np.asarray(
                    integration.eval_xc_eff(
                        self.reference.xc,
                        rho,
                        deriv=2,
                        xctype="LDA",
                        spin=0,
                    )[2]
                )[0, 0]
                rho_response = np.einsum(
                    "gp,xpq,gq->xg",
                    ao,
                    flat_density,
                    ao,
                    optimize=True,
                )
                xc_response += np.einsum(
                    "g,g,xg,gp,gq->xpq",
                    block_weights,
                    kernel,
                    rho_response,
                    ao,
                    ao,
                    optimize=True,
                )
        except Exception as error:
            raise RKSResponseError(
                f"independent dense LDA f_xc response failed: {error}"
            ) from error
        if not np.isfinite(xc_response).all():
            raise RKSResponseError("the induced LDA f_xc response is nonfinite")
        expected_shape = (*perturbation_shape, self.molecule.nao, self.molecule.nao)
        return coulomb.reshape(expected_shape), xc_response.reshape(expected_shape)

    def _induced_potential(self, density_response: np.ndarray) -> np.ndarray:
        coulomb, xc_response = self._induced_potential_components(density_response)
        return coulomb + xc_response

    def _pyscf_induced_potential(self, density_response: np.ndarray) -> np.ndarray:
        """Apply PySCF's CPKS J + nr_rks_fxc action for the iterative solver."""
        perturbation_shape = density_response.shape[:-2]
        flat_density = np.asarray(density_response).reshape(
            -1,
            self.molecule.nao,
            self.molecule.nao,
        )
        try:
            coulomb, _exchange = scf_hf.get_jk(
                self.molecule,
                flat_density,
                hermi=1,
            )
            xc_response = self.reference._numint.nr_rks_fxc(
                self.molecule,
                self.reference.grids,
                self.reference.xc,
                np.asarray(self.reference.make_rdm1()),
                flat_density,
                hermi=1,
            )
        except Exception as error:
            raise RKSResponseError(
                f"PySCF iterative J + LDA f_xc action failed: {error}"
            ) from error
        expected = flat_density.shape
        coulomb = _validated_float64_array(
            coulomb,
            expected,
            "PySCF iterative Coulomb response",
        )
        xc_response = _validated_float64_array(
            xc_response,
            expected,
            "PySCF iterative LDA f_xc response",
        )
        return (coulomb + xc_response).reshape(
            *perturbation_shape,
            self.molecule.nao,
            self.molecule.nao,
        )

    def _induced_mo_potential(
        self,
        mo_response: np.ndarray,
        coefficient: np.ndarray,
        occupation: np.ndarray,
        occupied: np.ndarray,
    ) -> np.ndarray:
        density_response = self._density_from_mo_response(
            mo_response,
            coefficient,
            occupation,
            occupied,
        )
        induced = self._induced_potential(density_response)
        return np.einsum(
            "mp,...mn,ni->...pi",
            coefficient,
            induced,
            coefficient[:, occupied],
        )

    def _pyscf_induced_mo_potential(
        self,
        mo_response: np.ndarray,
        coefficient: np.ndarray,
        occupation: np.ndarray,
        occupied: np.ndarray,
    ) -> np.ndarray:
        density_response = self._density_from_mo_response(
            mo_response,
            coefficient,
            occupation,
            occupied,
        )
        induced = self._pyscf_induced_potential(density_response)
        return np.einsum(
            "mp,...mn,ni->...pi",
            coefficient,
            induced,
            coefficient[:, occupied],
        )

    def _apply_occupied_virtual_operator(
        self,
        occupied_virtual_response: np.ndarray,
        coefficient: np.ndarray,
        energy: np.ndarray,
        occupation: np.ndarray,
        occupied: np.ndarray,
        virtual: np.ndarray,
    ) -> np.ndarray:
        self._count_operation("response_operator_actions")
        nocc = int(np.count_nonzero(occupied))
        nvir = int(np.count_nonzero(virtual))
        response = np.asarray(occupied_virtual_response)
        if response.shape[-2:] != (nvir, nocc):
            raise RKSResponseError("the RKS occupied-virtual response shape is invalid")
        full_response = np.zeros(
            (*response.shape[:-2], coefficient.shape[1], nocc),
            dtype=np.float64,
        )
        full_response[..., virtual, :] = response
        induced = self._induced_mo_potential(
            full_response,
            coefficient,
            occupation,
            occupied,
        )[..., virtual, :]
        gaps = energy[virtual, None] - energy[occupied]
        return gaps * response + induced

    def _response_operator_matrix_and_diagnostics(
        self,
        coefficient: np.ndarray,
        energy: np.ndarray,
        occupation: np.ndarray,
        occupied: np.ndarray,
        virtual: np.ndarray,
    ) -> tuple[np.ndarray, int, float, float, float, float, float]:
        from .audits.rks_operator import _response_operator_matrix_and_diagnostics as audit
        return audit(self, coefficient, energy, occupation, occupied, virtual)

    def validate_response_operator_exact(
        self,
    ) -> tuple[int, float, float, float, float, float]:
        from .audits.rks_operator import validate_response_operator_exact as audit
        return audit(self)

    def linear_response_problem(self) -> ScalarAdjointProblem:
        """Return the action-only RKS operator through the neutral protocol."""
        validate_rks_reference(self.reference)
        problem = _RKSLinearResponseProblem(self)
        if not isinstance(problem, ScalarAdjointProblem):
            raise RKSResponseError(
                "the RKS linear-response problem violates the neutral adjoint protocol"
            )
        return problem
