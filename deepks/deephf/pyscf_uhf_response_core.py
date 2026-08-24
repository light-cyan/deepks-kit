"""Internal implementation extracted from pyscf_uhf.py."""

import numpy as np
from pyscf.hessian import uhf as uhf_hessian
from .capabilities import DeePHFCapabilityError
from .pyscf_uhf_reference import (
    UHFResponseError,
    _cycle_limit,
    _direct_effective_potential,
    _response_real_control,
    _validated_float64_array,
    uhf_reference_fingerprint,
    validate_pyscf_version,
    validate_uhf_reference,
)

class _UHFLinearResponseCore:
    """Share the strict coupled UHF operator and nuclear perturbation primitives."""

    @staticmethod
    def _validate_reference(reference):
        return validate_uhf_reference(reference)

    @staticmethod
    def _reference_fingerprint(reference) -> str:
        return uhf_reference_fingerprint(reference)

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
        validate_pyscf_version()
        self.reference = self._validate_reference(reference)
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
        tolerance_values = (
            self.cphf_tolerance,
            self.residual_tolerance,
            self.invariant_tolerance,
            self.orbital_gap_tolerance,
            self.operator_stability_tolerance,
            self.operator_condition_tolerance,
            self.operator_symmetry_tolerance,
        )
        if any(value <= 0 for value in tolerance_values[:5]):
            raise ValueError("response tolerances must be positive")
        if self.operator_condition_tolerance <= 1:
            raise ValueError("operator_condition_tolerance must exceed one")
        if self.operator_symmetry_tolerance <= 0:
            raise ValueError("operator_symmetry_tolerance must be positive")
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
        minimum_gaps = []
        for spin_index, spin_name in enumerate(("alpha", "beta")):
            gaps = (
                energy[spin_index, virtual[spin_index], None]
                - energy[spin_index, occupied[spin_index]]
            )
            minimum_gap = float(np.min(gaps))
            if (
                not np.isfinite(minimum_gap)
                or minimum_gap <= self.orbital_gap_tolerance
            ):
                raise DeePHFCapabilityError(
                    f"UHF {spin_name} occupied-virtual gap is outside the strict "
                    f"response domain: {minimum_gap:.3e} <= "
                    f"{self.orbital_gap_tolerance:.3e}"
                )
            minimum_gaps.append(minimum_gap)
        return coefficient, energy, occupation, occupied, virtual, tuple(minimum_gaps)

    def _overlap_derivative(self, atom_indices=None) -> np.ndarray:
        molecule = self.molecule
        atom_indices = self._response_atom_indices(atom_indices)
        try:
            integral = -molecule.intor("int1e_ipovlp", comp=3)
        except Exception as error:
            raise UHFResponseError(
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
            result[result_index, :, ao_start:ao_stop] += integral[
                :, ao_start:ao_stop
            ]
            result[result_index, :, :, ao_start:ao_stop] += integral[
                :, ao_start:ao_stop
            ].transpose(0, 2, 1)
        return result

    def _hamiltonian_derivative(
        self,
        coefficient: np.ndarray,
        occupation: np.ndarray,
        atom_indices=None,
    ) -> tuple[np.ndarray, np.ndarray]:
        atom_indices = self._response_atom_indices(atom_indices)
        try:
            derivatives = uhf_hessian.Hessian(self.reference).make_h1(
                coefficient,
                occupation,
                atmlst=atom_indices,
            )
        except Exception as error:
            raise UHFResponseError(
                f"PySCF UHF Hamiltonian derivative construction failed: {error}"
            ) from error
        if derivatives is None or len(derivatives) != 2:
            raise UHFResponseError("PySCF UHF Hamiltonian derivative is incomplete")
        expected = (len(atom_indices), 3, self.molecule.nao, self.molecule.nao)
        return tuple(
            _validated_float64_array(
                [spin_derivative[index] for index in atom_indices],
                expected,
                f"{spin_name} Hamiltonian derivative",
            )
            for spin_derivative, spin_name in zip(
                derivatives,
                ("alpha", "beta"),
                strict=True,
            )
        )

    @staticmethod
    def _density_from_mo_response(
        mo_response: np.ndarray,
        coefficient: np.ndarray,
        occupied: np.ndarray,
    ) -> np.ndarray:
        occupied_coefficient = coefficient[:, occupied]
        coefficient_response = np.einsum(
            "mp,...pi->...mi",
            coefficient,
            mo_response,
        )
        one_sided = np.einsum(
            "...pi,qi->...pq",
            coefficient_response,
            occupied_coefficient,
        )
        return one_sided + one_sided.swapaxes(-1, -2)

    def _induced_potential(
        self,
        alpha_density: np.ndarray,
        beta_density: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        perturbation_shape = alpha_density.shape[:-2]
        flat_alpha = np.asarray(alpha_density).reshape(
            -1,
            self.molecule.nao,
            self.molecule.nao,
        )
        flat_beta = np.asarray(beta_density).reshape(flat_alpha.shape)
        density = np.stack((flat_alpha, flat_beta), axis=0)
        try:
            potential = _direct_effective_potential(self.molecule, density)
        except Exception as error:
            if isinstance(error, UHFResponseError):
                raise
            raise UHFResponseError(
                f"PySCF induced UHF potential construction failed: {error}"
            ) from error
        potential = _validated_float64_array(
            potential,
            (2, *flat_alpha.shape),
            "induced UHF potential",
        )
        expected = (*perturbation_shape, self.molecule.nao, self.molecule.nao)
        return potential[0].reshape(expected), potential[1].reshape(expected)

    def _induced_mo_potential(
        self,
        alpha_response: np.ndarray,
        beta_response: np.ndarray,
        coefficient: np.ndarray,
        occupied: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        alpha_density = self._density_from_mo_response(
            alpha_response,
            coefficient[0],
            occupied[0],
        )
        beta_density = self._density_from_mo_response(
            beta_response,
            coefficient[1],
            occupied[1],
        )
        alpha_potential, beta_potential = self._induced_potential(
            alpha_density,
            beta_density,
        )
        alpha_mo = np.einsum(
            "mp,...mn,ni->...pi",
            coefficient[0],
            alpha_potential,
            coefficient[0][:, occupied[0]],
        )
        beta_mo = np.einsum(
            "mp,...mn,ni->...pi",
            coefficient[1],
            beta_potential,
            coefficient[1][:, occupied[1]],
        )
        return alpha_mo, beta_mo

    @staticmethod
    def _dimensions(occupied, virtual) -> tuple[int, int, int, int, int, int]:
        alpha_nocc = int(np.count_nonzero(occupied[0]))
        beta_nocc = int(np.count_nonzero(occupied[1]))
        alpha_nvir = int(np.count_nonzero(virtual[0]))
        beta_nvir = int(np.count_nonzero(virtual[1]))
        alpha_dimension = alpha_nocc * alpha_nvir
        beta_dimension = beta_nocc * beta_nvir
        return (
            alpha_nocc,
            beta_nocc,
            alpha_nvir,
            beta_nvir,
            alpha_dimension,
            beta_dimension,
        )

    def _split_occupied_virtual(self, vectors, occupied, virtual):
        (
            alpha_nocc,
            beta_nocc,
            alpha_nvir,
            beta_nvir,
            alpha_dimension,
            beta_dimension,
        ) = self._dimensions(occupied, virtual)
        vectors = np.asarray(vectors)
        expected_dimension = alpha_dimension + beta_dimension
        if vectors.shape[-1] != expected_dimension:
            raise UHFResponseError(
                "the coupled UHF occupied-virtual response has an invalid shape"
            )
        alpha = vectors[..., :alpha_dimension].reshape(
            *vectors.shape[:-1],
            alpha_nvir,
            alpha_nocc,
        )
        beta = vectors[..., alpha_dimension:].reshape(
            *vectors.shape[:-1],
            beta_nvir,
            beta_nocc,
        )
        return alpha, beta

    def _apply_occupied_virtual_operator(
        self,
        vectors: np.ndarray,
        coefficient: np.ndarray,
        energy: np.ndarray,
        occupied: np.ndarray,
        virtual: np.ndarray,
    ) -> np.ndarray:
        self._count_operation("response_operator_actions")
        alpha, beta = self._split_occupied_virtual(vectors, occupied, virtual)
        alpha_full = np.zeros(
            (*alpha.shape[:-2], coefficient.shape[2], alpha.shape[-1]),
            dtype=np.float64,
        )
        beta_full = np.zeros(
            (*beta.shape[:-2], coefficient.shape[2], beta.shape[-1]),
            dtype=np.float64,
        )
        alpha_full[..., virtual[0], :] = alpha
        beta_full[..., virtual[1], :] = beta
        induced_alpha, induced_beta = self._induced_mo_potential(
            alpha_full,
            beta_full,
            coefficient,
            occupied,
        )
        alpha_image = (
            (
                energy[0, virtual[0], None]
                - energy[0, occupied[0]]
            )
            * alpha
            + induced_alpha[..., virtual[0], :]
        )
        beta_image = (
            (
                energy[1, virtual[1], None]
                - energy[1, occupied[1]]
            )
            * beta
            + induced_beta[..., virtual[1], :]
        )
        return np.concatenate(
            (
                alpha_image.reshape(*alpha.shape[:-2], -1),
                beta_image.reshape(*beta.shape[:-2], -1),
            ),
            axis=-1,
        )

    def _response_operator_matrix_and_diagnostics(
        self,
        coefficient: np.ndarray,
        energy: np.ndarray,
        occupied: np.ndarray,
        virtual: np.ndarray,
    ) -> tuple[np.ndarray, int, int, int, float, float, float, float]:
        from .audits.uhf_operator import _response_operator_matrix_and_diagnostics as audit
        return audit(self, coefficient, energy, occupied, virtual)

    def validate_response_operator_exact(
        self,
    ) -> tuple[int, int, int, float, float, float, float]:
        from .audits.uhf_operator import validate_response_operator_exact as audit
        return audit(self)
