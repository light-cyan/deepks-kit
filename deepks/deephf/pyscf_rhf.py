"""Isolated PySCF 2.14 adapter for molecular RHF nuclear response."""

from dataclasses import dataclass, fields, replace
import hashlib
import operator

import numpy as np
import pyscf
from pyscf.hessian import rhf as rhf_hessian
from pyscf.scf import cphf

from .capabilities import DeePHFCapabilityError, validate_reference


SUPPORTED_PYSCF_SERIES = (2, 14)


class RHFResponseError(RuntimeError):
    """Raised when the RHF response equations fail the strict contract."""


@dataclass(frozen=True)
class RHFResponseDiagnostics:
    """Independent diagnostics for one complete nuclear CPHF solve."""

    minimum_orbital_gap: float
    pyscf_version: str
    cphf_tolerance: float
    maximum_residual: float
    residual_rms: float
    residual_tolerance: float
    invariant_tolerance: float
    orbital_gap_tolerance: float
    max_cycle: int
    max_refinement_cycles: int
    level_shift: float
    response_dimension: int
    operator_stability_tolerance: float
    operator_condition_tolerance: float
    operator_symmetry_tolerance: float
    operator_dimension_limit: int
    operator_minimum_eigenvalue: float
    operator_maximum_eigenvalue: float
    operator_condition_number: float
    operator_symmetry_residual: float
    metric_residual: float
    idempotency_residual: float
    particle_number_residual: float
    refinement_cycles: int
    residual_history: tuple[float, ...]


@dataclass(frozen=True)
class RHFResponse:
    """Complete first-order RHF state for all nuclear coordinates."""

    reference_identity: int
    state_fingerprint: str
    integrity_fingerprint: str
    mo_response: np.ndarray
    mo_response_occupied_virtual: np.ndarray
    mo_response_metric: np.ndarray
    coefficient_response: np.ndarray
    coefficient_response_occupied_virtual: np.ndarray
    coefficient_response_metric: np.ndarray
    density_response: np.ndarray
    density_response_occupied_virtual: np.ndarray
    density_response_metric: np.ndarray
    overlap_derivative: np.ndarray
    hamiltonian_derivative: np.ndarray
    orbital_response_residual: np.ndarray
    diagnostics: RHFResponseDiagnostics


def _version_series(version: str) -> tuple[int, int]:
    components = version.split(".")
    try:
        return int(components[0]), int(components[1])
    except (IndexError, ValueError) as error:
        raise DeePHFCapabilityError(
            f"cannot interpret the PySCF version {version!r}"
        ) from error


def validate_pyscf_version() -> None:
    """Require the PySCF series characterized by the direct oracle."""
    series = _version_series(pyscf.__version__)
    if series != SUPPORTED_PYSCF_SERIES:
        raise DeePHFCapabilityError(
            "the RHF direct-response adapter supports PySCF 2.14; "
            f"found {pyscf.__version__}"
        )


def reference_fingerprint(reference) -> str:
    """Return a deterministic fingerprint of the response-defining RHF state."""
    digest = hashlib.sha256()
    digest.update(pyscf.__version__.encode("utf-8"))
    molecule = reference.mol
    scalar_state = (
        molecule.charge,
        molecule.spin,
        molecule.cart,
        molecule.symmetry,
        reference.e_tot,
    )
    digest.update(repr(scalar_state).encode("utf-8"))
    arrays = (
        molecule._atm,
        molecule._bas,
        molecule._env,
        molecule.atom_coords(unit="Bohr"),
        molecule.atom_charges(),
        reference.mo_coeff,
        reference.mo_energy,
        reference.mo_occ,
        reference.make_rdm1(),
    )
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(repr(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _immutable_array(value: np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(value)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def response_integrity_fingerprint(response: RHFResponse) -> str:
    """Return a digest covering every response field except the digest itself."""
    digest = hashlib.sha256()
    for field in fields(response):
        if field.name == "integrity_fingerprint":
            continue
        value = getattr(response, field.name)
        digest.update(field.name.encode("utf-8"))
        if isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            digest.update(array.dtype.str.encode("ascii"))
            digest.update(repr(array.shape).encode("ascii"))
            digest.update(array.tobytes())
        else:
            digest.update(repr(value).encode("utf-8"))
    return digest.hexdigest()


def _cycle_limit(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"response {name} must be an integer")
    try:
        return operator.index(value)
    except TypeError as error:
        raise ValueError(f"response {name} must be an integer") from error


def _validated_float64_array(value, expected_shape, name: str) -> np.ndarray:
    try:
        array = np.asarray(value)
    except Exception as error:
        raise RHFResponseError(f"{name} is not a numerical array: {error}") from error
    if array.shape != expected_shape:
        raise RHFResponseError(
            f"unexpected {name} shape {array.shape}; expected {expected_shape}"
        )
    if array.dtype != np.dtype(np.float64) or np.iscomplexobj(array):
        raise RHFResponseError(f"{name} must be a real float64 array")
    if not np.isfinite(array).all():
        raise RHFResponseError(f"{name} must be finite")
    return array


class RHFResponseAdapter:
    """Solve and audit molecular RHF nuclear CPHF through PySCF 2.14."""

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
        validate_pyscf_version()
        self.reference = validate_reference(reference)
        self.cphf_tolerance = float(cphf_tolerance)
        self.residual_tolerance = float(residual_tolerance)
        self.invariant_tolerance = float(invariant_tolerance)
        self.orbital_gap_tolerance = float(orbital_gap_tolerance)
        self.max_cycle = _cycle_limit(max_cycle, "max_cycle")
        self.max_refinement_cycles = _cycle_limit(
            max_refinement_cycles,
            "max_refinement_cycles",
        )
        self.level_shift = float(level_shift)
        self.operator_stability_tolerance = float(
            operator_stability_tolerance
        )
        self.operator_condition_tolerance = float(
            operator_condition_tolerance
        )
        self.operator_symmetry_tolerance = float(
            operator_symmetry_tolerance
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
        if not np.isfinite(tolerance_values).all():
            raise ValueError("response tolerances must be finite")
        if not np.isfinite(self.level_shift):
            raise ValueError("response level_shift must be finite")
        if self.cphf_tolerance <= 0 or self.residual_tolerance <= 0:
            raise ValueError("response tolerances must be positive")
        if self.invariant_tolerance <= 0 or self.orbital_gap_tolerance <= 0:
            raise ValueError("response tolerances must be positive")
        if (
            self.operator_stability_tolerance <= 0
            or self.operator_condition_tolerance <= 1
            or self.operator_symmetry_tolerance <= 0
        ):
            raise ValueError("response operator tolerances are invalid")
        if self.max_cycle <= 0 or self.max_refinement_cycles < 0:
            raise ValueError("response cycle limits are invalid")
        if self.operator_dimension_limit <= 0:
            raise ValueError("response operator_dimension_limit must be positive")

    @property
    def molecule(self):
        return self.reference.mol

    def _state(self):
        coefficient = np.asarray(self.reference.mo_coeff)
        energy = np.asarray(self.reference.mo_energy)
        occupation = np.asarray(self.reference.mo_occ)
        occupied = occupation > 0
        virtual = occupation == 0
        if not np.any(occupied) or not np.any(virtual):
            raise DeePHFCapabilityError(
                "RHF response requires occupied and virtual orbitals"
            )
        gaps = energy[virtual, None] - energy[occupied]
        minimum_gap = float(np.min(gaps))
        if not np.isfinite(minimum_gap) or minimum_gap <= self.orbital_gap_tolerance:
            raise DeePHFCapabilityError(
                "RHF occupied-virtual gap is outside the strict response domain: "
                f"{minimum_gap:.3e} <= {self.orbital_gap_tolerance:.3e}"
            )
        return coefficient, energy, occupation, occupied, virtual, minimum_gap

    def _overlap_derivative(self) -> np.ndarray:
        molecule = self.molecule
        nao = molecule.nao
        try:
            integral = -molecule.intor("int1e_ipovlp", comp=3)
        except Exception as error:
            raise RHFResponseError(
                f"PySCF overlap-derivative construction failed: {error}"
            ) from error
        integral = _validated_float64_array(
            integral,
            (3, nao, nao),
            "overlap-derivative integral",
        )
        result = np.zeros((molecule.natm, 3, nao, nao))
        for atom_index, atom_slice in enumerate(molecule.aoslice_by_atom()):
            ao_start, ao_stop = atom_slice[2:]
            result[atom_index, :, ao_start:ao_stop] += integral[
                :, ao_start:ao_stop
            ]
            result[atom_index, :, :, ao_start:ao_stop] += integral[
                :, ao_start:ao_stop
            ].transpose(0, 2, 1)
        return result

    def _hamiltonian_derivative(
        self,
        coefficient: np.ndarray,
        occupation: np.ndarray,
    ) -> np.ndarray:
        try:
            hessian = rhf_hessian.Hessian(self.reference)
            derivatives = hessian.make_h1(
                coefficient,
                occupation,
                atmlst=range(self.molecule.natm),
            )
        except Exception as error:
            raise RHFResponseError(
                f"PySCF RHF Hamiltonian derivative construction failed: {error}"
            ) from error
        if derivatives is None:
            raise RHFResponseError(
                "PySCF RHF Hamiltonian derivative is incomplete"
            )
        expected = (self.molecule.natm, 3, self.molecule.nao, self.molecule.nao)
        return _validated_float64_array(
            derivatives,
            expected,
            "Hamiltonian derivative",
        )

    def _density_from_mo_response(
        self,
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
        density_response = np.einsum(
            "...pi,qi,i->...pq",
            coefficient_response,
            occupied_coefficients,
            occupation[occupied],
        )
        return density_response + density_response.swapaxes(-1, -2)

    def _induced_potential(self, density_response: np.ndarray) -> np.ndarray:
        flat_density = np.asarray(density_response).reshape(
            -1,
            self.molecule.nao,
            self.molecule.nao,
        )
        try:
            coulomb, exchange = self.reference.get_jk(
                self.molecule,
                flat_density,
                hermi=1,
            )
            coulomb = _validated_float64_array(
                coulomb,
                flat_density.shape,
                "induced Coulomb response",
            )
            exchange = _validated_float64_array(
                exchange,
                flat_density.shape,
                "induced exchange response",
            )
        except RHFResponseError:
            raise
        except Exception as error:
            raise RHFResponseError(
                f"PySCF induced-potential construction failed: {error}"
            ) from error
        return (coulomb - 0.5 * exchange).reshape(density_response.shape)

    def _response_operator_diagnostics(
        self,
        coefficient: np.ndarray,
        energy: np.ndarray,
        occupation: np.ndarray,
        occupied: np.ndarray,
        virtual: np.ndarray,
    ) -> tuple[int, float, float, float, float]:
        """Build and audit the unshifted physical occupied-virtual operator."""
        nocc = int(np.count_nonzero(occupied))
        nvir = int(np.count_nonzero(virtual))
        dimension = nocc * nvir
        if dimension > self.operator_dimension_limit:
            raise DeePHFCapabilityError(
                "RHF occupied-virtual response dimension exceeds the explicit "
                f"condition-audit limit: {dimension} > {self.operator_dimension_limit}"
            )
        identity = np.eye(dimension, dtype=np.float64)
        matrix = np.empty((dimension, dimension), dtype=np.float64)
        orbital_gap = energy[virtual, None] - energy[occupied]
        batch_size = min(64, dimension)
        for start in range(0, dimension, batch_size):
            stop = min(start + batch_size, dimension)
            roots = identity[start:stop].reshape(-1, nvir, nocc)
            full_response = np.zeros(
                (stop - start, coefficient.shape[1], nocc),
                dtype=np.float64,
            )
            full_response[:, virtual] = roots
            induced = self._induced_mo_potential(
                full_response,
                coefficient,
                occupation,
                occupied,
            )[:, virtual]
            images = orbital_gap * roots + induced
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
            dimension,
            minimum_eigenvalue,
            maximum_eigenvalue,
            float(condition_number),
            symmetry_residual,
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

    def _orbital_residual(
        self,
        mo_response: np.ndarray,
        hamiltonian_mo: np.ndarray,
        overlap_mo: np.ndarray,
        coefficient: np.ndarray,
        energy: np.ndarray,
        occupation: np.ndarray,
        occupied: np.ndarray,
        virtual: np.ndarray,
    ) -> np.ndarray:
        induced_mo = self._induced_mo_potential(
            mo_response,
            coefficient,
            occupation,
            occupied,
        )
        residual = (
            hamiltonian_mo
            + induced_mo
            - overlap_mo * energy[occupied]
            + (energy[:, None] - energy[occupied]) * mo_response
        )
        return residual[..., virtual, :]

    def audit_response_equations(self, response: RHFResponse) -> None:
        """Rebuild derivative inputs, equations, and invariants for a supplied response."""
        validate_reference(self.reference)
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
        (
            response_dimension,
            operator_minimum_eigenvalue,
            operator_maximum_eigenvalue,
            operator_condition_number,
            operator_symmetry_residual,
        ) = self._response_operator_diagnostics(
            coefficient,
            energy,
            occupation,
            occupied,
            virtual,
        )
        expected_overlap_derivative = self._overlap_derivative()
        expected_hamiltonian_derivative = self._hamiltonian_derivative(
            coefficient,
            occupation,
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
            "operator_minimum_eigenvalue": operator_minimum_eigenvalue,
            "operator_maximum_eigenvalue": operator_maximum_eigenvalue,
            "operator_condition_number": operator_condition_number,
            "operator_symmetry_residual": operator_symmetry_residual,
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

    def _solve_orbitals(
        self,
        hamiltonian_mo: np.ndarray,
        overlap_mo: np.ndarray,
        coefficient: np.ndarray,
        energy: np.ndarray,
        occupation: np.ndarray,
        occupied: np.ndarray,
        virtual: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, tuple[float, ...]]:
        perturbation_shape = hamiltonian_mo.shape[:-2]
        nmo = coefficient.shape[1]
        nocc = int(np.count_nonzero(occupied))
        flattened_hamiltonian = hamiltonian_mo.reshape(-1, nmo, nocc)
        flattened_overlap = overlap_mo.reshape(-1, nmo, nocc)

        def induced_full(response):
            response = np.asarray(response).reshape(-1, nmo, nocc)
            return self._induced_mo_potential(
                response,
                coefficient,
                occupation,
                occupied,
            )

        try:
            response, _ = cphf.solve(
                induced_full,
                energy,
                occupation,
                flattened_hamiltonian,
                flattened_overlap,
                max_cycle=self.max_cycle,
                tol=self.cphf_tolerance,
                level_shift=self.level_shift,
                verbose=self.reference.verbose,
            )
        except Exception as error:
            raise RHFResponseError(f"PySCF RHF CPHF solve failed: {error}") from error
        response = _validated_float64_array(
            response,
            flattened_hamiltonian.shape,
            "PySCF RHF CPHF response",
        ).reshape(*perturbation_shape, nmo, nocc)
        residual = self._orbital_residual(
            response,
            hamiltonian_mo,
            overlap_mo,
            coefficient,
            energy,
            occupation,
            occupied,
            virtual,
        )
        residual_history = [float(np.max(np.abs(residual), initial=0.0))]
        while (
            residual_history[-1] > self.residual_tolerance
            and len(residual_history) - 1 < self.max_refinement_cycles
        ):
            flat_residual = residual.reshape(-1, int(np.count_nonzero(virtual)), nocc)
            root_scales = np.linalg.norm(flat_residual.reshape(len(flat_residual), -1), axis=1)
            active = root_scales > np.finfo(float).eps
            correction = np.zeros_like(flat_residual)

            def induced_virtual(virtual_response):
                virtual_response = np.asarray(virtual_response).reshape(
                    -1,
                    int(np.count_nonzero(virtual)),
                    nocc,
                )
                full_response = np.zeros(
                    (len(virtual_response), nmo, nocc),
                    dtype=virtual_response.dtype,
                )
                full_response[:, virtual] = virtual_response
                return induced_full(full_response)[:, virtual]

            if np.any(active):
                try:
                    normalized_correction, _ = cphf.solve(
                        induced_virtual,
                        energy,
                        occupation,
                        flat_residual[active] / root_scales[active, None, None],
                        s1=None,
                        max_cycle=self.max_cycle,
                        tol=self.cphf_tolerance,
                        level_shift=self.level_shift,
                        verbose=self.reference.verbose,
                    )
                except Exception as error:
                    raise RHFResponseError(
                        f"PySCF RHF CPHF residual refinement failed: {error}"
                    ) from error
                normalized_correction = _validated_float64_array(
                    normalized_correction,
                    flat_residual[active].shape,
                    "PySCF RHF CPHF refinement response",
                )
                correction[active] = (
                    normalized_correction * root_scales[active, None, None]
                )
                response[..., virtual, :] += correction.reshape(
                    *perturbation_shape,
                    int(np.count_nonzero(virtual)),
                    nocc,
                )
            residual = self._orbital_residual(
                response,
                hamiltonian_mo,
                overlap_mo,
                coefficient,
                energy,
                occupation,
                occupied,
                virtual,
            )
            residual_history.append(
                float(np.max(np.abs(residual), initial=0.0))
            )
        return response, residual, tuple(residual_history)

    def solve(self) -> RHFResponse:
        """Return the audited complete first-order AO density response."""
        validate_reference(self.reference)
        (
            coefficient,
            energy,
            occupation,
            occupied,
            virtual,
            minimum_gap,
        ) = self._state()
        (
            response_dimension,
            operator_minimum_eigenvalue,
            operator_maximum_eigenvalue,
            operator_condition_number,
            operator_symmetry_residual,
        ) = self._response_operator_diagnostics(
            coefficient,
            energy,
            occupation,
            occupied,
            virtual,
        )
        overlap = np.asarray(self.reference.get_ovlp())
        overlap_derivative = self._overlap_derivative()
        hamiltonian_derivative = self._hamiltonian_derivative(
            coefficient,
            occupation,
        )
        occupied_coefficients = coefficient[:, occupied]
        hamiltonian_mo = np.einsum(
            "mp,...mn,ni->...pi",
            coefficient,
            hamiltonian_derivative,
            occupied_coefficients,
        )
        overlap_mo = np.einsum(
            "mp,...mn,ni->...pi",
            coefficient,
            overlap_derivative,
            occupied_coefficients,
        )
        mo_response, residual, residual_history = self._solve_orbitals(
            hamiltonian_mo,
            overlap_mo,
            coefficient,
            energy,
            occupation,
            occupied,
            virtual,
        )
        metric_response = np.zeros_like(mo_response)
        metric_response[..., occupied, :] = mo_response[..., occupied, :]
        occupied_virtual_response = np.zeros_like(mo_response)
        occupied_virtual_response[..., virtual, :] = mo_response[..., virtual, :]
        coefficient_response = np.einsum(
            "mp,...pi->...mi",
            coefficient,
            mo_response,
        )
        coefficient_response_metric = np.einsum(
            "mp,...pi->...mi",
            coefficient,
            metric_response,
        )
        coefficient_response_occupied_virtual = np.einsum(
            "mp,...pi->...mi",
            coefficient,
            occupied_virtual_response,
        )
        density_metric = self._density_from_mo_response(
            metric_response,
            coefficient,
            occupation,
            occupied,
        )
        density_occupied_virtual = self._density_from_mo_response(
            occupied_virtual_response,
            coefficient,
            occupation,
            occupied,
        )
        density_response = density_metric + density_occupied_virtual
        density_ground = self.reference.make_rdm1()
        overlap_occupied = overlap_mo[..., occupied, :]
        metric_residual = np.max(
            np.abs(
                mo_response[..., occupied, :]
                + mo_response[..., occupied, :].swapaxes(-1, -2)
                + overlap_occupied
            ),
            initial=0.0,
        )
        idempotency = (
            np.einsum("...ij,jk,kl->...il", density_response, overlap, density_ground)
            + np.einsum(
                "ij,...jk,kl->...il",
                density_ground,
                overlap_derivative,
                density_ground,
            )
            + np.einsum("ij,jk,...kl->...il", density_ground, overlap, density_response)
            - 2.0 * density_response
        )
        particle_number = (
            np.einsum("...ij,ji->...", density_response, overlap)
            + np.einsum("ij,...ji->...", density_ground, overlap_derivative)
        )
        diagnostics = RHFResponseDiagnostics(
            minimum_orbital_gap=minimum_gap,
            pyscf_version=pyscf.__version__,
            cphf_tolerance=self.cphf_tolerance,
            maximum_residual=float(np.max(np.abs(residual), initial=0.0)),
            residual_rms=float(np.sqrt(np.mean(np.square(residual)))),
            residual_tolerance=self.residual_tolerance,
            invariant_tolerance=self.invariant_tolerance,
            orbital_gap_tolerance=self.orbital_gap_tolerance,
            max_cycle=self.max_cycle,
            max_refinement_cycles=self.max_refinement_cycles,
            level_shift=self.level_shift,
            response_dimension=response_dimension,
            operator_stability_tolerance=self.operator_stability_tolerance,
            operator_condition_tolerance=self.operator_condition_tolerance,
            operator_symmetry_tolerance=self.operator_symmetry_tolerance,
            operator_dimension_limit=self.operator_dimension_limit,
            operator_minimum_eigenvalue=operator_minimum_eigenvalue,
            operator_maximum_eigenvalue=operator_maximum_eigenvalue,
            operator_condition_number=operator_condition_number,
            operator_symmetry_residual=operator_symmetry_residual,
            metric_residual=float(metric_residual),
            idempotency_residual=float(np.max(np.abs(idempotency), initial=0.0)),
            particle_number_residual=float(
                np.max(np.abs(particle_number), initial=0.0)
            ),
            refinement_cycles=len(residual_history) - 1,
            residual_history=residual_history,
        )
        arrays = {
            "mo_response": mo_response,
            "mo_response_occupied_virtual": occupied_virtual_response,
            "mo_response_metric": metric_response,
            "coefficient_response": coefficient_response,
            "coefficient_response_occupied_virtual": (
                coefficient_response_occupied_virtual
            ),
            "coefficient_response_metric": coefficient_response_metric,
            "density_response": density_response,
            "density_response_occupied_virtual": density_occupied_virtual,
            "density_response_metric": density_metric,
            "overlap_derivative": overlap_derivative,
            "hamiltonian_derivative": hamiltonian_derivative,
            "orbital_response_residual": residual,
        }
        nonfinite = [name for name, value in arrays.items() if not np.isfinite(value).all()]
        if nonfinite:
            raise RHFResponseError(
                f"nonfinite RHF response quantities: {', '.join(nonfinite)}"
            )
        diagnostic_values = (
            diagnostics.minimum_orbital_gap,
            diagnostics.maximum_residual,
            diagnostics.residual_rms,
            diagnostics.metric_residual,
            diagnostics.idempotency_residual,
            diagnostics.particle_number_residual,
            diagnostics.operator_minimum_eigenvalue,
            diagnostics.operator_maximum_eigenvalue,
            diagnostics.operator_condition_number,
            diagnostics.operator_symmetry_residual,
            *diagnostics.residual_history,
        )
        if not np.isfinite(diagnostic_values).all():
            raise RHFResponseError("nonfinite RHF response diagnostics")
        if diagnostics.maximum_residual > self.residual_tolerance:
            history = " -> ".join(
                f"{value:.3e}" for value in diagnostics.residual_history
            )
            raise RHFResponseError(
                "RHF response residual exceeds tolerance: "
                f"{diagnostics.maximum_residual:.3e} > {self.residual_tolerance:.3e}; "
                f"refinement history: {history}"
            )
        invariant_failures = {
            "metric": diagnostics.metric_residual,
            "idempotency": diagnostics.idempotency_residual,
            "particle number": diagnostics.particle_number_residual,
        }
        invariant_failures = {
            name: value
            for name, value in invariant_failures.items()
            if value > self.invariant_tolerance
        }
        if invariant_failures:
            details = ", ".join(
                f"{name}={value:.3e}" for name, value in invariant_failures.items()
            )
            raise RHFResponseError(
                "RHF response invariant exceeds tolerance "
                f"{self.invariant_tolerance:.3e}: {details}"
            )
        response = RHFResponse(
            reference_identity=id(self.reference),
            state_fingerprint=reference_fingerprint(self.reference),
            integrity_fingerprint="",
            mo_response=_immutable_array(mo_response),
            mo_response_occupied_virtual=_immutable_array(
                occupied_virtual_response
            ),
            mo_response_metric=_immutable_array(metric_response),
            coefficient_response=_immutable_array(coefficient_response),
            coefficient_response_occupied_virtual=(
                _immutable_array(coefficient_response_occupied_virtual)
            ),
            coefficient_response_metric=_immutable_array(
                coefficient_response_metric
            ),
            density_response=_immutable_array(density_response),
            density_response_occupied_virtual=_immutable_array(
                density_occupied_virtual
            ),
            density_response_metric=_immutable_array(density_metric),
            overlap_derivative=_immutable_array(overlap_derivative),
            hamiltonian_derivative=_immutable_array(
                hamiltonian_derivative
            ),
            orbital_response_residual=_immutable_array(residual),
            diagnostics=diagnostics,
        )
        return replace(
            response,
            integrity_fingerprint=response_integrity_fingerprint(response),
        )
