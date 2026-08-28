"""GPU-native DeePHF trajectory scanner with unrestricted root tracking."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
from pyscf import gto

from deepks.gpu import as_numpy

from .contracts import (
    RootContinuityError,
    occupied_coefficients,
    occupied_subspace_overlaps,
    validate_root_overlap_tolerance,
)
from .gpu_method import GPUDeePHF, gpu_reference_family
from .pyscf_rhf_reference import _molecule_static_fingerprint


SCF_CONTROL_NAMES = (
    "conv_tol",
    "conv_tol_grad",
    "conv_tol_cpscf",
    "max_cycle",
    "diis_space",
    "diis_start_cycle",
    "damp",
    "init_guess",
    "init_guess_breaksym",
    "level_shift",
    "direct_scf",
    "direct_scf_tol",
    "conv_check",
)


class GPUDeePHFScannerError(RuntimeError):
    """Raised when a GPU trajectory frame cannot be safely published."""


class GPUDeePHFGradientScanner:
    """Evaluate one continuous GPU DeePHF energy-and-gradient trajectory."""

    @classmethod
    def from_method(
        cls,
        method,
        *,
        backend="direct",
        backend_options=None,
        root_overlap_tolerance=0.5,
    ):
        if not isinstance(method, GPUDeePHF):
            raise TypeError("the GPU scanner requires a GPUDeePHF method")
        if backend != "direct":
            raise ValueError("the GPU scanner requires the direct backend")
        scf_args = getattr(method.reference, "_deepks_scf_args", None)
        if scf_args is None:
            scf_args = {
                name: getattr(method.reference, name)
                for name in SCF_CONTROL_NAMES
                if hasattr(method.reference, name)
            }
        return cls(
            method,
            scf_args=scf_args,
            backend=backend,
            backend_options=backend_options,
            root_overlap_tolerance=root_overlap_tolerance,
        )

    def __init__(
        self,
        method,
        *,
        scf_args=None,
        backend="direct",
        backend_options=None,
        root_overlap_tolerance=0.5,
    ):
        if not isinstance(method, GPUDeePHF):
            raise TypeError("the GPU scanner requires a GPUDeePHF method")
        if backend != "direct":
            raise ValueError("the GPU scanner requires the direct backend")
        self.family = method.reference_family
        self.model = method.model
        self.projector_basis = deepcopy(method._descriptor.projector_basis)
        self.device = method.device
        self.response_options = deepcopy(method.response_options)
        self.adjoint_options = deepcopy(method.adjoint_options)
        self.scf_args = {} if scf_args is None else dict(scf_args)
        self.dft_args = deepcopy(
            getattr(method.reference, "_deepks_dft_args", None)
        )
        self.backend = backend
        self.backend_options = dict(backend_options or {})
        self.root_overlap_tolerance = validate_root_overlap_tolerance(
            root_overlap_tolerance,
            owner="GPU DeePHF scanner",
        )
        self._system_fingerprint = _molecule_static_fingerprint(method.mol)
        self._initial_method = method
        self._anchor_reference = method.reference
        self._anchor_density = method.reference.make_rdm1().copy()
        self._anchor_occupations = np.array(
            as_numpy(method.reference.mo_occ), copy=True
        )
        self._anchor_occupied = occupied_coefficients(
            as_numpy(method.reference.mo_coeff),
            self._anchor_occupations,
        )
        self._anchor_state_fingerprint = self._state_fingerprint(method.reference)
        self.records = []
        self._clear_public_result()

    @staticmethod
    def _state_fingerprint(reference) -> str:
        from .workflow import _reference_state_fingerprint

        return _reference_state_fingerprint(reference)

    def _clear_public_result(self) -> None:
        self.mol = None
        self.reference = None
        self.method = None
        self.gradient_driver = None
        self.e_tot = None
        self.de = None
        self.converged = False

    def _molecule(self, mol_or_coordinates):
        if isinstance(mol_or_coordinates, gto.Mole):
            molecule = mol_or_coordinates
        else:
            molecule = self._anchor_reference.mol.set_geom_(
                mol_or_coordinates,
                inplace=False,
            )
        if _molecule_static_fingerprint(molecule) != self._system_fingerprint:
            raise RootContinuityError(
                "the molecular system or AO basis changed within the GPU trajectory"
            )
        return molecule

    def _is_initial_geometry(self, molecule) -> bool:
        return self._initial_method is not None and np.array_equal(
            molecule.atom_coords(unit="Bohr"),
            self._initial_method.mol.atom_coords(unit="Bohr"),
        )

    def _candidate_reference(self, molecule):
        if self._is_initial_geometry(molecule):
            return self._initial_method.reference, "existing_reference"
        from .workflow import build_reference

        return (
            build_reference(
                molecule,
                self.family,
                scf_args=self.scf_args,
                dft_args=deepcopy(self.dft_args),
                dm0=self._anchor_density,
                verbose=getattr(self._anchor_reference, "verbose", 0),
            ),
            "previous_density",
        )

    def _root_evidence(self, reference, initial_guess_source):
        occupations = np.array(as_numpy(reference.mo_occ), copy=True)
        candidate_occupied = occupied_coefficients(
            as_numpy(reference.mo_coeff),
            occupations,
        )
        if not np.array_equal(occupations, self._anchor_occupations):
            raise RootContinuityError(
                f"the {self.family.upper()} occupations changed from the accepted root"
            )
        if initial_guess_source == "existing_reference":
            overlaps = tuple(1.0 for _ in candidate_occupied)
        else:
            overlaps = occupied_subspace_overlaps(
                self._anchor_reference.mol,
                self._anchor_occupied,
                reference.mol,
                candidate_occupied,
            )
        minimum_overlap = float(min(overlaps))
        if minimum_overlap < self.root_overlap_tolerance:
            raise RootContinuityError(
                f"the {self.family.upper()} occupied subspace is discontinuous: "
                f"minimum overlap {minimum_overlap:.6f} < "
                f"{self.root_overlap_tolerance:.6f}"
            )
        return occupations, candidate_occupied, overlaps, minimum_overlap

    def _candidate_method(self, reference, initial_guess_source):
        if initial_guess_source == "existing_reference":
            return self._initial_method
        from .workflow import make_deephf

        return make_deephf(
            reference,
            self.model,
            projector_basis=deepcopy(self.projector_basis),
            device=self.device,
            response_options=deepcopy(self.response_options),
            adjoint_options=deepcopy(self.adjoint_options),
        )

    def _accept_reference(
        self,
        reference,
        method,
        occupations,
        candidate_occupied,
        state_fingerprint,
    ) -> None:
        self._anchor_reference = reference
        self._anchor_density = reference.make_rdm1().copy()
        self._anchor_occupations = occupations
        self._anchor_occupied = tuple(
            value.copy() for value in candidate_occupied
        )
        self._anchor_state_fingerprint = state_fingerprint
        self._initial_method = None
        self.mol = reference.mol
        self.reference = reference
        self.method = method
        self.converged = True

    def __call__(self, mol_or_coordinates, *, atmlst=None):
        """Return one atomically accepted DeePHF energy and gradient frame."""
        self._clear_public_result()
        molecule = self._molecule(mol_or_coordinates)
        reference, initial_guess_source = self._candidate_reference(molecule)
        if gpu_reference_family(reference) != self.family:
            raise GPUDeePHFScannerError(
                "the candidate reference changed GPU4PySCF family"
            )
        (
            occupations,
            candidate_occupied,
            overlaps,
            minimum_overlap,
        ) = self._root_evidence(reference, initial_guess_source)
        method = self._candidate_method(reference, initial_guess_source)
        energy = float(method.kernel())
        gradient_driver = method.nuc_grad_method(
            backend=self.backend,
            retain_details=False,
            **deepcopy(self.backend_options),
        )
        gradient = np.asarray(
            gradient_driver.kernel(atmlst=atmlst), dtype=np.float64
        )
        expected_atoms = molecule.natm if atmlst is None else len(tuple(atmlst))
        if (
            not np.isfinite(energy)
            or gradient.shape != (expected_atoms, 3)
            or not np.isfinite(gradient).all()
        ):
            raise GPUDeePHFScannerError(
                "the candidate DeePHF energy or gradient is invalid"
            )
        state_fingerprint = self._state_fingerprint(reference)
        channel_names = (
            ("restricted",)
            if self.family in {"rhf", "rks"}
            else ("alpha", "beta")
        )
        record = {
            "frame_index": len(self.records),
            "reference_state_fingerprint": state_fingerprint,
            "parent_reference_state_fingerprint": (
                None if not self.records else self._anchor_state_fingerprint
            ),
            "initial_guess_source": initial_guess_source,
            "occupied_subspace_overlaps": {
                name: float(overlap)
                for name, overlap in zip(channel_names, overlaps, strict=True)
            },
            "minimum_occupied_overlap": minimum_overlap,
        }
        self._accept_reference(
            reference,
            method,
            occupations,
            candidate_occupied,
            state_fingerprint,
        )
        self.records.append(record)
        self.gradient_driver = gradient_driver
        self.e_tot = energy
        self.de = gradient.copy()
        return energy, gradient.copy()


class GPUDeePHFFiniteDifferenceScanner(GPUDeePHFGradientScanner):
    """Evaluate DeePHF forces by central differences of the total energy."""

    @classmethod
    def from_method(
        cls,
        method,
        *,
        finite_difference_step_bohr=1.0e-4,
        root_overlap_tolerance=0.5,
    ):
        if not isinstance(method, GPUDeePHF):
            raise TypeError("the GPU scanner requires a GPUDeePHF method")
        scf_args = getattr(method.reference, "_deepks_scf_args", None)
        if scf_args is None:
            scf_args = {
                name: getattr(method.reference, name)
                for name in SCF_CONTROL_NAMES
                if hasattr(method.reference, name)
            }
        return cls(
            method,
            scf_args=scf_args,
            finite_difference_step_bohr=finite_difference_step_bohr,
            root_overlap_tolerance=root_overlap_tolerance,
        )

    def __init__(
        self,
        method,
        *,
        scf_args=None,
        finite_difference_step_bohr=1.0e-4,
        root_overlap_tolerance=0.5,
    ):
        super().__init__(
            method,
            scf_args=scf_args,
            backend="direct",
            root_overlap_tolerance=root_overlap_tolerance,
        )
        step = float(finite_difference_step_bohr)
        if step <= 0.0 or not np.isfinite(step):
            raise ValueError(
                "finite-difference step in Bohr must be finite and positive"
            )
        self.finite_difference_step_bohr = step

    def _displaced_energy(
        self,
        molecule,
        *,
        central_reference,
        central_density,
        central_occupations,
        central_occupied,
    ) -> tuple[float, float]:
        from .workflow import build_reference, make_deephf

        reference = build_reference(
            molecule,
            self.family,
            scf_args=self.scf_args,
            dft_args=deepcopy(self.dft_args),
            dm0=central_density,
            verbose=getattr(central_reference, "verbose", 0),
        )
        if gpu_reference_family(reference) != self.family:
            raise GPUDeePHFScannerError(
                "a finite-difference reference changed GPU4PySCF family"
            )
        occupations = np.array(as_numpy(reference.mo_occ), copy=True)
        if not np.array_equal(occupations, central_occupations):
            raise RootContinuityError(
                f"a finite-difference {self.family.upper()} occupation changed"
            )
        occupied = occupied_coefficients(
            as_numpy(reference.mo_coeff),
            occupations,
        )
        overlaps = occupied_subspace_overlaps(
            central_reference.mol,
            central_occupied,
            reference.mol,
            occupied,
        )
        minimum_overlap = float(min(overlaps))
        if minimum_overlap < self.root_overlap_tolerance:
            raise RootContinuityError(
                f"a finite-difference {self.family.upper()} occupied subspace "
                f"is discontinuous: minimum overlap {minimum_overlap:.6f} < "
                f"{self.root_overlap_tolerance:.6f}"
            )
        method = make_deephf(
            reference,
            self.model,
            projector_basis=deepcopy(self.projector_basis),
            device=self.device,
            response_options=deepcopy(self.response_options),
            adjoint_options=deepcopy(self.adjoint_options),
        )
        energy = float(method.kernel())
        if not np.isfinite(energy):
            raise GPUDeePHFScannerError(
                "a finite-difference DeePHF energy is nonfinite"
            )
        return energy, minimum_overlap

    def _gradient(
        self,
        molecule,
        reference,
        occupations,
        occupied,
    ) -> tuple[np.ndarray, float]:
        coordinates = np.asarray(
            molecule.atom_coords(unit="Bohr"), dtype=np.float64
        )
        density = reference.make_rdm1().copy()
        gradient = np.empty_like(coordinates)
        minimum_displaced_overlap = 1.0
        for atom_index, coordinate_index in np.ndindex(coordinates.shape):
            energies = []
            for direction in (1.0, -1.0):
                displaced_coordinates = coordinates.copy()
                displaced_coordinates[atom_index, coordinate_index] += (
                    direction * self.finite_difference_step_bohr
                )
                displaced_molecule = molecule.set_geom_(
                    displaced_coordinates,
                    unit="Bohr",
                    inplace=False,
                )
                energy, overlap = self._displaced_energy(
                    displaced_molecule,
                    central_reference=reference,
                    central_density=density,
                    central_occupations=occupations,
                    central_occupied=occupied,
                )
                energies.append(energy)
                minimum_displaced_overlap = min(
                    minimum_displaced_overlap, overlap
                )
            gradient[atom_index, coordinate_index] = (
                energies[0] - energies[1]
            ) / (2.0 * self.finite_difference_step_bohr)
        if not np.isfinite(gradient).all():
            raise GPUDeePHFScannerError(
                "the finite-difference DeePHF gradient is nonfinite"
            )
        return gradient, minimum_displaced_overlap

    def __call__(self, mol_or_coordinates, *, atmlst=None):
        """Return one total-energy central-difference gradient frame."""
        if atmlst is not None:
            raise ValueError(
                "finite-difference DeePHF trajectories require all atoms"
            )
        self._clear_public_result()
        molecule = self._molecule(mol_or_coordinates)
        reference, initial_guess_source = self._candidate_reference(molecule)
        if gpu_reference_family(reference) != self.family:
            raise GPUDeePHFScannerError(
                "the candidate reference changed GPU4PySCF family"
            )
        (
            occupations,
            candidate_occupied,
            overlaps,
            minimum_overlap,
        ) = self._root_evidence(reference, initial_guess_source)
        method = self._candidate_method(reference, initial_guess_source)
        energy = float(method.kernel())
        gradient, minimum_displaced_overlap = self._gradient(
            molecule,
            reference,
            occupations,
            candidate_occupied,
        )
        if not np.isfinite(energy):
            raise GPUDeePHFScannerError("the candidate DeePHF energy is nonfinite")
        state_fingerprint = self._state_fingerprint(reference)
        channel_names = (
            ("restricted",)
            if self.family in {"rhf", "rks"}
            else ("alpha", "beta")
        )
        record = {
            "frame_index": len(self.records),
            "reference_state_fingerprint": state_fingerprint,
            "parent_reference_state_fingerprint": (
                None if not self.records else self._anchor_state_fingerprint
            ),
            "initial_guess_source": initial_guess_source,
            "occupied_subspace_overlaps": {
                name: float(overlap)
                for name, overlap in zip(channel_names, overlaps, strict=True)
            },
            "minimum_occupied_overlap": minimum_overlap,
            "force_mode": "central_finite_difference",
            "finite_difference_step_bohr": self.finite_difference_step_bohr,
            "energy_evaluations": 1 + 6 * molecule.natm,
            "minimum_displaced_occupied_overlap": minimum_displaced_overlap,
        }
        self._accept_reference(
            reference,
            method,
            occupations,
            candidate_occupied,
            state_fingerprint,
        )
        self.records.append(record)
        self.e_tot = energy
        self.de = gradient.copy()
        return energy, gradient.copy()


GPUUHFDeePHFGradientScanner = GPUDeePHFGradientScanner


__all__ = [
    "GPUDeePHFFiniteDifferenceScanner",
    "GPUDeePHFGradientScanner",
    "GPUDeePHFScannerError",
    "GPUUHFDeePHFGradientScanner",
]
