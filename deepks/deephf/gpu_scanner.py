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
        if initial_guess_source == "existing_reference":
            method = self._initial_method
        else:
            from .workflow import make_deephf

            method = make_deephf(
                reference,
                self.model,
                projector_basis=deepcopy(self.projector_basis),
                device=self.device,
                response_options=deepcopy(self.response_options),
                adjoint_options=deepcopy(self.adjoint_options),
            )
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
        self._anchor_reference = reference
        self._anchor_density = reference.make_rdm1().copy()
        self._anchor_occupations = occupations
        self._anchor_occupied = tuple(value.copy() for value in candidate_occupied)
        self._anchor_state_fingerprint = state_fingerprint
        self._initial_method = None
        self.records.append(record)
        self.mol = reference.mol
        self.reference = reference
        self.method = method
        self.gradient_driver = gradient_driver
        self.e_tot = energy
        self.de = gradient.copy()
        self.converged = True
        return energy, gradient.copy()


GPUUHFDeePHFGradientScanner = GPUDeePHFGradientScanner


__all__ = [
    "GPUDeePHFGradientScanner",
    "GPUDeePHFScannerError",
    "GPUUHFDeePHFGradientScanner",
]
