"""Internal implementation extracted from pyscf_rhf.py."""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
from typing import Any
import weakref
import numpy as np
from pyscf import gto
from pyscf.gto import mole as gto_mole
from pyscf.scf import hf as scf_hf
from .capabilities import DeePHFCapabilityError
from .pyscf_rhf_reference import (
    RHFAdjoint,
    RHFResponse,
    RHFResponseError,
    RHFRootSnapshot,
    RHFScannerReferenceError,
    _immutable_array,
    _molecule_static_fingerprint,
    _reject_molecule_instance_callables,
    _root_integrity_fingerprint,
    _scanner_scf_controls,
    molecule_science_fingerprint,
    reference_fingerprint,
    validate_pyscf_version,
    validate_reference,
)
from functools import partial

from .contracts import (
    dataclass_fingerprint,
    integer_control,
    real_control,
    validated_float64_array,
)

class RHFScannerReferenceFactory:
    """Build independent native RHF references and track one continuous root."""

    def __init__(self, reference, *, root_overlap_tolerance: float = 0.5):
        validate_pyscf_version()
        reference = validate_reference(reference)
        if isinstance(root_overlap_tolerance, (bool, np.bool_)):
            raise TypeError(
                "scanner root_overlap_tolerance must be a real number"
            )
        try:
            root_overlap_tolerance = float(root_overlap_tolerance)
        except (TypeError, ValueError) as error:
            raise TypeError(
                "scanner root_overlap_tolerance must be a real number"
            ) from error
        if (
            not np.isfinite(root_overlap_tolerance)
            or root_overlap_tolerance <= 0.0
            or root_overlap_tolerance > 1.0
        ):
            raise ValueError(
                "scanner root_overlap_tolerance must be finite and in (0, 1]"
            )
        try:
            template = deepcopy(reference.mol)
        except Exception as error:
            raise RHFScannerReferenceError(
                f"the scanner molecule template could not be copied: {error}"
            ) from error
        if type(template) is not gto.mole.Mole:
            raise RHFScannerReferenceError(
                "the scanner molecule copy is not an exact native pyscf.gto.Mole"
            )
        self.root_overlap_tolerance = root_overlap_tolerance
        self._template = template
        self._system_fingerprint = _molecule_static_fingerprint(template)
        self._atom_count = int(template.natm)
        self._ao_count = int(template.nao)
        self._occupations = _immutable_array(np.asarray(reference.mo_occ))
        self._scf_controls = _scanner_scf_controls(reference)
        self._issued_roots = {}
        self._initial_root = self._root_snapshot(
            reference,
            parent_state_fingerprint=None,
            minimum_occupied_overlap=1.0,
        )

    @property
    def initial_root(self) -> RHFRootSnapshot:
        return self._initial_root

    @property
    def system_fingerprint(self) -> str:
        return self._system_fingerprint

    @property
    def scf_controls(self) -> Mapping[str, Any]:
        return self._scf_controls

    def _root_snapshot(
        self,
        reference,
        *,
        parent_state_fingerprint: str | None,
        minimum_occupied_overlap: float,
    ) -> RHFRootSnapshot:
        occupations = np.asarray(reference.mo_occ)
        occupied = occupations > 0
        occupied_coefficients = np.asarray(reference.mo_coeff)[:, occupied]
        root = RHFRootSnapshot(
            system_fingerprint=self._system_fingerprint,
            state_fingerprint=reference_fingerprint(reference),
            integrity_fingerprint="",
            parent_state_fingerprint=parent_state_fingerprint,
            minimum_occupied_overlap=float(minimum_occupied_overlap),
            occupied_coefficients=_immutable_array(occupied_coefficients),
            occupations=_immutable_array(occupations),
            _molecule=deepcopy(reference.mol),
        )
        root = replace(
            root,
            integrity_fingerprint=_root_integrity_fingerprint(root),
        )
        self._register_root(root)
        return root

    def _register_root(self, root: RHFRootSnapshot) -> None:
        """Record one factory-issued root without retaining it indefinitely."""
        identity = id(root)
        factory_reference = weakref.ref(self)

        def discard(reference, *, identity=identity, factory_reference=factory_reference):
            factory = factory_reference()
            if factory is None:
                return
            issued = factory._issued_roots.get(identity)
            if issued is not None and issued[0] is reference:
                factory._issued_roots.pop(identity, None)

        root_reference = weakref.ref(root, discard)
        self._issued_roots[identity] = (
            root_reference,
            root.integrity_fingerprint,
            root.state_fingerprint,
            root.parent_state_fingerprint,
            root.minimum_occupied_overlap,
        )

    def _validate_root(self, root) -> RHFRootSnapshot:
        if type(root) is not RHFRootSnapshot:
            raise TypeError("scanner previous_root must be an RHFRootSnapshot")
        issued = self._issued_roots.get(id(root))
        if issued is None or issued[0]() is not root:
            raise RHFScannerReferenceError(
                "scanner previous_root was not issued by this reference factory"
            )
        if (
            root.integrity_fingerprint != issued[1]
            or root.state_fingerprint != issued[2]
            or root.parent_state_fingerprint != issued[3]
            or root.minimum_occupied_overlap != issued[4]
        ):
            raise RHFScannerReferenceError(
                "scanner previous_root changed after it was issued"
            )
        if root.system_fingerprint != self._system_fingerprint:
            raise RHFScannerReferenceError(
                "scanner previous_root belongs to another molecular system"
            )
        if root.integrity_fingerprint != _root_integrity_fingerprint(root):
            raise RHFScannerReferenceError(
                "scanner previous_root failed its integrity check"
            )
        if type(root._molecule) is not gto.mole.Mole:
            raise RHFScannerReferenceError(
                "scanner previous_root has an invalid molecule type"
            )
        if _molecule_static_fingerprint(root._molecule) != self._system_fingerprint:
            raise RHFScannerReferenceError(
                "scanner previous_root molecule has incompatible static metadata"
            )
        expected_occupied = int(np.count_nonzero(self._occupations > 0))
        array_fields = (
            (
                root.occupied_coefficients,
                (self._ao_count, expected_occupied),
                "occupied coefficients",
            ),
            (root.occupations, (self._ao_count,), "occupations"),
        )
        for value, shape, name in array_fields:
            if (
                not isinstance(value, np.ndarray)
                or value.shape != shape
                or value.dtype != np.dtype(np.float64)
                or np.iscomplexobj(value)
                or not np.isfinite(value).all()
                or value.flags.writeable
            ):
                raise RHFScannerReferenceError(
                    f"scanner previous_root {name} are invalid"
                )
        if not np.array_equal(root.occupations, self._occupations):
            raise RHFScannerReferenceError(
                "scanner previous_root occupations changed"
            )
        coordinates = np.asarray(root._molecule.atom_coords(unit="Bohr"))
        if (
            coordinates.shape != (self._atom_count, 3)
            or coordinates.dtype != np.dtype(np.float64)
            or not np.isfinite(coordinates).all()
        ):
            raise RHFScannerReferenceError(
                "scanner previous_root geometry is invalid"
            )
        if (
            not np.isfinite(root.minimum_occupied_overlap)
            or root.minimum_occupied_overlap < 0.0
            or root.minimum_occupied_overlap > 1.0 + 1.0e-10
        ):
            raise RHFScannerReferenceError(
                "scanner previous_root overlap diagnostic is invalid"
            )
        fingerprints = (
            root.system_fingerprint,
            root.state_fingerprint,
            root.integrity_fingerprint,
        )
        if root.parent_state_fingerprint is not None:
            fingerprints += (root.parent_state_fingerprint,)
        if any(
            type(value) is not str
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in fingerprints
        ):
            raise RHFScannerReferenceError(
                "scanner previous_root fingerprints are invalid"
            )
        return root

    def _coordinates(self, mol_or_coordinates) -> np.ndarray:
        if type(mol_or_coordinates) is gto.mole.Mole:
            _reject_molecule_instance_callables(mol_or_coordinates)
            fingerprints_before = (
                _molecule_static_fingerprint(mol_or_coordinates),
                molecule_science_fingerprint(mol_or_coordinates),
            )
            if fingerprints_before[0] != self._system_fingerprint:
                raise RHFScannerReferenceError(
                    "scanner molecule static metadata does not match the template"
                )
            value = gto_mole.Mole.atom_coords(
                mol_or_coordinates,
                unit="Bohr",
            )
            _reject_molecule_instance_callables(mol_or_coordinates)
            fingerprints_after = (
                _molecule_static_fingerprint(mol_or_coordinates),
                molecule_science_fingerprint(mol_or_coordinates),
            )
            if fingerprints_after != fingerprints_before:
                raise RHFScannerReferenceError(
                    "scanner molecule state changed while coordinates were read"
                )
        elif isinstance(mol_or_coordinates, gto.mole.Mole):
            raise TypeError(
                "scanner molecules must be exact native pyscf.gto.Mole objects"
            )
        else:
            try:
                value = np.asarray(mol_or_coordinates)
            except Exception as error:
                raise TypeError(
                    f"scanner coordinates are not a numerical array: {error}"
                ) from error
            if (
                value.dtype.hasobject
                or np.iscomplexobj(value)
                or not (
                    np.issubdtype(value.dtype, np.integer)
                    or np.issubdtype(value.dtype, np.floating)
                )
            ):
                raise TypeError(
                    "scanner coordinates must contain real integer or floating values"
                )
        if value.shape != (self._atom_count, 3):
            raise ValueError(
                "scanner coordinates have shape "
                f"{value.shape}; expected {(self._atom_count, 3)}"
            )
        coordinates = np.asarray(value, dtype=np.float64)
        if not np.isfinite(coordinates).all():
            raise ValueError("scanner coordinates must be finite")
        return np.ascontiguousarray(coordinates).copy()

    def _fresh_molecule(self, coordinates: np.ndarray):
        try:
            molecule = deepcopy(self._template)
            molecule.set_geom_(coordinates, unit="Bohr", inplace=True)
        except Exception as error:
            raise RHFScannerReferenceError(
                f"the fresh scanner molecule could not be built: {error}"
            ) from error
        if (
            type(molecule) is not gto.mole.Mole
            or _molecule_static_fingerprint(molecule) != self._system_fingerprint
        ):
            raise RHFScannerReferenceError(
                "the fresh scanner molecule changed its static metadata"
            )
        return molecule

    def _fresh_reference(self, molecule):
        reference = scf_hf.RHF(molecule)
        if type(reference) is not scf_hf.RHF:
            raise RHFScannerReferenceError(
                "the scanner did not construct an exact native RHF reference"
            )
        for name, value in self._scf_controls.items():
            setattr(reference, name, value)
        reference.chkfile = None
        reference.callback = None
        reference.diis_file = None
        try:
            reference.kernel(dm0=None)
        except Exception as error:
            raise RHFScannerReferenceError(
                f"fresh scanner RHF evaluation failed: {error}"
            ) from error
        if not reference.converged:
            raise DeePHFCapabilityError(
                "the fresh scanner RHF reference did not converge"
            )
        return validate_reference(reference)

    def _occupied_overlap(
        self,
        previous_root: RHFRootSnapshot,
        candidate_reference,
    ) -> float:
        candidate_occupations = np.asarray(candidate_reference.mo_occ)
        if not np.array_equal(candidate_occupations, previous_root.occupations):
            raise RHFScannerReferenceError(
                "the fresh scanner RHF occupations changed from the root anchor"
            )
        candidate_occupied = np.asarray(candidate_reference.mo_coeff)[
            :, candidate_occupations > 0
        ]
        try:
            cross_overlap = gto.intor_cross(
                "int1e_ovlp",
                previous_root._molecule,
                candidate_reference.mol,
            )
        except Exception as error:
            raise RHFScannerReferenceError(
                f"scanner cross-AO overlap construction failed: {error}"
            ) from error
        cross_overlap = _validated_float64_array(
            cross_overlap,
            (self._ao_count, self._ao_count),
            "scanner cross-AO overlap",
        )
        occupied_overlap = (
            previous_root.occupied_coefficients.T
            @ cross_overlap
            @ candidate_occupied
        )
        if not np.isfinite(occupied_overlap).all():
            raise RHFScannerReferenceError(
                "scanner occupied-subspace overlap is nonfinite"
            )
        try:
            singular_values = np.linalg.svd(
                occupied_overlap,
                compute_uv=False,
            )
        except np.linalg.LinAlgError as error:
            raise RHFScannerReferenceError(
                f"scanner occupied-subspace overlap SVD failed: {error}"
            ) from error
        minimum_overlap = float(np.min(singular_values))
        if (
            not np.isfinite(minimum_overlap)
            or minimum_overlap < self.root_overlap_tolerance
        ):
            raise RHFScannerReferenceError(
                "fresh scanner RHF occupied subspace is discontinuous: "
                f"minimum overlap {minimum_overlap:.6f} < "
                f"{self.root_overlap_tolerance:.6f}"
            )
        return minimum_overlap

    def build(
        self,
        mol_or_coordinates,
        previous_root: RHFRootSnapshot,
    ) -> tuple[Any, RHFRootSnapshot]:
        """Build one fresh RHF reference without changing the root anchor."""
        previous_root = self._validate_root(previous_root)
        coordinates = self._coordinates(mol_or_coordinates)
        molecule = self._fresh_molecule(coordinates)
        reference = self._fresh_reference(molecule)
        minimum_overlap = self._occupied_overlap(previous_root, reference)
        candidate_root = self._root_snapshot(
            reference,
            parent_state_fingerprint=previous_root.state_fingerprint,
            minimum_occupied_overlap=minimum_overlap,
        )
        return reference, candidate_root


def response_integrity_fingerprint(response: RHFResponse) -> str:
    """Return a digest covering every response field except the digest itself."""
    return dataclass_fingerprint(
        response,
        excluded=frozenset({"integrity_fingerprint"}),
    )


def adjoint_integrity_fingerprint(adjoint: RHFAdjoint) -> str:
    """Return a digest covering every RHF adjoint field except its digest."""
    return dataclass_fingerprint(
        adjoint,
        excluded=frozenset({"integrity_fingerprint"}),
    )


_cycle_limit = integer_control
_adjoint_real_control = partial(real_control, prefix="adjoint")
_validated_float64_array = partial(
    validated_float64_array,
    error_type=RHFResponseError,
)
