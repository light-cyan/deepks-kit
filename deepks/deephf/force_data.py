"""Strict RHF DeePHF relaxed-force dataset production.

This module is the method-specific producer for the method-neutral force-data
schema.  It deliberately builds every frame in memory before asking the data
layer to write anything, so a reference or response failure cannot leave an
explicit-only or partially generated dataset behind.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import hashlib
import json
import platform
from types import MappingProxyType
from typing import Any

import numpy as np
import pyscf
import torch

import deepks
from deepks.descriptor import DescriptorDifferentiabilityError

from .capabilities import validate_reference
from .method import DeePHF
from .pyscf_rhf import RHFResponseError, reference_fingerprint


GENERATOR_NAME = "deepks.deephf.force_data"
GENERATOR_VERSION = 1


class RHFForceDataError(ValueError):
    """Raised when a force-data frame violates the persisted P3A contract."""


@dataclass(frozen=True)
class RHFForceFrame:
    """One validated in-memory RHF force-data frame in atomic units."""

    arrays: Mapping[str, np.ndarray]
    provenance: Mapping[str, Any]


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise RHFForceDataError(
        f"force-data provenance cannot serialize {type(value).__name__}"
    )


def _canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        _json_safe(value),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _float64_array(value: Any, expected_shape: tuple[int, ...], name: str) -> np.ndarray:
    try:
        array = np.asarray(value)
    except Exception as error:
        raise RHFForceDataError(f"{name} is not a numerical array: {error}") from error
    if array.shape != expected_shape:
        raise RHFForceDataError(
            f"{name} has shape {array.shape}; expected {expected_shape}"
        )
    if array.dtype != np.dtype(np.float64) or np.iscomplexobj(array):
        raise RHFForceDataError(f"{name} must be a real float64 array")
    if not np.isfinite(array).all():
        raise RHFForceDataError(f"{name} must be finite")
    return np.ascontiguousarray(array)


def _scalar_target(value: Any, name: str) -> float:
    array = np.asarray(value)
    if array.dtype != np.dtype(np.float64) or np.iscomplexobj(array):
        raise RHFForceDataError(f"{name} must be a real float64 scalar")
    if array.size != 1:
        raise RHFForceDataError(f"{name} must contain exactly one scalar")
    scalar = float(array.reshape(-1)[0])
    if not np.isfinite(scalar):
        raise RHFForceDataError(f"{name} must be finite")
    return scalar


def _immutable(array: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(array, dtype=np.float64)
    return np.frombuffer(contiguous.tobytes(), dtype=np.float64).reshape(
        contiguous.shape
    )


def _response_provenance(response) -> dict[str, Any]:
    diagnostics = response.diagnostics
    if diagnostics.maximum_residual > diagnostics.residual_tolerance:
        raise RHFResponseError(
            "force-data generation rejected an RHF response whose independently "
            "computed residual exceeds its tolerance: "
            f"{diagnostics.maximum_residual:.3e} > "
            f"{diagnostics.residual_tolerance:.3e}"
        )
    diagnostic_values = _json_safe(asdict(diagnostics))
    return {
        "backend": "pyscf-2.14-rhf-direct",
        "converged": True,
        "state_fingerprint": response.state_fingerprint,
        "integrity_fingerprint": response.integrity_fingerprint,
        "diagnostics": diagnostic_values,
    }


def _descriptor_diagnostics_provenance(diagnostics) -> dict[str, Any]:
    values = asdict(diagnostics)
    minimum_scaled_gap = float(values["minimum_scaled_gap"])
    if np.isinf(minimum_scaled_gap) and minimum_scaled_gap > 0:
        values["minimum_scaled_gap"] = None
        values["minimum_scaled_gap_unbounded"] = True
    elif not np.isfinite(minimum_scaled_gap):
        raise DescriptorDifferentiabilityError(
            "descriptor differentiability diagnostics contain an invalid "
            f"minimum scaled gap {minimum_scaled_gap!r}"
        )
    return _json_safe(values)


def _reference_provenance(reference) -> dict[str, Any]:
    molecule = reference.mol
    controls = {
        name: _json_safe(getattr(reference, name, None))
        for name in (
            "conv_tol",
            "conv_tol_grad",
            "conv_tol_cpscf",
            "max_cycle",
            "level_shift",
            "diis_space",
            "direct_scf",
            "conv_check",
        )
    }
    return {
        "class": f"{type(reference).__module__}.{type(reference).__qualname__}",
        "method": "RHF",
        "converged": bool(reference.converged),
        "state_fingerprint": reference_fingerprint(reference),
        "charge": int(molecule.charge),
        "spin": int(molecule.spin),
        "electron_count": int(molecule.nelectron),
        "occupations": _json_safe(np.asarray(reference.mo_occ)),
        "basis": _json_safe(molecule._basis),
        "basis_fingerprint": _canonical_digest(molecule._basis),
        "ecp": _json_safe(molecule._ecp),
        "geometry_bohr": _json_safe(molecule.atom_coords(unit="Bohr")),
        "atom_charges": _json_safe(molecule.atom_charges()),
        "ao_count": int(molecule.nao),
        "ao_labels": list(molecule.ao_labels()),
        "scf_controls": controls,
    }


def _frame_signature(frame: RHFForceFrame) -> tuple[Any, ...]:
    provenance = frame.provenance
    reference = provenance["reference"]
    descriptor = provenance["descriptor"]
    return (
        tuple(reference["atom_charges"]),
        reference["charge"],
        reference["spin"],
        tuple(reference["occupations"]),
        reference["basis_fingerprint"],
        reference["ao_count"],
        tuple(reference["ao_labels"]),
        _canonical_digest(reference["scf_controls"]),
        tuple(descriptor["raw_to_descriptor_atom"]),
        tuple(descriptor["shell_sizes"]),
        descriptor["projector_fingerprint"],
    )


def generate_rhf_force_frame(
    reference,
    *,
    projector_basis,
    e_target,
    f_target,
    response_options: Mapping[str, Any] | None = None,
) -> RHFForceFrame:
    """Generate one strict RHF relaxed-force frame without writing files.

    ``e_target`` and ``f_target`` are the complete target energy and force in
    ``Eh`` and ``Eh/Bohr``.  The correction targets are formed relative to the
    native RHF energy and force.  The correction model is intentionally
    ``None`` because the relaxed descriptor Jacobian is model independent.
    """

    options = dict(response_options or {})
    method = DeePHF(
        reference,
        None,
        projector_basis=projector_basis,
        response_options=options,
    )
    method.kernel()
    descriptor_diagnostics = method.validate_force_compatibility()
    if descriptor_diagnostics.structural_zero_blocks:
        raise DescriptorDifferentiabilityError(
            "RHF force-data generation does not persist individual relaxed "
            "descriptor derivatives for structural zero blocks: "
            f"{descriptor_diagnostics.structural_zero_blocks}"
        )

    molecule = method.mol
    target_energy = _scalar_target(e_target, "e_target")
    target_force = _float64_array(
        f_target,
        (molecule.natm, 3),
        "f_target",
    )

    gradient = method.nuc_grad_method(**options)
    gradient.kernel()
    response = gradient.response_result
    if response is None:
        raise RHFResponseError(
            "RHF force-data generation did not receive a direct response"
        )
    response_provenance = _response_provenance(response)

    explicit = _float64_array(
        gradient.dq_dR_explicit,
        (
            molecule.natm,
            3,
            method.n_descriptor_atoms,
            method.n_descriptor_features,
        ),
        "dq_dR_explicit",
    )
    response_part = _float64_array(
        gradient.dq_dR_response,
        explicit.shape,
        "dq_dR_response",
    )
    relaxed = _float64_array(
        gradient.dq_dR_relaxed,
        explicit.shape,
        "dq_dR_relaxed",
    )
    if not np.allclose(
        relaxed,
        explicit + response_part,
        rtol=2.0e-13,
        atol=2.0e-13,
    ):
        maximum_error = float(np.max(np.abs(relaxed - explicit - response_part)))
        raise RHFResponseError(
            "dq_dR_relaxed does not equal explicit plus response: "
            f"maximum error {maximum_error:.3e}"
        )

    atom = np.concatenate(
        (
            molecule.atom_charges().astype(np.float64).reshape(-1, 1),
            np.asarray(molecule.atom_coords(unit="Bohr"), dtype=np.float64),
        ),
        axis=1,
    )
    descriptor = _float64_array(
        method.descriptor(),
        (method.n_descriptor_atoms, method.n_descriptor_features),
        "descriptor",
    )
    reference_gradient = _float64_array(
        gradient.reference_gradient,
        (molecule.natm, 3),
        "native RHF gradient",
    )
    base_force = -reference_gradient
    base_energy = float(method.e_base)

    arrays = {
        "atom": atom,
        "descriptor": descriptor,
        "e_base": np.array([base_energy], dtype=np.float64),
        "f_base": base_force,
        "e_target": np.array([target_energy], dtype=np.float64),
        "f_target": target_force,
        "e_corr_target": np.array([target_energy - base_energy], dtype=np.float64),
        "f_corr_target": target_force - base_force,
        "dq_dR_relaxed": relaxed,
    }
    expected_shapes = {
        "atom": (molecule.natm, 4),
        "descriptor": (method.n_descriptor_atoms, method.n_descriptor_features),
        "e_base": (1,),
        "f_base": (molecule.natm, 3),
        "e_target": (1,),
        "f_target": (molecule.natm, 3),
        "e_corr_target": (1,),
        "f_corr_target": (molecule.natm, 3),
        "dq_dR_relaxed": explicit.shape,
    }
    arrays = {
        name: _immutable(_float64_array(value, expected_shapes[name], name))
        for name, value in arrays.items()
    }

    normalized_projector = _json_safe(method._descriptor.projector_basis)
    raw_to_descriptor = [-1] * molecule.natm
    for descriptor_index, raw_index in enumerate(
        method._descriptor.descriptor_atom_indices
    ):
        raw_to_descriptor[raw_index] = descriptor_index
    provenance = {
        "reference": _reference_provenance(reference),
        "response": response_provenance,
        "descriptor": {
            "definition": "ordered projected-density eigenvalues",
            "spin_semantics": "spin-summed RHF AO density",
            "projector_basis": normalized_projector,
            "projector_fingerprint": _canonical_digest(normalized_projector),
            "shell_sizes": list(method._descriptor.shell_sizes),
            "raw_to_descriptor_atom": raw_to_descriptor,
            "differentiability": _descriptor_diagnostics_provenance(
                descriptor_diagnostics
            ),
        },
    }
    return RHFForceFrame(
        arrays=MappingProxyType(arrays),
        provenance=MappingProxyType(provenance),
    )


def _as_references(references_or_reference) -> tuple[Any, ...]:
    if hasattr(references_or_reference, "mol"):
        return (references_or_reference,)
    if isinstance(references_or_reference, (str, bytes)):
        raise TypeError("references must be native RHF objects, not paths")
    try:
        references = tuple(references_or_reference)
    except TypeError as error:
        raise TypeError(
            "references must be one native RHF object or an iterable of them"
        ) from error
    if not references:
        raise ValueError("at least one RHF reference is required")
    return references


def _energy_targets(values, frame_count: int) -> np.ndarray:
    array = np.asarray(values)
    if frame_count == 1 and array.shape == ():
        array = array.reshape(1)
    elif array.shape == (frame_count, 1):
        array = array.reshape(frame_count)
    if array.shape != (frame_count,):
        raise RHFForceDataError(
            f"e_target has shape {array.shape}; expected ({frame_count},) or "
            f"({frame_count}, 1)"
        )
    if array.dtype != np.dtype(np.float64) or np.iscomplexobj(array):
        raise RHFForceDataError("e_target must be a real float64 array")
    if not np.isfinite(array).all():
        raise RHFForceDataError("e_target must be finite")
    return array


def _force_targets(values, frame_count: int, atom_count: int) -> np.ndarray:
    array = np.asarray(values)
    if frame_count == 1 and array.shape == (atom_count, 3):
        array = array.reshape(1, atom_count, 3)
    expected = (frame_count, atom_count, 3)
    if array.shape != expected:
        raise RHFForceDataError(
            f"f_target has shape {array.shape}; expected {expected}"
        )
    if array.dtype != np.dtype(np.float64) or np.iscomplexobj(array):
        raise RHFForceDataError("f_target must be a real float64 array")
    if not np.isfinite(array).all():
        raise RHFForceDataError("f_target must be finite")
    return array


def write_rhf_force_dataset(
    directory,
    references_or_reference,
    *,
    projector_basis,
    e_target,
    f_target,
    response_options: Mapping[str, Any] | None = None,
):
    """Generate and atomically persist one or more strict RHF force frames.

    All references are validated and all direct responses are completed before
    the method-neutral writer is invoked.  Frames in one dataset must describe
    the same ordered atoms, AO basis, projector, and tensor dimensions.
    """

    references = tuple(
        validate_reference(reference)
        for reference in _as_references(references_or_reference)
    )
    atom_count = int(references[0].mol.natm)
    energies = _energy_targets(e_target, len(references))
    forces = _force_targets(f_target, len(references), atom_count)

    frames = tuple(
        generate_rhf_force_frame(
            reference,
            projector_basis=projector_basis,
            e_target=energies[index],
            f_target=forces[index],
            response_options=response_options,
        )
        for index, reference in enumerate(references)
    )
    signature = _frame_signature(frames[0])
    for frame_index, frame in enumerate(frames[1:], start=1):
        if _frame_signature(frame) != signature:
            raise RHFForceDataError(
                "all RHF force-data frames must share atom ordering, AO basis, "
                f"projector, and tensor dimensions; frame {frame_index} differs"
            )

    arrays = {
        name: np.stack([frame.arrays[name] for frame in frames], axis=0)
        for name in frames[0].arrays
    }
    first_descriptor = frames[0].provenance["descriptor"]
    first_reference = frames[0].provenance["reference"]
    first_response = frames[0].provenance["response"]
    descriptor_to_raw = [
        raw_index
        for raw_index, descriptor_index in enumerate(
            first_descriptor["raw_to_descriptor_atom"]
        )
        if descriptor_index >= 0
    ]
    first_response_diagnostics = first_response["diagnostics"]
    response_control_names = (
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
    )
    frame_provenance = [
        {
            "reference_state_fingerprint": frame.provenance["reference"][
                "state_fingerprint"
            ],
            "reference_converged": frame.provenance["reference"]["converged"],
            "response_converged": frame.provenance["response"]["converged"],
            "response_integrity_fingerprint": frame.provenance["response"][
                "integrity_fingerprint"
            ],
            "response_diagnostics": frame.provenance["response"]["diagnostics"],
            "descriptor_diagnostics": frame.provenance["descriptor"][
                "differentiability"
            ],
            "geometry_bohr": frame.provenance["reference"]["geometry_bohr"],
        }
        for frame in frames
    ]
    provenance = {
        "atom_mapping": {
            "descriptor_to_raw": descriptor_to_raw,
            "raw_to_descriptor": first_descriptor["raw_to_descriptor_atom"],
            "nuclear_charges": first_reference["atom_charges"],
            "ghost_policy": "rejected",
        },
        "descriptor": {
            "definition": "ordered_projected_density_eigenvalues",
            "spin_semantics": "spin_summed",
            "shell_sizes": first_descriptor["shell_sizes"],
            "projector_basis": first_descriptor["projector_basis"],
            "projector_sha256": first_descriptor["projector_fingerprint"],
            "differentiability_controls": {
                "structural_zero_blocks": "rejected",
                "validator": "deepks.descriptor.validate_differentiability",
            },
        },
        "reference": {
            "family": "RHF",
            "python_class": first_reference["class"],
            "basis_content": first_reference["basis"],
            "basis_sha256": first_reference["basis_fingerprint"],
            "ecp": None,
            "charge": first_reference["charge"],
            "spin": first_reference["spin"],
            "occupations": first_reference["occupations"],
            "scf_controls": first_reference["scf_controls"],
        },
        "response": {
            "backend": "rhf_direct",
            "adapter": "deepks.deephf.pyscf_rhf.RHFResponseAdapter",
            "controls": {
                name: first_response_diagnostics[name]
                for name in response_control_names
            },
        },
        "frames": frame_provenance,
        "generation": {
            "producer": f"{GENERATOR_NAME}.rhf_direct",
            "producer_version": GENERATOR_VERSION,
            "deepks_version": deepks.__version__,
            "deepks_commit": getattr(
                getattr(deepks, "_version", None),
                "commit_id",
                None,
            ),
            "pyscf_version": pyscf.__version__,
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "python_version": platform.python_version(),
        },
    }
    # Import locally so the scientific producer remains usable while the
    # method-neutral schema evolves, and so no output path is touched before
    # every direct-response frame has passed validation.
    from deepks.data.force_schema import write_force_dataset

    return write_force_dataset(
        directory,
        arrays=arrays,
        provenance=_json_safe(provenance),
    )


__all__ = [
    "GENERATOR_NAME",
    "GENERATOR_VERSION",
    "RHFForceDataError",
    "RHFForceFrame",
    "generate_rhf_force_frame",
    "write_rhf_force_dataset",
]
