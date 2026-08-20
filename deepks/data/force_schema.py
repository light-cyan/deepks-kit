"""Strict persisted data contract for RHF DeePHF force training.

The schema in this module is deliberately method-neutral at import time.  A
method-specific producer is responsible for proving that the array called
``dq_dR_relaxed`` came from an accepted response calculation; this module
records and validates that proof without importing a method implementation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_FILENAME = "force_data.json"
SCHEMA_ID = "deepks.deephf.rhf-force-data"
SCHEMA_VERSION = 1
JACOBIAN_NAME = "dq_dR_relaxed"
JACOBIAN_SEMANTICS = "complete_relaxed_reference_response"
TARGET_IDENTITY_TOLERANCE = 1.0e-12


class ForceDataError(ValueError):
    """Raised when persisted force data violate the strict v1 contract."""


@dataclass(frozen=True)
class _FieldSpec:
    axes: tuple[str, ...]
    unit: str
    sign: str
    semantics: str


_FIELD_SPECS = {
    "atom": _FieldSpec(
        ("frame", "raw_atom", "charge_and_cartesian"),
        "nuclear_charge+Bohr",
        "not_applicable",
        "nuclear_charge_then_raw_atom_position",
    ),
    "descriptor": _FieldSpec(
        ("frame", "descriptor_atom", "feature"),
        "1",
        "not_applicable",
        "ordered_projected_density_eigenvalues",
    ),
    "e_base": _FieldSpec(
        ("frame", "scalar"),
        "Eh",
        "+energy",
        "native_rhf_energy",
    ),
    "f_base": _FieldSpec(
        ("frame", "raw_atom", "cartesian"),
        "Eh/Bohr",
        "force=-dE/dR",
        "native_rhf_force",
    ),
    "e_target": _FieldSpec(
        ("frame", "scalar"),
        "Eh",
        "+energy",
        "supervised_total_energy",
    ),
    "f_target": _FieldSpec(
        ("frame", "raw_atom", "cartesian"),
        "Eh/Bohr",
        "force=-dE/dR",
        "supervised_total_force",
    ),
    "e_corr_target": _FieldSpec(
        ("frame", "scalar"),
        "Eh",
        "+energy",
        "e_target-e_base",
    ),
    "f_corr_target": _FieldSpec(
        ("frame", "raw_atom", "cartesian"),
        "Eh/Bohr",
        "force=-dE/dR",
        "f_target-f_base",
    ),
    JACOBIAN_NAME: _FieldSpec(
        (
            "frame",
            "raw_atom",
            "cartesian",
            "descriptor_atom",
            "feature",
        ),
        "Bohr^-1",
        "+dq/dR",
        JACOBIAN_SEMANTICS,
    ),
}

CANONICAL_FORCE_FIELDS = tuple(_FIELD_SPECS)

_CONVENTIONS = {
    "length_unit": "Bohr",
    "energy_unit": "Eh",
    "force_unit": "Eh/Bohr",
    "jacobian_unit": "Bohr^-1",
    "cartesian_order": ["x", "y", "z"],
    "force_sign": "force=-dE/dR",
    "jacobian_sign": "+dq/dR",
    "jacobian_name": JACOBIAN_NAME,
    "jacobian_semantics": JACOBIAN_SEMANTICS,
}

_TOP_LEVEL_KEYS = {
    "schema",
    "dimensions",
    "conventions",
    "fields",
    "atom_mapping",
    "descriptor",
    "reference",
    "response",
    "frames",
    "generation",
    "compatibility_fingerprint",
    "manifest_fingerprint",
}


def _error(message: str) -> ForceDataError:
    return ForceDataError(f"RHF DeePHF force-data contract: {message}")


def _json_value(value: Any, path: str = "value") -> Any:
    """Convert supported metadata to a deterministic JSON value."""
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise _error(f"{path} must not contain NaN or infinity")
        return value
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist(), path)
    if isinstance(value, (list, tuple)):
        return [_json_value(item, f"{path}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, Mapping):
        normalized = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise _error(f"{path} metadata keys must be strings")
            normalized[key] = _json_value(item, f"{path}.{key}")
        return normalized
    raise _error(f"{path} contains unsupported {type(value).__name__} metadata")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _json_value(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _json_fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _array_fingerprint(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(_canonical_json(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _error(f"{name} must be a mapping")
    return _json_value(value, name)


def _require_keys(value: Mapping[str, Any], keys: set[str], name: str) -> None:
    missing = sorted(keys - set(value))
    if missing:
        raise _error(f"{name} is missing required keys: {', '.join(missing)}")


def _require_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise _error(f"{name} must be boolean")
    return value


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise _error(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise _error(f"{name} must be a finite number") from error
    if not math.isfinite(result):
        raise _error(f"{name} must be a finite number")
    return result


def _sha256_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise _error(f"{name} must be a SHA-256 hexadecimal string")
    try:
        int(value, 16)
    except ValueError as error:
        raise _error(f"{name} must be a SHA-256 hexadecimal string") from error
    return value.lower()


def _expected_shapes(dimensions: Mapping[str, int]) -> dict[str, tuple[int, ...]]:
    frame = dimensions["n_frame"]
    raw_atom = dimensions["n_raw_atom"]
    descriptor_atom = dimensions["n_descriptor_atom"]
    feature = dimensions["n_feature"]
    return {
        "atom": (frame, raw_atom, 4),
        "descriptor": (frame, descriptor_atom, feature),
        "e_base": (frame, 1),
        "f_base": (frame, raw_atom, 3),
        "e_target": (frame, 1),
        "f_target": (frame, raw_atom, 3),
        "e_corr_target": (frame, 1),
        "f_corr_target": (frame, raw_atom, 3),
        JACOBIAN_NAME: (frame, raw_atom, 3, descriptor_atom, feature),
    }


def _validate_arrays(
    arrays: Mapping[str, np.ndarray],
    dimensions: Mapping[str, int] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    if not isinstance(arrays, Mapping):
        raise _error("arrays must be a mapping")
    actual_names = set(arrays)
    expected_names = set(CANONICAL_FORCE_FIELDS)
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if extra:
            details.append("unexpected " + ", ".join(extra))
        raise _error("canonical arrays are incomplete: " + "; ".join(details))

    normalized = {}
    for name in CANONICAL_FORCE_FIELDS:
        value = arrays[name]
        if not isinstance(value, np.ndarray):
            raise _error(f"field {name} must be a numpy.ndarray")
        if value.dtype != np.dtype(np.float64) or np.iscomplexobj(value):
            raise _error(f"field {name} must be real numpy.float64")
        if not np.isfinite(value).all():
            raise _error(f"field {name} must contain only finite values")
        normalized[name] = np.ascontiguousarray(value)

    if normalized["atom"].ndim != 3 or normalized["atom"].shape[-1] != 4:
        raise _error("field atom must have exact axes (frame, raw_atom, 4)")
    if normalized["descriptor"].ndim != 3:
        raise _error(
            "field descriptor must have exact axes (frame, descriptor_atom, feature)"
        )
    inferred = {
        "n_frame": normalized["atom"].shape[0],
        "n_raw_atom": normalized["atom"].shape[1],
        "n_descriptor_atom": normalized["descriptor"].shape[1],
        "n_feature": normalized["descriptor"].shape[2],
    }
    if any(value <= 0 for value in inferred.values()):
        raise _error("all force-data dimensions must be positive")
    if dimensions is not None and dict(dimensions) != inferred:
        raise _error(
            f"manifest dimensions {dict(dimensions)} do not match arrays {inferred}"
        )
    expected_shapes = _expected_shapes(inferred)
    for name, expected_shape in expected_shapes.items():
        if normalized[name].shape != expected_shape:
            raise _error(
                f"field {name} has shape {normalized[name].shape}; "
                f"expected {expected_shape} with canonical axes"
            )

    energy_error = np.max(
        np.abs(
            normalized["e_corr_target"]
            - (normalized["e_target"] - normalized["e_base"])
        ),
        initial=0.0,
    )
    if energy_error > TARGET_IDENTITY_TOLERANCE:
        raise _error(
            "e_corr_target does not equal e_target-e_base; "
            f"maximum residual {energy_error:.3e}"
        )
    force_error = np.max(
        np.abs(
            normalized["f_corr_target"]
            - (normalized["f_target"] - normalized["f_base"])
        ),
        initial=0.0,
    )
    if force_error > TARGET_IDENTITY_TOLERANCE:
        raise _error(
            "f_corr_target does not equal f_target-f_base; "
            f"maximum residual {force_error:.3e}"
        )
    return normalized, inferred


def _normalize_provenance(
    provenance: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    dimensions: Mapping[str, int],
) -> dict[str, Any]:
    provenance = _require_mapping(provenance, "provenance")
    required_sections = {
        "atom_mapping",
        "descriptor",
        "reference",
        "response",
        "frames",
        "generation",
    }
    _require_keys(provenance, required_sections, "provenance")
    extra_sections = sorted(set(provenance) - required_sections)
    if extra_sections:
        raise _error(
            "provenance has unexpected top-level sections: "
            + ", ".join(extra_sections)
        )

    raw_atom = dimensions["n_raw_atom"]
    descriptor_atom = dimensions["n_descriptor_atom"]
    feature = dimensions["n_feature"]

    mapping = _require_mapping(provenance["atom_mapping"], "atom_mapping")
    _require_keys(
        mapping,
        {
            "descriptor_to_raw",
            "raw_to_descriptor",
            "nuclear_charges",
            "ghost_policy",
        },
        "atom_mapping",
    )
    descriptor_to_raw = mapping["descriptor_to_raw"]
    raw_to_descriptor = mapping["raw_to_descriptor"]
    nuclear_charges = mapping["nuclear_charges"]
    if not isinstance(descriptor_to_raw, list) or len(descriptor_to_raw) != descriptor_atom:
        raise _error("atom_mapping.descriptor_to_raw has the wrong length")
    if not isinstance(raw_to_descriptor, list) or len(raw_to_descriptor) != raw_atom:
        raise _error("atom_mapping.raw_to_descriptor has the wrong length")
    if not isinstance(nuclear_charges, list) or len(nuclear_charges) != raw_atom:
        raise _error("atom_mapping.nuclear_charges has the wrong length")
    if mapping["ghost_policy"] != "rejected":
        raise _error("v1 RHF force data require ghost_policy='rejected'")
    if raw_atom != descriptor_atom:
        raise _error("v1 RHF force data require every raw atom to be a descriptor atom")
    if any(isinstance(index, bool) or not isinstance(index, int) for index in descriptor_to_raw):
        raise _error("descriptor_to_raw indices must be integers")
    if any(isinstance(index, bool) or not isinstance(index, int) for index in raw_to_descriptor):
        raise _error("raw_to_descriptor indices must be integers")
    if sorted(descriptor_to_raw) != list(range(raw_atom)):
        raise _error("descriptor_to_raw must be a permutation of raw atoms")
    if sorted(raw_to_descriptor) != list(range(descriptor_atom)):
        raise _error("raw_to_descriptor must be a permutation of descriptor atoms")
    for descriptor_index, raw_index in enumerate(descriptor_to_raw):
        if raw_to_descriptor[raw_index] != descriptor_index:
            raise _error("raw/descriptor atom mappings are not mutual inverses")
    if any(
        isinstance(charge, bool) or not isinstance(charge, int) or charge <= 0
        for charge in nuclear_charges
    ):
        raise _error("v1 RHF nuclear charges must be positive integers")
    stored_charges = arrays["atom"][..., 0]
    expected_charges = np.asarray(nuclear_charges, dtype=np.float64)
    if not np.array_equal(
        stored_charges,
        np.broadcast_to(expected_charges, stored_charges.shape),
    ):
        raise _error("atom nuclear charges do not match atom_mapping provenance")

    descriptor = _require_mapping(provenance["descriptor"], "descriptor")
    _require_keys(
        descriptor,
        {
            "definition",
            "spin_semantics",
            "shell_sizes",
            "projector_basis",
            "differentiability_controls",
        },
        "descriptor",
    )
    if descriptor["definition"] != "ordered_projected_density_eigenvalues":
        raise _error("descriptor definition is not the canonical ordered spectrum")
    if descriptor["spin_semantics"] != "spin_summed":
        raise _error("v1 RHF force data require spin_summed descriptor semantics")
    shell_sizes = descriptor["shell_sizes"]
    if (
        not isinstance(shell_sizes, list)
        or not shell_sizes
        or any(isinstance(size, bool) or not isinstance(size, int) or size <= 0 for size in shell_sizes)
        or sum(shell_sizes) != feature
    ):
        raise _error("descriptor shell_sizes must be positive and sum to n_feature")
    projector_hash = _json_fingerprint(descriptor["projector_basis"])
    supplied_projector_hash = descriptor.get("projector_sha256")
    if supplied_projector_hash is not None and supplied_projector_hash != projector_hash:
        raise _error("descriptor projector_sha256 does not match projector_basis")
    descriptor["projector_sha256"] = projector_hash
    if not isinstance(descriptor["differentiability_controls"], dict):
        raise _error("descriptor differentiability_controls must be a mapping")

    reference = _require_mapping(provenance["reference"], "reference")
    _require_keys(
        reference,
        {
            "family",
            "python_class",
            "basis_content",
            "ecp",
            "charge",
            "spin",
            "occupations",
            "scf_controls",
        },
        "reference",
    )
    if reference["family"] != "RHF" or reference["python_class"] != "pyscf.scf.hf.RHF":
        raise _error("v1 force data require an exact native pyscf.scf.hf.RHF reference")
    if reference["ecp"] not in (None, {}):
        raise _error("v1 RHF force data require an all-electron reference")
    reference["ecp"] = None
    if isinstance(reference["charge"], bool) or not isinstance(reference["charge"], int):
        raise _error("reference charge must be an integer")
    if reference["spin"] != 0:
        raise _error("v1 RHF force data require spin zero")
    occupations = reference["occupations"]
    if (
        not isinstance(occupations, list)
        or not occupations
        or any(value not in (0, 0.0, 2, 2.0) for value in occupations)
        or 0.0 not in [float(value) for value in occupations]
        or 2.0 not in [float(value) for value in occupations]
    ):
        raise _error("reference occupations must contain closed-shell occupied and virtual spaces")
    if not isinstance(reference["scf_controls"], dict):
        raise _error("reference scf_controls must be a mapping")
    basis_hash = _json_fingerprint(reference["basis_content"])
    supplied_basis_hash = reference.get("basis_sha256")
    if supplied_basis_hash is not None and supplied_basis_hash != basis_hash:
        raise _error("reference basis_sha256 does not match basis_content")
    reference["basis_sha256"] = basis_hash

    response = _require_mapping(provenance["response"], "response")
    _require_keys(response, {"backend", "adapter", "controls"}, "response")
    if response["backend"] != "rhf_direct":
        raise _error("v1 force data require the rhf_direct response backend")
    if response["adapter"] != "deepks.deephf.pyscf_rhf.RHFResponseAdapter":
        raise _error("response adapter identity is not the accepted RHF adapter")
    if not isinstance(response["controls"], dict):
        raise _error("response controls must be a mapping")

    frames = provenance["frames"]
    if not isinstance(frames, list) or len(frames) != dimensions["n_frame"]:
        raise _error("frames provenance must contain exactly one record per frame")
    normalized_frames = []
    compatibility_seed = {
        "schema": {"id": SCHEMA_ID, "version": SCHEMA_VERSION},
        "conventions": _CONVENTIONS,
        "descriptor": {
            "definition": descriptor["definition"],
            "spin_semantics": descriptor["spin_semantics"],
            "shell_sizes": descriptor["shell_sizes"],
            "projector_sha256": descriptor["projector_sha256"],
        },
        "reference": {
            "family": reference["family"],
            "python_class": reference["python_class"],
        },
        "response": {
            "backend": response["backend"],
            "adapter": response["adapter"],
        },
        "dtype": "float64",
        "n_feature": feature,
        "field_contracts": {
            name: {
                "axes": list(spec.axes),
                "unit": spec.unit,
                "sign": spec.sign,
                "semantics": spec.semantics,
            }
            for name, spec in _FIELD_SPECS.items()
        },
    }
    compatibility_fingerprint = _json_fingerprint(compatibility_seed)
    for frame_index, raw_frame in enumerate(frames):
        frame = _require_mapping(raw_frame, f"frames[{frame_index}]")
        _require_keys(
            frame,
            {
                "reference_state_fingerprint",
                "reference_converged",
                "response_converged",
                "response_diagnostics",
                "descriptor_diagnostics",
            },
            f"frames[{frame_index}]",
        )
        frame["reference_state_fingerprint"] = _sha256_string(
            frame["reference_state_fingerprint"],
            f"frames[{frame_index}].reference_state_fingerprint",
        )
        if not _require_bool(
            frame["reference_converged"],
            f"frames[{frame_index}].reference_converged",
        ):
            raise _error(f"frame {frame_index} has an unconverged RHF reference")
        if not _require_bool(
            frame["response_converged"],
            f"frames[{frame_index}].response_converged",
        ):
            raise _error(f"frame {frame_index} has an unconverged RHF response")
        diagnostics = _require_mapping(
            frame["response_diagnostics"],
            f"frames[{frame_index}].response_diagnostics",
        )
        _require_keys(
            diagnostics,
            {"maximum_residual", "residual_tolerance"},
            f"frames[{frame_index}].response_diagnostics",
        )
        maximum_residual = _finite_float(
            diagnostics["maximum_residual"],
            f"frames[{frame_index}].response_diagnostics.maximum_residual",
        )
        residual_tolerance = _finite_float(
            diagnostics["residual_tolerance"],
            f"frames[{frame_index}].response_diagnostics.residual_tolerance",
        )
        if residual_tolerance <= 0:
            raise _error(f"frame {frame_index} response residual tolerance must be positive")
        if maximum_residual > residual_tolerance:
            raise _error(
                f"frame {frame_index} response residual {maximum_residual:.3e} "
                f"exceeds tolerance {residual_tolerance:.3e}"
            )
        frame["response_diagnostics"] = diagnostics
        frame["descriptor_diagnostics"] = _require_mapping(
            frame["descriptor_diagnostics"],
            f"frames[{frame_index}].descriptor_diagnostics",
        )
        supplied_sample_id = frame.pop("sample_id", None)
        sample_identity = {
            "compatibility_fingerprint": compatibility_fingerprint,
            "frame_provenance": frame,
            "arrays": {
                name: _array_fingerprint(value[frame_index])
                for name, value in arrays.items()
            },
        }
        sample_id = _json_fingerprint(sample_identity)
        if supplied_sample_id is not None and supplied_sample_id != sample_id:
            raise _error(f"frame {frame_index} sample_id does not match its data")
        frame["sample_id"] = sample_id
        normalized_frames.append(frame)

    generation = _require_mapping(provenance["generation"], "generation")
    _require_keys(
        generation,
        {
            "deepks_version",
            "deepks_commit",
            "pyscf_version",
            "torch_version",
            "numpy_version",
            "python_version",
            "producer",
        },
        "generation",
    )
    for name in (
        "deepks_version",
        "pyscf_version",
        "torch_version",
        "numpy_version",
        "python_version",
    ):
        if not isinstance(generation[name], str) or not generation[name]:
            raise _error(f"generation.{name} must be a nonempty string")
    if generation["deepks_commit"] is not None and not isinstance(
        generation["deepks_commit"], str
    ):
        raise _error("generation.deepks_commit must be a string or null")
    if generation["producer"] != "deepks.deephf.force_data.rhf_direct":
        raise _error("generation producer is not the accepted RHF direct producer")

    return {
        "atom_mapping": mapping,
        "descriptor": descriptor,
        "reference": reference,
        "response": response,
        "frames": normalized_frames,
        "generation": generation,
        "compatibility_fingerprint": compatibility_fingerprint,
    }


def _build_manifest(
    arrays: Mapping[str, np.ndarray],
    dimensions: Mapping[str, int],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    fields = {}
    for name, spec in _FIELD_SPECS.items():
        value = arrays[name]
        fields[name] = {
            "file": f"{name}.npy",
            "axes": list(spec.axes),
            "shape": list(value.shape),
            "dtype": "float64",
            "unit": spec.unit,
            "sign": spec.sign,
            "semantics": spec.semantics,
            "sha256": _array_fingerprint(value),
        }
    manifest = {
        "schema": {"id": SCHEMA_ID, "version": SCHEMA_VERSION},
        "dimensions": dict(dimensions),
        "conventions": _CONVENTIONS,
        "fields": fields,
        "atom_mapping": provenance["atom_mapping"],
        "descriptor": provenance["descriptor"],
        "reference": provenance["reference"],
        "response": provenance["response"],
        "frames": provenance["frames"],
        "generation": provenance["generation"],
        "compatibility_fingerprint": provenance["compatibility_fingerprint"],
    }
    manifest["manifest_fingerprint"] = _json_fingerprint(manifest)
    return manifest


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise _error(f"manifest contains duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value):
    raise _error(f"manifest contains invalid JSON constant {value}")


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise _error(f"cannot read {SCHEMA_FILENAME}: {error}") from error
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except ForceDataError:
        raise
    except (UnicodeError, json.JSONDecodeError) as error:
        raise _error(f"{SCHEMA_FILENAME} is not valid UTF-8 JSON: {error}") from error
    if not isinstance(value, dict):
        raise _error("manifest root must be a JSON object")
    return value


def _validate_manifest_header(manifest: Mapping[str, Any]) -> dict[str, int]:
    actual_keys = set(manifest)
    if actual_keys != _TOP_LEVEL_KEYS:
        missing = sorted(_TOP_LEVEL_KEYS - actual_keys)
        extra = sorted(actual_keys - _TOP_LEVEL_KEYS)
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if extra:
            details.append("unexpected " + ", ".join(extra))
        raise _error("manifest top-level keys are invalid: " + "; ".join(details))
    if manifest["schema"] != {"id": SCHEMA_ID, "version": SCHEMA_VERSION}:
        raise _error(
            f"unsupported schema identity/version {manifest.get('schema')!r}"
        )
    if manifest["conventions"] != _CONVENTIONS:
        raise _error("manifest axes, units, signs, or relaxed-Jacobian semantics differ from v1")
    dimensions = manifest["dimensions"]
    if not isinstance(dimensions, dict) or set(dimensions) != {
        "n_frame",
        "n_raw_atom",
        "n_descriptor_atom",
        "n_feature",
    }:
        raise _error("manifest dimensions are incomplete")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in dimensions.values()
    ):
        raise _error("manifest dimensions must be positive integers")
    fingerprint = manifest["manifest_fingerprint"]
    unsigned = dict(manifest)
    unsigned.pop("manifest_fingerprint")
    if fingerprint != _json_fingerprint(unsigned):
        raise _error("manifest fingerprint does not match its contents")
    return dimensions


def _validate_field_manifest(
    manifest: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
) -> None:
    fields = manifest["fields"]
    if not isinstance(fields, dict) or set(fields) != set(CANONICAL_FORCE_FIELDS):
        raise _error("manifest must describe exactly the nine canonical force fields")
    for name, spec in _FIELD_SPECS.items():
        expected = {
            "file": f"{name}.npy",
            "axes": list(spec.axes),
            "shape": list(arrays[name].shape),
            "dtype": "float64",
            "unit": spec.unit,
            "sign": spec.sign,
            "semantics": spec.semantics,
            "sha256": _array_fingerprint(arrays[name]),
        }
        if fields[name] != expected:
            raise _error(
                f"manifest contract or content hash for field {name} does not match v1 data"
            )


@dataclass(frozen=True)
class ForceDataContract:
    """Immutable view of one validated strict force-data manifest."""

    _canonical_manifest: str

    @classmethod
    def _from_manifest(cls, manifest: Mapping[str, Any]) -> "ForceDataContract":
        return cls(_canonical_json(manifest))

    @property
    def manifest(self) -> dict[str, Any]:
        return json.loads(self._canonical_manifest)

    @property
    def schema_id(self) -> str:
        return SCHEMA_ID

    @property
    def schema_version(self) -> int:
        return SCHEMA_VERSION

    @property
    def dimensions(self) -> dict[str, int]:
        return dict(self.manifest["dimensions"])

    @property
    def compatibility_fingerprint(self) -> str:
        return self.manifest["compatibility_fingerprint"]

    @property
    def force_contract_fingerprint(self) -> str:
        """Compatibility alias for checkpoint callers."""
        return self.compatibility_fingerprint

    @property
    def manifest_fingerprint(self) -> str:
        return self.manifest["manifest_fingerprint"]

    @property
    def jacobian_semantics(self) -> str:
        return JACOBIAN_NAME


def write_force_dataset(
    directory,
    *,
    arrays: Mapping[str, np.ndarray],
    provenance: Mapping[str, Any],
) -> ForceDataContract:
    """Validate and write one complete strict v1 force dataset.

    Existing nonempty directories are rejected so an interrupted or unrelated
    dataset cannot be silently overwritten.
    """
    normalized_arrays, dimensions = _validate_arrays(arrays)
    normalized_provenance = _normalize_provenance(
        provenance,
        normalized_arrays,
        dimensions,
    )
    manifest = _build_manifest(
        normalized_arrays,
        dimensions,
        normalized_provenance,
    )
    contract = ForceDataContract._from_manifest(manifest)

    path = Path(directory)
    if path.exists() and not path.is_dir():
        raise _error(f"output path {path} is not a directory")
    if path.exists() and any(path.iterdir()):
        raise _error(f"output directory {path} must be empty")
    path.mkdir(parents=True, exist_ok=True)

    temporary_paths = []
    final_paths = []
    try:
        for name in CANONICAL_FORCE_FIELDS:
            final_path = path / f"{name}.npy"
            temporary_path = path / f".{name}.npy.tmp"
            temporary_paths.append(temporary_path)
            with temporary_path.open("wb") as stream:
                np.save(stream, normalized_arrays[name], allow_pickle=False)
            os.replace(temporary_path, final_path)
            final_paths.append(final_path)
        manifest_path = path / SCHEMA_FILENAME
        temporary_manifest_path = path / f".{SCHEMA_FILENAME}.tmp"
        temporary_paths.append(temporary_manifest_path)
        temporary_manifest_path.write_text(
            json.dumps(
                manifest,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_manifest_path, manifest_path)
        final_paths.append(manifest_path)
    except Exception:
        for created_path in temporary_paths + final_paths:
            try:
                created_path.unlink()
            except FileNotFoundError:
                pass
        raise
    return contract


def load_force_dataset(directory) -> tuple[ForceDataContract, dict[str, np.ndarray]]:
    """Load and strictly validate one complete v1 force dataset."""
    path = Path(directory)
    if not path.is_dir():
        raise _error(f"dataset path {path} is not a directory")
    manifest = _load_manifest(path / SCHEMA_FILENAME)
    dimensions = _validate_manifest_header(manifest)
    arrays = {}
    for name in CANONICAL_FORCE_FIELDS:
        field_path = path / f"{name}.npy"
        try:
            arrays[name] = np.load(field_path, allow_pickle=False)
        except (OSError, ValueError) as error:
            raise _error(f"cannot load canonical field {name}: {error}") from error
    arrays, _ = _validate_arrays(arrays, dimensions)
    _validate_field_manifest(manifest, arrays)

    reconstructed_provenance = {
        name: manifest[name]
        for name in (
            "atom_mapping",
            "descriptor",
            "reference",
            "response",
            "frames",
            "generation",
        )
    }
    normalized_provenance = _normalize_provenance(
        reconstructed_provenance,
        arrays,
        dimensions,
    )
    if normalized_provenance["compatibility_fingerprint"] != manifest[
        "compatibility_fingerprint"
    ]:
        raise _error("force compatibility fingerprint does not match provenance")
    rebuilt = _build_manifest(arrays, dimensions, normalized_provenance)
    if rebuilt != manifest:
        raise _error("manifest is not the canonical representation of this dataset")
    return ForceDataContract._from_manifest(manifest), arrays


def force_checkpoint_metadata(contract: ForceDataContract) -> dict[str, Any]:
    """Return the exact force contract that a checkpoint must retain."""
    if not isinstance(contract, ForceDataContract):
        raise TypeError("contract must be a ForceDataContract")
    manifest = contract.manifest
    descriptor = manifest["descriptor"]
    return {
        "schema_id": contract.schema_id,
        "schema_version": contract.schema_version,
        "compatibility_fingerprint": contract.compatibility_fingerprint,
        "jacobian_semantics": contract.jacobian_semantics,
        "n_feature": contract.dimensions["n_feature"],
        "descriptor_definition": descriptor["definition"],
        "descriptor_spin_semantics": descriptor["spin_semantics"],
        "descriptor_shell_sizes": list(descriptor["shell_sizes"]),
        "projector_sha256": descriptor["projector_sha256"],
        "reference_family": manifest["reference"]["family"],
        "response_backend": manifest["response"]["backend"],
    }


def validate_force_checkpoint_metadata(
    metadata: Mapping[str, Any],
    contract: ForceDataContract,
) -> None:
    """Reject checkpoint metadata that differ from a dataset force contract."""
    if not isinstance(metadata, Mapping):
        raise _error("checkpoint force metadata must be a mapping")
    expected = force_checkpoint_metadata(contract)
    try:
        actual = _json_value(metadata, "checkpoint force metadata")
    except ForceDataError:
        raise
    if actual != expected:
        differing = sorted(
            key
            for key in set(actual) | set(expected)
            if actual.get(key) != expected.get(key)
        )
        raise _error(
            "checkpoint force contract is incompatible; differing keys: "
            + ", ".join(differing)
        )


__all__ = [
    "CANONICAL_FORCE_FIELDS",
    "ForceDataContract",
    "ForceDataError",
    "JACOBIAN_NAME",
    "SCHEMA_FILENAME",
    "SCHEMA_ID",
    "SCHEMA_VERSION",
    "force_checkpoint_metadata",
    "load_force_dataset",
    "validate_force_checkpoint_metadata",
    "write_force_dataset",
]
