import copy
import hashlib
import json

import numpy as np
import pytest

from deepks.data.force_schema import (
    ForceDataError,
    force_checkpoint_metadata,
    load_force_dataset,
    validate_force_checkpoint_metadata,
    write_force_dataset,
)
from test_force_schema import make_schema_inputs


def _rewrite_manifest(directory, mutate):
    path = directory / "force_data.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    mutate(manifest)
    manifest.pop("manifest_fingerprint", None)
    encoded = json.dumps(
        manifest,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    manifest["manifest_fingerprint"] = hashlib.sha256(encoded).hexdigest()
    path.write_text(
        json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


@pytest.mark.parametrize("missing", ["f_corr_target", "dq_dR_relaxed"])
def test_writer_rejects_missing_force_fields(tmp_path, missing):
    arrays, provenance = make_schema_inputs()
    arrays.pop(missing)

    with pytest.raises(ForceDataError, match=f"missing {missing}"):
        write_force_dataset(tmp_path / missing, arrays=arrays, provenance=provenance)


def test_writer_rejects_explicit_jacobian_even_when_shape_matches(tmp_path):
    arrays, provenance = make_schema_inputs()
    arrays["dq_dR_explicit"] = arrays.pop("dq_dR_relaxed")

    with pytest.raises(ForceDataError, match="dq_dR_relaxed"):
        write_force_dataset(tmp_path / "explicit", arrays=arrays, provenance=provenance)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda arrays: arrays["descriptor"].astype(np.float32), "float64"),
        (
            lambda arrays: np.full_like(arrays["f_corr_target"], np.nan),
            "finite",
        ),
        (
            lambda arrays: arrays["dq_dR_relaxed"].swapaxes(1, 2),
            "canonical axes",
        ),
    ],
)
def test_writer_rejects_dtype_nonfinite_and_axis_errors(tmp_path, mutation, message):
    arrays, provenance = make_schema_inputs()
    field = (
        "descriptor"
        if "float64" in message
        else "f_corr_target"
        if "finite" in message
        else "dq_dR_relaxed"
    )
    arrays[field] = mutation(arrays)

    with pytest.raises(ForceDataError, match=message):
        write_force_dataset(tmp_path / message, arrays=arrays, provenance=provenance)


@pytest.mark.parametrize("target", ["energy", "force"])
def test_writer_rejects_inconsistent_derived_targets(tmp_path, target):
    arrays, provenance = make_schema_inputs()
    field = "e_corr_target" if target == "energy" else "f_corr_target"
    arrays[field] = arrays[field].copy()
    arrays[field].flat[0] += 1.0e-6

    with pytest.raises(ForceDataError, match=field):
        write_force_dataset(tmp_path / target, arrays=arrays, provenance=provenance)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda provenance: provenance["atom_mapping"].update(
                raw_to_descriptor=[1, 0]
            ),
            "mutual inverses",
        ),
        (
            lambda provenance: provenance["descriptor"].update(
                projector_sha256="0" * 64
            ),
            "projector_sha256",
        ),
        (
            lambda provenance: provenance["reference"].update(family="UHF"),
            "RHF reference",
        ),
        (
            lambda provenance: provenance["frames"][0].update(
                reference_converged=False
            ),
            "unconverged RHF reference",
        ),
        (
            lambda provenance: provenance["frames"][0].update(
                response_converged=False
            ),
            "unconverged RHF response",
        ),
        (
            lambda provenance: provenance["frames"][0][
                "response_diagnostics"
            ].update(maximum_residual=1.0e-4),
            "exceeds tolerance",
        ),
        (
            lambda provenance: provenance["generation"].update(
                producer="explicit-fallback"
            ),
            "producer",
        ),
    ],
)
def test_writer_rejects_invalid_provenance(tmp_path, mutate, message):
    arrays, provenance = make_schema_inputs()
    mutate(provenance)

    with pytest.raises(ForceDataError, match=message):
        write_force_dataset(tmp_path / "invalid", arrays=arrays, provenance=provenance)


def test_loader_rejects_explicit_semantics_under_relaxed_filename(tmp_path):
    arrays, provenance = make_schema_inputs()
    directory = tmp_path / "masquerade"
    write_force_dataset(directory, arrays=arrays, provenance=provenance)

    def mark_explicit(manifest):
        manifest["conventions"]["jacobian_semantics"] = "fixed_density_explicit"
        manifest["fields"]["dq_dR_relaxed"]["semantics"] = (
            "fixed_density_explicit"
        )

    _rewrite_manifest(directory, mark_explicit)

    with pytest.raises(ForceDataError, match="relaxed-Jacobian semantics"):
        load_force_dataset(directory)


def test_loader_rejects_array_content_not_matching_hash(tmp_path):
    arrays, provenance = make_schema_inputs()
    directory = tmp_path / "tampered"
    write_force_dataset(directory, arrays=arrays, provenance=provenance)
    tampered = arrays["dq_dR_relaxed"].copy()
    tampered.flat[0] += 1.0
    np.save(directory / "dq_dR_relaxed.npy", tampered, allow_pickle=False)

    with pytest.raises(ForceDataError, match="content hash"):
        load_force_dataset(directory)


def test_loader_rejects_unit_axis_and_mapping_manifest_changes(tmp_path):
    arrays, provenance = make_schema_inputs()
    directory = tmp_path / "bad-unit"
    write_force_dataset(directory, arrays=arrays, provenance=provenance)

    def change_unit(manifest):
        manifest["conventions"]["jacobian_unit"] = "Angstrom^-1"

    _rewrite_manifest(directory, change_unit)
    with pytest.raises(ForceDataError, match="units"):
        load_force_dataset(directory)


def test_loader_rejects_missing_manifest_and_nonempty_overwrite(tmp_path):
    missing = tmp_path / "missing"
    missing.mkdir()
    with pytest.raises(ForceDataError, match="cannot read force_data.json"):
        load_force_dataset(missing)

    arrays, provenance = make_schema_inputs()
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "user-data.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(ForceDataError, match="must be empty"):
        write_force_dataset(occupied, arrays=arrays, provenance=provenance)
    assert (occupied / "user-data.txt").read_text(encoding="utf-8") == "keep"


def test_checkpoint_metadata_must_match_exact_force_contract(tmp_path):
    arrays, provenance = make_schema_inputs()
    contract = write_force_dataset(
        tmp_path / "data",
        arrays=arrays,
        provenance=provenance,
    )
    metadata = force_checkpoint_metadata(contract)
    metadata["projector_sha256"] = "0" * 64

    with pytest.raises(ForceDataError, match="projector_sha256"):
        validate_force_checkpoint_metadata(metadata, contract)

    with pytest.raises(ForceDataError, match="must be a mapping"):
        validate_force_checkpoint_metadata(None, contract)


@pytest.mark.parametrize(
    ("document", "message"),
    [
        ('{"schema": {}, "schema": {}}', "duplicate JSON key"),
        ('{"bad": NaN}', "invalid JSON constant"),
    ],
)
def test_manifest_duplicate_keys_and_nan_are_rejected(tmp_path, document, message):
    directory = tmp_path / message.replace(" ", "-")
    directory.mkdir()
    (directory / "force_data.json").write_text(document, encoding="utf-8")

    with pytest.raises(ForceDataError, match=message):
        load_force_dataset(directory)
