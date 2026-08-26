import hashlib
import json

import numpy as np
import pytest
import torch
from pyscf import gto

import deepks.deephf.workflow as workflow
from deepks.data.force_schema import target_identity_fingerprint
from deepks.deephf.contracts import (
    RootContinuityError,
    occupied_coefficients,
    occupied_subspace_overlaps,
)
from deepks.model.model import CorrNet


PROJECTOR_BASIS = [[0, [0.8, 1.0]]]
TARGET_IDENTITY = {
    "method": "deterministic DeePHF teacher",
    "basis": "sto-3g",
    "software": "deepks-kit",
    "version": "test",
    "frozen_core": False,
    "relativistic": "none",
    "state": "closed-shell singlet ground state",
    "energy_force_consistent": True,
    "settings": {"teacher": "zero CorrNet"},
}


def _h2(distance):
    return gto.M(
        atom=f"H 0 0 0; H 0 0 {distance}",
        basis="sto-3g",
        unit="Bohr",
        spin=0,
        verbose=0,
    )


def _lithium(z):
    return gto.M(
        atom=f"Li 0 0 {z}",
        basis="sto-3g",
        unit="Bohr",
        spin=3,
        verbose=0,
    )


def _write_h2_system(path):
    path.mkdir()
    coordinates = np.array(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.40]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.41]],
        ],
        dtype=np.float64,
    )
    np.save(path / "coord.npy", coordinates)
    (path / "type.raw").write_text("H H\n", encoding="utf-8")


def _projector_fingerprint(projector_basis):
    encoded = json.dumps(
        projector_basis,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_force_checkpoint(path):
    model = CorrNet(
        input_dim=1,
        hidden_sizes=(2,),
        actv_fn="tanh",
        use_resnet=False,
        proj_basis=PROJECTOR_BASIS,
    ).double()
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.zero_()
    model.save(
        path,
        force_training={
            "schema_id": "deepks.deephf.rhf-force-data",
            "schema_version": 2,
            "compatibility_fingerprint": "0" * 64,
            "jacobian_semantics": "dq_dR_relaxed",
            "n_feature": 1,
            "descriptor_definition": "ordered_projected_density_eigenvalues",
            "descriptor_spin_semantics": "spin_summed",
            "descriptor_shell_sizes": [1],
            "projector_sha256": _projector_fingerprint(PROJECTOR_BASIS),
            "reference_family": "RHF",
            "response_backend": "rhf_direct",
            "target": TARGET_IDENTITY,
            "target_fingerprint": target_identity_fingerprint(TARGET_IDENTITY),
        },
    )


def test_unrestricted_empty_spin_channel_has_neutral_overlap():
    previous = workflow.build_reference(_lithium(0.0), "uhf")
    candidate = workflow.build_reference(
        _lithium(0.01),
        "uhf",
        dm0=previous.make_rdm1(),
    )
    overlaps = occupied_subspace_overlaps(
        previous.mol,
        occupied_coefficients(previous.mo_coeff, previous.mo_occ),
        candidate.mol,
        occupied_coefficients(candidate.mo_coeff, candidate.mo_occ),
    )

    assert len(overlaps) == 2
    assert overlaps[0] > 0.99
    assert overlaps[1] == 1.0


@pytest.mark.parametrize(
    ("family", "channel_names"),
    [
        ("rhf", {"restricted"}),
        ("rks", {"restricted"}),
        ("uhf", {"alpha", "beta"}),
        ("uks", {"alpha", "beta"}),
    ],
)
def test_every_reference_family_reuses_previous_density_and_shared_overlap(
    family,
    channel_names,
    monkeypatch,
):
    density_guesses = []
    overlap_calls = []
    original_build_reference = workflow.build_reference
    original_overlap = workflow.occupied_subspace_overlaps

    def recording_build_reference(*args, **kwargs):
        density_guesses.append(kwargs.get("dm0"))
        return original_build_reference(*args, **kwargs)

    def recording_overlap(*args, **kwargs):
        overlap_calls.append(True)
        return original_overlap(*args, **kwargs)

    monkeypatch.setattr(workflow, "build_reference", recording_build_reference)
    monkeypatch.setattr(workflow, "occupied_subspace_overlaps", recording_overlap)
    sequence = workflow._ReferenceSequence(family)
    sequence.build(_h2(1.40))
    sequence.build(_h2(1.41))

    assert density_guesses[0] is None
    assert density_guesses[1] is not None
    assert overlap_calls == [True]
    assert set(sequence.records[1]["occupied_subspace_overlaps"]) == channel_names
    assert sequence.records[1]["initial_guess_source"] == "previous_density"
    assert sequence.records[1]["parent_reference_state_fingerprint"] == (
        sequence.records[0]["reference_state_fingerprint"]
    )


def test_rejected_candidate_does_not_advance_root_anchor(monkeypatch):
    density_guesses = []
    original_build_reference = workflow.build_reference

    def recording_build_reference(*args, **kwargs):
        density = kwargs.get("dm0")
        density_guesses.append(None if density is None else np.array(density, copy=True))
        return original_build_reference(*args, **kwargs)

    monkeypatch.setattr(workflow, "build_reference", recording_build_reference)
    sequence = workflow._ReferenceSequence("rhf", root_overlap_tolerance=0.5)
    sequence.build(_h2(1.40))
    first_fingerprint = sequence.records[0]["reference_state_fingerprint"]
    monkeypatch.setattr(
        workflow,
        "occupied_subspace_overlaps",
        lambda *args, **kwargs: (0.1,),
    )
    with pytest.raises(RootContinuityError, match="discontinuous"):
        sequence.build(_h2(1.41))
    assert len(sequence.records) == 1

    monkeypatch.setattr(
        workflow,
        "occupied_subspace_overlaps",
        lambda *args, **kwargs: (0.9,),
    )
    sequence.build(_h2(1.42))
    np.testing.assert_array_equal(density_guesses[1], density_guesses[2])
    assert sequence.records[1]["parent_reference_state_fingerprint"] == first_fingerprint


def test_cli_rejects_discontinuous_system_without_partial_output(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    _write_h2_system(tmp_path / "system")
    monkeypatch.setattr(
        workflow,
        "occupied_subspace_overlaps",
        lambda *args, **kwargs: (0.1,),
    )

    with pytest.raises(RootContinuityError, match="discontinuous"):
        workflow.main(
            ["system"],
            reference="rhf",
            basis="sto-3g",
            projector_basis=PROJECTOR_BASIS,
            dump_dir="output",
        )

    assert not (tmp_path / "output" / "system").exists()


def test_cli_persists_root_lineage_and_force_checkpoint_target(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_h2_system(tmp_path / "system")
    checkpoint = tmp_path / "force-model.pth"
    _write_force_checkpoint(checkpoint)

    outputs = workflow.main(
        ["system"],
        reference="rhf",
        model_file=checkpoint,
        basis="sto-3g",
        projector_basis=PROJECTOR_BASIS,
        dump_dir="output",
    )

    provenance_path = (
        tmp_path
        / outputs[0][0]
        / workflow.INFERENCE_PROVENANCE_FILENAME
    )
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert provenance["reference_family"] == "RHF"
    assert len(provenance["frames"]) == 2
    assert provenance["frames"][0]["initial_guess_source"] == "independent"
    assert provenance["frames"][1]["initial_guess_source"] == "previous_density"
    assert provenance["frames"][1]["parent_reference_state_fingerprint"] == (
        provenance["frames"][0]["reference_state_fingerprint"]
    )
    assert provenance["model_target"] == {
        "identity": TARGET_IDENTITY,
        "fingerprint": target_identity_fingerprint(TARGET_IDENTITY),
    }
