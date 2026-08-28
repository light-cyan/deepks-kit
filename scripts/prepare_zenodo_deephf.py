#!/usr/bin/env python
"""Prepare the published B3LYP(G&T) model and selected MD systems."""

from __future__ import annotations

import argparse
from collections import Counter
from io import BytesIO
import hashlib
import json
from pathlib import Path
import tarfile

import numpy as np
from ase.data import chemical_symbols

from deepks.model.legacy import load_legacy_corrnet_bytes


ZENODO_RECORD = "https://zenodo.org/records/14876882"
ARCHIVE_MD5 = "b649b9818a454e6d561583715f579f96"
MODEL_MEMBER = "project2/B3LYP/GRAMandT1X/model.out/model.pth"
MODEL_SHA256 = "cabe08ca87ee65408ee78b9945a2c75932223b8754c663b3fe848f68259da2a5"
DATA_PREFIX = "project2/B3LYP/GRAM/datasets/data_test"
SELECTED_SYSTEMS = (
    "rxn000026_p000026_0.log",
    "rxn000060_p000060_1.log",
    "rxn000390_p000390_1.log",
    "rxn003237_p003237_0.log",
    "rxn005972_p005972_0.log",
    "rxn002232_p002232_0.log",
    "rxn001158_r001158.log",
    "rxn000715_r000715.log",
    "rxn002081_p002081_1.log",
)
DATA_FILES = (
    "atom.npy",
    "conv.npy",
    "dm_eig.npy",
    "e_base.npy",
    "e_tot.npy",
    "l_e_delta.npy",
    "system.raw",
)
BOHR_TO_ANGSTROM = 0.529177210903


def file_digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def formula(charges: np.ndarray) -> str:
    counts = Counter(int(value) for value in charges)
    order = [6, 1, 7, 8]
    fields = []
    for charge in order:
        count = counts.pop(charge, 0)
        if count:
            fields.append(chemical_symbols[charge] + (str(count) if count > 1 else ""))
    for charge in sorted(counts):
        count = counts[charge]
        fields.append(chemical_symbols[charge] + (str(count) if count > 1 else ""))
    return "".join(fields)


def read_archive(archive: Path) -> tuple[bytes, dict[str, dict[str, bytes]]]:
    wanted = {
        f"{DATA_PREFIX}/{system}/{filename}": (system, filename)
        for system in SELECTED_SYSTEMS
        for filename in DATA_FILES
    }
    systems = {system: {} for system in SELECTED_SYSTEMS}
    model_payload = None
    with tarfile.open(archive, "r|gz") as stream:
        for member in stream:
            if member.name == MODEL_MEMBER:
                model_payload = stream.extractfile(member).read()
            target = wanted.get(member.name)
            if target is not None:
                system, filename = target
                systems[system][filename] = stream.extractfile(member).read()
    if model_payload is None:
        raise ValueError(f"archive is missing {MODEL_MEMBER}")
    missing = {
        system: sorted(set(DATA_FILES) - set(files))
        for system, files in systems.items()
        if set(files) != set(DATA_FILES)
    }
    if missing:
        raise ValueError(f"archive is missing selected system data: {missing}")
    return model_payload, systems


def load_array(payload: bytes) -> np.ndarray:
    return np.load(BytesIO(payload), allow_pickle=False)


def write_xyz(path: Path, atom: np.ndarray, source: str) -> None:
    charges = np.rint(atom[:, 0]).astype(int)
    coordinates = np.asarray(atom[:, 1:], dtype=np.float64) * BOHR_TO_ANGSTROM
    lines = [
        str(len(atom)),
        f"charge=0 multiplicity=1 dataset=GRAM split=test source={source}",
    ]
    lines.extend(
        f"{chemical_symbols[charge]} {coordinate[0]:.15f} {coordinate[1]:.15f} {coordinate[2]:.15f}"
        for charge, coordinate in zip(charges, coordinates, strict=True)
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare(archive: Path, output: Path) -> dict:
    if file_digest(archive, "md5") != ARCHIVE_MD5:
        raise ValueError("project archive MD5 does not match the Zenodo record")
    model_payload, raw_systems = read_archive(archive)
    if hashlib.sha256(model_payload).hexdigest() != MODEL_SHA256:
        raise ValueError("B3LYP(G&T) model SHA256 does not match the audited member")
    model = load_legacy_corrnet_bytes(model_payload).double().eval()
    if set(model.elem_dict) != {1, 6, 7, 8}:
        raise ValueError("B3LYP(G&T) model element table is not H/C/N/O")
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    systems_directory = output / "systems"
    reference_directory = output / "reference"
    systems_directory.mkdir(parents=True)
    reference_directory.mkdir()
    model_path = output / "b3lyp_gram_t1x.pth"
    model.save(
        model_path,
        legacy_import={
            "archive_md5": ARCHIVE_MD5,
            "member": MODEL_MEMBER,
            "member_sha256": MODEL_SHA256,
            "zenodo_record": ZENODO_RECORD,
        },
    )
    records = []
    for index, source in enumerate(SELECTED_SYSTEMS, start=1):
        raw = raw_systems[source]
        atom = load_array(raw["atom.npy"])
        if atom.ndim != 3 or atom.shape[0] != 1 or atom.shape[2] != 4:
            raise ValueError(f"invalid atom array for {source}")
        atom = atom[0]
        charges = np.rint(atom[:, 0]).astype(int)
        if not set(charges).issubset(model.elem_dict):
            raise ValueError(f"unsupported element in {source}")
        if int(charges.sum()) % 2:
            raise ValueError(f"neutral singlet electron parity is invalid for {source}")
        name = f"gram_{index:02d}_{Path(source).stem}"
        xyz_path = systems_directory / f"{name}.xyz"
        write_xyz(xyz_path, atom, source)
        np.savez_compressed(
            reference_directory / f"{name}.npz",
            atom=atom[np.newaxis],
            conv=load_array(raw["conv.npy"]),
            dm_eig=load_array(raw["dm_eig.npy"]),
            e_base=load_array(raw["e_base.npy"]),
            e_tot=load_array(raw["e_tot.npy"]),
            l_e_delta=load_array(raw["l_e_delta.npy"]),
        )
        metadata = raw["system.raw"].decode("utf-8").strip().splitlines()[-1].split()
        records.append(
            {
                "archive_directory": f"{DATA_PREFIX}/{source}",
                "atoms": int(len(atom)),
                "formula": formula(charges),
                "name": name,
                "neutral_electrons": int(charges.sum()),
                "state": {"charge": 0, "multiplicity": 1},
                "system_raw": [int(value) for value in metadata],
                "xyz": str(xyz_path.relative_to(output)),
            }
        )
    manifest = {
        "archive": {
            "filename": archive.name,
            "md5": ARCHIVE_MD5,
            "zenodo_record": ZENODO_RECORD,
        },
        "model": {
            "baseline": {
                "basis": "def2-tzvp",
                "family": "RKS",
                "published_name": "B3LYP",
                "xc": "B3LYP5",
            },
            "checkpoint": str(model_path.relative_to(output)),
            "elements": ["H", "C", "N", "O"],
            "member": MODEL_MEMBER,
            "member_sha256": MODEL_SHA256,
            "training_sets": ["GRAM", "Transition1x"],
        },
        "systems": records,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.archive, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
