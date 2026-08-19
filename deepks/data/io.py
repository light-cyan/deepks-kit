"""Method-neutral molecular input and array output operations."""

import os

import numpy as np
from pyscf import gto

from deepks.utils import get_sys_name, get_with_prefix, is_xyz, load_array


MOLECULE_ATTRIBUTES = {"basis", "charge", "spin", "unit"}


def iter_system(path, labels=None):
    """Yield atoms, molecule attributes, and requested labels for each frame."""
    labels = set() if labels is None else set(labels)
    base = get_sys_name(path)
    attribute_paths = {
        attribute: get_with_prefix(attribute, base, ".npy", True)
        for attribute in MOLECULE_ATTRIBUTES
    }
    attribute_paths = {
        name: value
        for name, value in attribute_paths.items()
        if value is not None
    }
    label_paths = {
        label: get_with_prefix(label, base, prefer=".npy")
        for label in labels
    }
    if is_xyz(path):
        attributes = {
            name: load_array(value)
            for name, value in attribute_paths.items()
        }
        attributes.setdefault("unit", "Angstrom")
        frame_labels = {
            label: load_array(value) for label, value in label_paths.items()
        }
        yield path, attributes, frame_labels
        return
    if not os.path.isdir(path):
        raise ValueError(f"system {path} is neither an XYZ file nor a directory")
    all_attributes = {
        name: load_array(value) for name, value in attribute_paths.items()
    }
    all_labels = {
        label: load_array(value) for label, value in label_paths.items()
    }
    try:
        atom_array = load_array(get_with_prefix("atom", path, prefer=".npy"))
        if atom_array.ndim != 3 or atom_array.shape[2] != 4:
            raise ValueError("atom data must have shape (frame, atom, 4)")
        frame_count = atom_array.shape[0]
        elements = np.rint(atom_array[:, :, 0]).astype(int)
        coordinates = atom_array[:, :, 1:]
    except FileNotFoundError:
        coordinates = load_array(get_with_prefix("coord", path, prefer=".npy"))
        if coordinates.ndim != 3 or coordinates.shape[2] != 3:
            raise ValueError("coordinate data must have shape (frame, atom, 3)")
        frame_count = coordinates.shape[0]
        elements = np.loadtxt(
            os.path.join(path, "type.raw"),
            dtype=str,
        ).reshape(1, -1).repeat(frame_count, axis=0)
    for frame_index in range(frame_count):
        atoms = [
            [element, coordinate]
            for element, coordinate in zip(
                elements[frame_index],
                coordinates[frame_index],
            )
        ]
        attributes = {
            name: (
                values[frame_index]
                if values.ndim > 0 and values.shape[0] == frame_count
                else values
            )
            for name, values in all_attributes.items()
        }
        frame_labels = {
            label: values[frame_index]
            for label, values in all_labels.items()
        }
        yield atoms, attributes, frame_labels


def build_molecule(
    atom,
    basis="ccpvdz",
    unit="Bohr",
    spin=None,
    verbose=0,
    **kwargs,
):
    """Build a PySCF molecule while preserving an explicitly supplied spin."""
    mol = gto.Mole()
    mol.unit = unit.tolist() if isinstance(unit, np.ndarray) else unit
    mol.atom = atom
    mol.basis = basis
    mol.verbose = verbose
    mol.set(**kwargs)
    mol.spin = mol.nelectron % 2 if spin is None else int(np.asarray(spin))
    mol.build(0, 0)
    return mol


def collect_field_results(fields, metadata, results):
    """Stack per-frame field results using the declared field shapes."""
    if isinstance(fields, dict):
        fields = sum(fields.values(), [])
    if isinstance(results, dict):
        results = [results]
    frame_count = len(results)
    descriptor_atoms, raw_atoms, nao, descriptor_features = metadata
    local_shapes = {
        "nframe": frame_count,
        "natom": descriptor_atoms,
        "natom_raw": raw_atoms,
        "nao": nao,
        "nproj": descriptor_features,
    }
    collected = {}
    for field in fields:
        values = np.array([result[field.name] for result in results])
        if field.shape:
            values = values.reshape(eval(field.shape, {}, local_shapes))
        collected[field.name] = values
    return collected


def dump_metadata(directory, metadata):
    os.makedirs(directory, exist_ok=True)
    np.savetxt(
        os.path.join(directory, "system.raw"),
        np.reshape(metadata, (1, -1)),
        fmt="%d",
        header="natom natom_raw nao nproj",
    )


def dump_data(directory, **data):
    os.makedirs(directory, exist_ok=True)
    for name, value in data.items():
        np.save(os.path.join(directory, f"{name}.npy"), value)
