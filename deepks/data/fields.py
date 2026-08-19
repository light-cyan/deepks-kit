"""Canonical calculation fields exposed through stable method outputs."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import numpy as np

from deepks.utils import check_list


@dataclass(frozen=True)
class Field:
    name: str
    calculate: Callable
    shape: str
    required_labels: tuple[str, ...] = field(default_factory=tuple)


BOHR_TO_ANGSTROM = 0.52917721092


def coordinate_unit_scale(mol) -> float:
    return (
        1.0
        if mol.unit.upper().startswith(("B", "AU"))
        else BOHR_TO_ANGSTROM
    )


def atom_data(mol) -> np.ndarray:
    values = np.concatenate(
        [
            mol.atom_charges().reshape(-1, 1),
            mol.atom_coords(unit="Bohr"),
        ],
        axis=1,
    )
    return values[mol.atom_charges() != 0]


SCF_FIELDS = (
    Field("atom", lambda method: atom_data(method.mol), "(nframe, natom, 4)"),
    Field("e_base", lambda method: method.reference_energy(), "(nframe, 1)"),
    Field(
        "e_corr",
        lambda method: method.e_tot - method.reference_energy(),
        "(nframe, 1)",
    ),
    Field("e_tot", lambda method: method.e_tot, "(nframe, 1)"),
    Field("ao_density", lambda method: method.make_rdm1(), "(nframe, nao, nao)"),
    Field(
        "projected_density",
        lambda method: method.projected_density(flatten=True),
        "(nframe, natom, -1)",
    ),
    Field(
        "descriptor",
        lambda method: method.descriptor(),
        "(nframe, natom, nproj)",
    ),
    Field(
        "hcore_descriptor",
        lambda method: method.descriptor(method.get_hcore()),
        "(nframe, natom, nproj)",
    ),
    Field(
        "overlap_descriptor",
        lambda method: method.descriptor(method.get_ovlp()),
        "(nframe, natom, nproj)",
    ),
    Field(
        "effective_potential_descriptor",
        lambda method: method.descriptor(method.get_veff()),
        "(nframe, natom, nproj)",
    ),
    Field(
        "fock_descriptor",
        lambda method: method.descriptor(method.get_fock()),
        "(nframe, natom, nproj)",
    ),
    Field("converged", lambda method: method.converged, "(nframe, 1)"),
    Field(
        "occupied_mo_coeff",
        lambda method: method.mo_coeff[:, method.mo_occ > 0].T,
        "(nframe, -1, nao)",
    ),
    Field(
        "occupied_mo_energy",
        lambda method: method.mo_energy[method.mo_occ > 0],
        "(nframe, -1)",
    ),
    Field(
        "e_target",
        lambda method, **labels: labels["energy"],
        "(nframe, 1)",
        ("energy",),
    ),
    Field(
        "e_corr_target",
        lambda method, **labels: labels["energy"] - method.reference_energy(),
        "(nframe, 1)",
        ("energy",),
    ),
    Field(
        "e_error",
        lambda method, **labels: labels["energy"] - method.e_tot,
        "(nframe, 1)",
        ("energy",),
    ),
    Field(
        "descriptor_orbital_gradient_jacobian",
        lambda method: method.descriptor_orbital_gradient_jacobian(),
        "(nframe, natom, nproj, -1)",
    ),
    Field(
        "reference_orbital_gradient",
        lambda method: method.reference_orbital_gradient(),
        "(nframe, -1)",
    ),
    Field(
        "coulomb_loss_descriptor_gradient",
        lambda method, **labels: method.coulomb_loss_descriptor_gradient(
            labels["dm"]
        ),
        "(nframe, natom, nproj)",
        ("dm",),
    ),
    Field(
        "optimized_descriptor_potential_raw",
        lambda method, **labels: method.optimize_descriptor_potential(
            labels["dm"],
            nstep=1,
        ),
        "(nframe, natom, nproj)",
        ("dm",),
    ),
)


GRADIENT_FIELDS = (
    Field(
        "f_reference_variational",
        lambda gradient: -gradient.reference_gradient()
        / coordinate_unit_scale(gradient.mol),
        "(nframe, natom_raw, 3)",
    ),
    Field(
        "f_tot",
        lambda gradient: -gradient.de / coordinate_unit_scale(gradient.mol),
        "(nframe, natom_raw, 3)",
    ),
    Field(
        "f_corr_explicit",
        lambda gradient: -gradient.explicit_correction_gradient
        / coordinate_unit_scale(gradient.mol),
        "(nframe, natom_raw, 3)",
    ),
    Field(
        "dD_dR_explicit",
        lambda gradient: gradient.dD_dR_explicit(flatten=True)
        / coordinate_unit_scale(gradient.mol),
        "(nframe, natom_raw, 3, natom, -1)",
    ),
    Field(
        "dq_dR_explicit",
        lambda gradient: gradient.dq_dR_explicit()
        / coordinate_unit_scale(gradient.mol),
        "(nframe, natom_raw, 3, natom, nproj)",
    ),
    Field(
        "f_target",
        lambda gradient, **labels: labels["force"],
        "(nframe, natom_raw, 3)",
        ("force",),
    ),
    Field(
        "f_corr_explicit_target",
        lambda gradient, **labels: labels["force"]
        - (
            -gradient.reference_gradient()
            / coordinate_unit_scale(gradient.mol)
        ),
        "(nframe, natom_raw, 3)",
        ("force",),
    ),
    Field(
        "f_error",
        lambda gradient, **labels: labels["force"]
        - (-gradient.de / coordinate_unit_scale(gradient.mol)),
        "(nframe, natom_raw, 3)",
        ("force",),
    ),
    Field(
        "optimized_descriptor_potential",
        lambda gradient, **labels: gradient.optimize_descriptor_potential(
            labels["dm"],
            -coordinate_unit_scale(gradient.mol) * labels["force"],
            nstep=1,
        ),
        "(nframe, natom, nproj)",
        ("dm", "force"),
    ),
    Field(
        "optimized_descriptor_potential_without_force",
        lambda gradient, **labels: gradient.optimize_descriptor_potential(
            labels["dm"],
            gradient.de,
            nstep=1,
        ),
        "(nframe, natom, nproj)",
        ("dm",),
    ),
)


def select_fields(names: Iterable[str]):
    """Select canonical fields and reject every unknown name."""
    names = tuple(names)
    if len(set(names)) != len(names):
        raise ValueError("calculation field names must be unique")
    known = {field.name for field in SCF_FIELDS + GRADIENT_FIELDS}
    unknown = sorted(set(names) - known)
    if unknown:
        raise ValueError(f"unknown calculation fields: {', '.join(unknown)}")
    return {
        "scf": [field for field in SCF_FIELDS if field.name in names],
        "gradient": [field for field in GRADIENT_FIELDS if field.name in names],
    }


def required_field_labels(fields=None):
    """Return the labels required by a collection of selected fields."""
    labels = [
        check_list(calculation_field.required_labels)
        for calculation_field in check_list(fields)
    ]
    return set(sum(labels, []))
