"""ASE calculator for GPU-native DeePHF energy and analytic forces."""

from __future__ import annotations

import numpy as np
from ase import units
from ase.calculators.calculator import (
    CalculationFailed,
    Calculator,
    all_changes,
)

from .gpu_method import GPUDeePHF


class DeePHFCalculator(Calculator):
    """Expose one continuous GPU DeePHF electronic root to ASE."""

    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        method,
        *,
        backend="direct",
        root_overlap_tolerance=0.5,
        **kwargs,
    ):
        if not isinstance(method, GPUDeePHF):
            raise TypeError("DeePHFCalculator requires a GPUDeePHF method")
        super().__init__(**kwargs)
        self.method = method
        self.template_molecule = method.mol.copy()
        self.atomic_numbers = np.asarray(
            self.template_molecule.atom_charges(), dtype=np.int64
        )
        self.scanner = method.nuc_grad_method(
            backend=backend,
            retain_details=False,
        ).as_scanner(root_overlap_tolerance=root_overlap_tolerance)

    def calculate(
        self,
        atoms=None,
        properties=("energy", "forces"),
        system_changes=all_changes,
    ):
        super().calculate(atoms, properties, system_changes)
        if atoms is None:
            raise CalculationFailed("DeePHF calculation requires ASE atoms")
        if bool(np.any(atoms.get_pbc())):
            raise CalculationFailed("molecular DeePHF does not support periodic ASE atoms")
        numbers = np.asarray(atoms.get_atomic_numbers(), dtype=np.int64)
        if not np.array_equal(numbers, self.atomic_numbers):
            raise CalculationFailed(
                "the ASE atomic numbers changed within one DeePHF trajectory"
            )
        molecule = self.template_molecule.set_geom_(
            atoms.get_positions(),
            unit="Angstrom",
            inplace=False,
        )
        try:
            energy_hartree, gradient_hartree_per_bohr = self.scanner(molecule)
        except Exception as error:
            raise CalculationFailed(f"DeePHF frame failed: {error}") from error
        energy = float(energy_hartree) * units.Hartree
        forces = (
            -np.asarray(gradient_hartree_per_bohr, dtype=np.float64)
            * units.Hartree
            / units.Bohr
        )
        if not np.isfinite(energy) or not np.isfinite(forces).all():
            raise CalculationFailed("DeePHF returned nonfinite ASE properties")
        self.results = {
            "energy": energy,
            "forces": forces,
        }


__all__ = ["DeePHFCalculator"]
