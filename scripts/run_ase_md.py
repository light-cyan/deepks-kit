#!/usr/bin/env python
"""Run one GPU DeePHF NVE trajectory through ASE inside Slurm."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re

import numpy as np
from ase import units
from ase.io import read
from ase.io.trajectory import Trajectory
from ase.md.velocitydistribution import (
    Stationary,
    ZeroRotation,
    thermalize_momenta,
)
from ase.md.verlet import VelocityVerlet

from deepks.data.io import build_molecule
from deepks.deephf import (
    DeePHFCalculator,
    build_reference,
    make_deephf,
)
from deepks.gpu import require_cuda_device
from deepks.model.model import CorrNet


COMMENT_FIELD = re.compile(r"(?:^|\s)(charge|multiplicity)=([^\s]+)")


def xyz_state(path: Path) -> tuple[int, int]:
    """Return charge and multiplicity encoded in the XYZ comment line."""
    with path.open(encoding="utf-8") as stream:
        stream.readline()
        comment = stream.readline()
    fields = {name: value for name, value in COMMENT_FIELD.findall(comment)}
    missing = sorted({"charge", "multiplicity"} - set(fields))
    if missing:
        raise ValueError(
            f"XYZ comment is missing {', '.join(missing)}; provide explicit state options"
        )
    charge = int(fields["charge"])
    multiplicity = int(fields["multiplicity"])
    if multiplicity < 1:
        raise ValueError("multiplicity must be positive")
    return charge, multiplicity


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run an ASE NVE trajectory with GPU-native DeePHF forces."
    )
    parser.add_argument("xyz", type=Path)
    parser.add_argument("--model", default="NONE")
    parser.add_argument("--basis", required=True)
    parser.add_argument("--charge", type=int)
    parser.add_argument("--multiplicity", type=int)
    parser.add_argument("--temperature-k", type=float, default=100.0)
    parser.add_argument("--timestep-fs", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--root-overlap-tolerance", type=float, default=0.5)
    parser.add_argument("--scf-max-cycle", type=int, default=40)
    parser.add_argument("--scf-newton-max-cycle", type=int, default=50)
    parser.add_argument("--scf-conv-tol", type=float, default=1.0e-8)
    parser.add_argument("--scf-conv-tol-grad", type=float, default=1.0e-5)
    parser.add_argument("--scf-diis-space", type=int, default=12)
    parser.add_argument("--scf-level-shift", type=float, default=0.0)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--trajectory-interval", type=int, default=1)
    parser.add_argument("--verbose", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_arguments()
    require_cuda_device()
    if args.steps < 0:
        raise ValueError("steps must be nonnegative")
    if args.timestep_fs <= 0.0 or not np.isfinite(args.timestep_fs):
        raise ValueError("timestep-fs must be finite and positive")
    if args.temperature_k < 0.0 or not np.isfinite(args.temperature_k):
        raise ValueError("temperature-k must be finite and nonnegative")
    if args.trajectory_interval <= 0:
        raise ValueError("trajectory-interval must be positive")
    if args.scf_max_cycle <= 0 or args.scf_newton_max_cycle < 0:
        raise ValueError(
            "scf-max-cycle must be positive and scf-newton-max-cycle nonnegative"
        )
    if args.scf_diis_space <= 0:
        raise ValueError("scf-diis-space must be positive")
    if (
        args.scf_conv_tol <= 0.0
        or not np.isfinite(args.scf_conv_tol)
        or args.scf_conv_tol_grad <= 0.0
        or not np.isfinite(args.scf_conv_tol_grad)
    ):
        raise ValueError("SCF convergence tolerances must be finite and positive")
    if args.scf_level_shift < 0.0 or not np.isfinite(args.scf_level_shift):
        raise ValueError("scf-level-shift must be finite and nonnegative")

    encoded_charge, encoded_multiplicity = xyz_state(args.xyz)
    charge = encoded_charge if args.charge is None else args.charge
    multiplicity = (
        encoded_multiplicity
        if args.multiplicity is None
        else args.multiplicity
    )
    if multiplicity < 1:
        raise ValueError("multiplicity must be positive")
    spin = multiplicity - 1
    atoms = read(args.xyz, format="xyz")
    atoms.pbc = False
    molecule = build_molecule(
        atom=list(zip(atoms.get_chemical_symbols(), atoms.get_positions())),
        basis=args.basis,
        unit="Angstrom",
        charge=charge,
        spin=spin,
        symmetry=False,
        cart=False,
        verbose=args.verbose,
    )
    family = "rhf" if spin == 0 else "uhf"
    if str(args.model).upper() == "NONE":
        model = None
        projector_basis = None
    else:
        model = CorrNet.load(args.model, strict=True).double().eval()
        projector_basis = model._pbas
    reference = build_reference(
        molecule,
        family,
        scf_args={
            "max_cycle": args.scf_max_cycle,
            "newton_max_cycle": args.scf_newton_max_cycle,
            "conv_tol": args.scf_conv_tol,
            "conv_tol_grad": args.scf_conv_tol_grad,
            "diis_space": args.scf_diis_space,
            "level_shift": args.scf_level_shift,
        },
        verbose=args.verbose,
    )
    method = make_deephf(
        reference,
        model,
        projector_basis=projector_basis,
    )
    atoms.calc = DeePHFCalculator(
        method,
        root_overlap_tolerance=args.root_overlap_tolerance,
    )

    rng = np.random.default_rng(args.seed)
    thermalize_momenta(
        atoms,
        temperature_K=args.temperature_k,
        rng=rng,
    )
    Stationary(atoms)
    if np.all(atoms.get_moments_of_inertia() > 1.0e-12):
        ZeroRotation(atoms)

    args.output_directory.mkdir(parents=True, exist_ok=False)
    trajectory_path = args.output_directory / "trajectory.traj"
    energy_path = args.output_directory / "energy.csv"
    summary_path = args.output_directory / "summary.json"
    trajectory = Trajectory(trajectory_path, "w", atoms)
    dynamics = VelocityVerlet(
        atoms,
        timestep=args.timestep_fs * units.fs,
    )
    initial_total = None
    total_energies = []
    with energy_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "step",
                "time_fs",
                "potential_energy_eV",
                "kinetic_energy_eV",
                "total_energy_eV",
                "delta_total_energy_eV",
                "delta_total_energy_eV_per_atom",
                "temperature_K",
            )
        )

        def record_energy():
            nonlocal initial_total
            potential = float(atoms.get_potential_energy())
            kinetic = float(atoms.get_kinetic_energy())
            total = potential + kinetic
            if initial_total is None:
                initial_total = total
            delta = total - initial_total
            total_energies.append(total)
            writer.writerow(
                (
                    dynamics.nsteps,
                    dynamics.get_time() / units.fs,
                    potential,
                    kinetic,
                    total,
                    delta,
                    delta / len(atoms),
                    atoms.get_temperature(),
                )
            )
            stream.flush()

        dynamics.attach(record_energy, interval=1)
        dynamics.attach(trajectory.write, interval=args.trajectory_interval)
        dynamics.run(args.steps)
    trajectory.close()
    energies = np.asarray(total_energies, dtype=np.float64)
    times = np.arange(energies.size, dtype=np.float64) * args.timestep_fs
    drift = energies - energies[0]
    slope = (
        float(np.polyfit(times, drift, 1)[0])
        if energies.size >= 2 and times[-1] > 0.0
        else 0.0
    )
    summary = {
        "schema": {"id": "deepks.deephf.ase-nve", "version": 1},
        "input_xyz": str(args.xyz.resolve()),
        "reference_family": family.upper(),
        "charge": charge,
        "multiplicity": multiplicity,
        "basis": args.basis,
        "scf": {
            "conv_tol": args.scf_conv_tol,
            "conv_tol_grad": args.scf_conv_tol_grad,
            "diis_space": args.scf_diis_space,
            "level_shift": args.scf_level_shift,
            "max_cycle": args.scf_max_cycle,
            "newton_max_cycle": args.scf_newton_max_cycle,
        },
        "model": None if model is None else str(Path(args.model).resolve()),
        "temperature_K": args.temperature_k,
        "timestep_fs": args.timestep_fs,
        "steps": args.steps,
        "seed": args.seed,
        "maximum_absolute_energy_drift_eV": float(np.max(np.abs(drift))),
        "maximum_absolute_energy_drift_eV_per_atom": float(
            np.max(np.abs(drift)) / len(atoms)
        ),
        "linear_energy_drift_eV_per_fs": slope,
        "linear_energy_drift_eV_per_atom_per_fs": slope / len(atoms),
        "scanner_frames": atoms.calc.scanner.records,
    }
    summary_path.write_text(
        json.dumps(summary, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
