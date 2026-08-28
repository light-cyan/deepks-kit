#!/usr/bin/env python
"""Summarize NVE total-energy conservation and wall time for ASE runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ase.io.trajectory import Trajectory


ENERGY_STABILITY_MEV_PER_ATOM = 1.0
SYSTEM_NAME = re.compile(r"^(large|medium|small)_(\d+)")
PROTOCOL_FIELDS = (
    "reference_family",
    "xc",
    "grid_mode",
    "grid_level",
    "small_rho_cutoff",
    "basis",
    "model",
    "target_temperature_K",
    "timestep_fs",
    "steps",
)


def read_energy(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"no energy samples in {path}")
    return {
        name: np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        for name in rows[0]
    }


def system_label(directory: Path) -> str:
    match = SYSTEM_NAME.match(directory.name)
    return match.group(0) if match else directory.name


def stable_duration(times: np.ndarray, drift: np.ndarray) -> float:
    failed = np.flatnonzero(np.abs(drift) > ENERGY_STABILITY_MEV_PER_ATOM)
    if failed.size == 0:
        return float(times[-1])
    first = int(failed[0])
    return 0.0 if first == 0 else float(times[first - 1])


def validate_energy_series(
    energy: dict[str, np.ndarray],
    *,
    atoms: int,
    steps: int,
    timestep_fs: float,
) -> None:
    """Reject incomplete or internally inconsistent NVE energy records."""
    required = {
        "step",
        "time_fs",
        "potential_energy_eV",
        "kinetic_energy_eV",
        "total_energy_eV",
        "delta_total_energy_eV",
        "delta_total_energy_eV_per_atom",
        "temperature_K",
    }
    missing = sorted(required - set(energy))
    if missing:
        raise ValueError("energy record is missing columns: " + ", ".join(missing))
    expected_frames = steps + 1
    if atoms <= 0 or steps < 0 or timestep_fs <= 0.0:
        raise ValueError("trajectory metadata is invalid")
    if any(values.shape != (expected_frames,) for values in energy.values()):
        raise ValueError(
            f"energy record must contain exactly {expected_frames} complete frames"
        )
    if any(not np.isfinite(values).all() for values in energy.values()):
        raise ValueError("energy record contains nonfinite values")
    expected_steps = np.arange(expected_frames, dtype=np.float64)
    expected_times = expected_steps * timestep_fs
    if not np.array_equal(energy["step"], expected_steps):
        raise ValueError("energy record step indices are incomplete or unordered")
    if not np.allclose(energy["time_fs"], expected_times, rtol=0.0, atol=1.0e-12):
        raise ValueError("energy record times do not match the integration timestep")
    total = energy["potential_energy_eV"] + energy["kinetic_energy_eV"]
    if not np.allclose(energy["total_energy_eV"], total, rtol=0.0, atol=1.0e-10):
        raise ValueError("recorded total energy is not potential plus kinetic energy")
    drift = total - total[0]
    if not np.allclose(
        energy["delta_total_energy_eV"], drift, rtol=0.0, atol=1.0e-10
    ):
        raise ValueError("recorded total-energy drift is inconsistent")
    if not np.allclose(
        energy["delta_total_energy_eV_per_atom"],
        drift / atoms,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError("recorded per-atom total-energy drift is inconsistent")


def validate_common_protocol(results: list[dict]) -> None:
    """Require every trajectory in one report to use one common protocol."""
    if not results:
        raise ValueError("energy stability analysis requires completed trajectories")
    expected = {field: results[0][field] for field in PROTOCOL_FIELDS}
    for result in results[1:]:
        mismatched = [
            field for field in PROTOCOL_FIELDS if result[field] != expected[field]
        ]
        if mismatched:
            raise ValueError(
                f"system {result['system']} uses a different protocol: "
                + ", ".join(mismatched)
            )


def analyze_system(directory: Path) -> tuple[dict, dict[str, np.ndarray]]:
    energy = read_energy(directory / "energy.csv")
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    trajectory = Trajectory(directory / "trajectory.traj")
    initial_atoms = trajectory[0]
    atoms = len(initial_atoms)
    formula = initial_atoms.get_chemical_formula(mode="hill")
    trajectory_frames = len(trajectory)
    trajectory.close()
    validate_energy_series(
        energy,
        atoms=atoms,
        steps=summary["steps"],
        timestep_fs=summary["timestep_fs"],
    )
    if trajectory_frames != summary["steps"] + 1:
        raise ValueError(
            f"trajectory contains {trajectory_frames} frames, expected "
            f"{summary['steps'] + 1}"
        )
    times = energy["time_fs"]
    drift = 1000.0 * energy["delta_total_energy_eV_per_atom"]
    duration = float(times[-1])
    stable_for = stable_duration(times, drift)
    slope = (
        float(np.polyfit(times, drift, 1)[0])
        if times.size >= 2 and duration > 0.0
        else 0.0
    )
    timing = summary["timing"]
    result = {
        "system": system_label(directory),
        "formula": formula,
        "atoms": atoms,
        "charge": summary["charge"],
        "multiplicity": summary["multiplicity"],
        "reference_family": summary["reference_family"],
        "xc": None if summary["dft"] is None else summary["dft"]["xc"],
        "grid_mode": None if summary["dft"] is None else summary["dft"]["grid_mode"],
        "grid_level": None if summary["dft"] is None else summary["dft"]["grid_level"],
        "small_rho_cutoff": (
            None if summary["dft"] is None else summary["dft"]["small_rho_cutoff"]
        ),
        "basis": summary["basis"],
        "model": summary["model"],
        "target_temperature_K": summary["temperature_K"],
        "initial_temperature_K": float(energy["temperature_K"][0]),
        "timestep_fs": summary["timestep_fs"],
        "steps": summary["steps"],
        "slurm_job_id": summary["slurm"]["job_id"],
        "simulated_duration_fs": duration,
        "stable_duration_at_1meV_per_atom_fs": stable_for,
        "stable_through_full_run": stable_for == duration,
        "maximum_absolute_drift_meV_per_atom": float(np.max(np.abs(drift))),
        "rms_drift_meV_per_atom": float(np.sqrt(np.mean(drift * drift))),
        "final_drift_meV_per_atom": float(drift[-1]),
        "linear_drift_meV_per_atom_per_fs": slope,
        "initialization_wall_time_seconds": timing[
            "initialization_wall_time_seconds"
        ],
        "md_wall_time_seconds": timing["md_wall_time_seconds"],
        "total_wall_time_seconds": timing["total_wall_time_seconds"],
        "md_wall_seconds_per_simulated_fs": timing[
            "md_wall_seconds_per_simulated_fs"
        ],
        "total_wall_seconds_per_simulated_fs": timing[
            "total_wall_seconds_per_simulated_fs"
        ],
    }
    return result, {"time_fs": times, "drift_meV_per_atom": drift}


def plot_energy(results: list[dict], series: list[dict], output: Path) -> None:
    figure, axes = plt.subplots(3, 3, figsize=(14, 10), constrained_layout=True)
    for axis, result, values in zip(axes.flat, results, series, strict=True):
        axis.plot(
            values["time_fs"],
            values["drift_meV_per_atom"],
            color="#1565c0",
            linewidth=1.5,
        )
        axis.axhspan(
            -ENERGY_STABILITY_MEV_PER_ATOM,
            ENERGY_STABILITY_MEV_PER_ATOM,
            color="#2e7d32",
            alpha=0.12,
            label="stable band",
        )
        axis.axhline(0.0, color="#424242", linewidth=0.7)
        if not result["stable_through_full_run"]:
            axis.axvline(
                result["stable_duration_at_1meV_per_atom_fs"],
                color="#c62828",
                linestyle="--",
                linewidth=1.0,
            )
        axis.set_title(
            f"{result['system']} ({result['formula']}, {result['atoms']} atoms)"
        )
        axis.set_xlabel("Simulation time (fs)")
        axis.set_ylabel("Total-energy drift (meV/atom)")
        axis.grid(alpha=0.25)
        axis.text(
            0.03,
            0.95,
            f"max |drift| = {result['maximum_absolute_drift_meV_per_atom']:.3f}",
            transform=axis.transAxes,
            va="top",
            fontsize=9,
        )
    figure.suptitle(
        "NVE total-energy conservation (green band: +/-1 meV/atom)",
        fontsize=14,
    )
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_timing(results: list[dict], output: Path) -> None:
    labels = [result["system"] for result in results]
    md = np.asarray(
        [result["md_wall_seconds_per_simulated_fs"] for result in results]
    )
    total = np.asarray(
        [result["total_wall_seconds_per_simulated_fs"] for result in results]
    )
    positions = np.arange(len(results))
    width = 0.38
    figure, axis = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    axis.bar(positions - width / 2, md, width, label="MD loop", color="#1976d2")
    axis.bar(
        positions + width / 2,
        total,
        width,
        label="End-to-end",
        color="#ef6c00",
    )
    axis.set_xticks(positions, labels, rotation=35, ha="right")
    axis.set_ylabel("Wall time / simulated time (s/fs)")
    axis.set_title("GPU Slurm wall time per simulated femtosecond")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def write_report(results: list[dict], output: Path) -> None:
    validate_common_protocol(results)
    protocol = results[0]
    model_name = Path(protocol["model"]).name
    lines = [
        "# DeePHF NVE total-energy stability",
        "",
        f"All trajectories use analytic DeePHF forces with a GPU4PySCF {protocol['reference_family']} {protocol['xc']}/{protocol['basis']} reference and the {model_name} correction network.",
        "",
        "A trajectory is classified as stable through the last sampled frame before its absolute total-energy drift first exceeds 1 meV/atom. This criterion uses total energy only; potential and kinetic energies are recorded as components but are not separate stability criteria.",
        "",
        "## Simulation tasks and methods",
        "",
        "| System | Formula | Atoms | Charge | Multiplicity | Reference | XC | Basis | Grid | Correction model | Target T (K) | Frame-0 T (K) | Step (fs) | Steps | Slurm job |",
        "|---|---:|---:|---:|---:|---|---|---|---|---|---:|---:|---:|---:|---|",
    ]
    for result in results:
        model = Path(result["model"]).name
        grid = (
            f"{result['grid_mode']}/level-{result['grid_level']}"
            f"/rho-{result['small_rho_cutoff']:g}"
        )
        lines.append(
            f"| {result['system']} | {result['formula']} | {result['atoms']} | "
            f"{result['charge']} | {result['multiplicity']} | "
            f"GPU4PySCF {result['reference_family']} | {result['xc']} | "
            f"{result['basis']} | {grid} | {model} | "
            f"{result['target_temperature_K']:.1f} | "
            f"{result['initial_temperature_K']:.1f} | "
            f"{result['timestep_fs']:.3f} | "
            f"{result['steps']} | {result['slurm_job_id']} |"
        )
    lines.extend(
        [
            "",
            "## Total-energy results",
            "",
            "| System | Formula | Atoms | Simulated (fs) | Stable at 1 meV/atom (fs) | Max drift (meV/atom) | RMS drift (meV/atom) | Final drift (meV/atom) | Linear drift (meV/atom/fs) | MD wall time (s/fs) | End-to-end wall time (s/fs) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for result in results:
        stable = result["stable_duration_at_1meV_per_atom_fs"]
        if result["stable_through_full_run"]:
            stable_text = f">={stable:.2f}"
        else:
            stable_text = f"{stable:.2f}"
        lines.append(
            f"| {result['system']} | {result['formula']} | {result['atoms']} | "
            f"{result['simulated_duration_fs']:.2f} | {stable_text} | "
            f"{result['maximum_absolute_drift_meV_per_atom']:.6f} | "
            f"{result['rms_drift_meV_per_atom']:.6f} | "
            f"{result['final_drift_meV_per_atom']:.6f} | "
            f"{result['linear_drift_meV_per_atom_per_fs']:.6g} | "
            f"{result['md_wall_seconds_per_simulated_fs']:.3f} | "
            f"{result['total_wall_seconds_per_simulated_fs']:.3f} |"
        )
    lines.extend(
        [
            "",
            "![Total-energy stability](total_energy_stability.png)",
            "",
            "![Wall time per simulated femtosecond](wall_time_per_fs.png)",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    directories = sorted(
        path for path in args.root.iterdir() if (path / "summary.json").is_file()
    )
    if len(directories) != 9:
        raise ValueError(f"expected 9 completed systems, found {len(directories)}")
    analyses = [analyze_system(directory) for directory in directories]
    results = [item[0] for item in analyses]
    series = [item[1] for item in analyses]
    validate_common_protocol(results)
    (args.root / "energy_stability_summary.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (args.root / "energy_stability_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(results[0]))
        writer.writeheader()
        writer.writerows(results)
    plot_energy(results, series, args.root / "total_energy_stability.png")
    plot_timing(results, args.root / "wall_time_per_fs.png")
    write_report(results, args.root / "energy_stability_report.md")
    print(json.dumps(results, sort_keys=True))


if __name__ == "__main__":
    main()
