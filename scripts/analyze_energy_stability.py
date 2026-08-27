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


def analyze_system(directory: Path) -> tuple[dict, dict[str, np.ndarray]]:
    energy = read_energy(directory / "energy.csv")
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    trajectory = Trajectory(directory / "trajectory.traj")
    atoms = len(trajectory[0])
    trajectory.close()
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
        "atoms": atoms,
        "multiplicity": summary["multiplicity"],
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
        axis.set_title(
            f"{result['system']} ({result['atoms']} atoms, M={result['multiplicity']})"
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
    print(json.dumps(results, sort_keys=True))


if __name__ == "__main__":
    main()
