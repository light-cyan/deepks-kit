#!/usr/bin/env python
"""Compare analytic and total-energy finite-difference DeePHF NVE runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if __package__:
    from .analyze_energy_stability import (
        ENERGY_STABILITY_MEV_PER_ATOM,
        analyze_system,
        validate_common_protocol,
    )
else:
    from analyze_energy_stability import (
        ENERGY_STABILITY_MEV_PER_ATOM,
        analyze_system,
        validate_common_protocol,
    )


PAIR_PROTOCOL_FIELDS = (
    "formula",
    "atoms",
    "charge",
    "multiplicity",
    "reference_family",
    "xc",
    "grid_mode",
    "grid_level",
    "small_rho_cutoff",
    "basis",
    "model_name",
    "target_temperature_K",
    "timestep_fs",
)


def read_campaign(root: Path) -> dict[str, tuple[dict, dict]]:
    directories = sorted(
        path for path in root.iterdir() if (path / "summary.json").is_file()
    )
    if not directories:
        raise ValueError(f"no completed trajectories in {root}")
    analyses = [analyze_system(directory) for directory in directories]
    validate_common_protocol([result for result, _series in analyses])
    return {
        result["system"]: (result, series) for result, series in analyses
    }


def validate_campaign_pair(
    analytic: dict[str, tuple[dict, dict]],
    numerical: dict[str, tuple[dict, dict]],
) -> list[str]:
    if set(analytic) != set(numerical):
        missing_numerical = sorted(set(analytic) - set(numerical))
        missing_analytic = sorted(set(numerical) - set(analytic))
        raise ValueError(
            "campaign systems differ; missing numerical="
            f"{missing_numerical}, missing analytic={missing_analytic}"
        )
    names = sorted(analytic)
    if analytic[names[0]][0]["force_mode"] != "analytic":
        raise ValueError("the analytic campaign does not use analytic forces")
    if numerical[names[0]][0]["force_mode"] != "central_finite_difference":
        raise ValueError(
            "the numerical campaign does not use central finite differences"
        )
    for name in names:
        analytic_result = analytic[name][0]
        numerical_result = numerical[name][0]
        mismatched = [
            field
            for field in PAIR_PROTOCOL_FIELDS
            if analytic_result[field] != numerical_result[field]
        ]
        if mismatched:
            raise ValueError(
                f"system {name} differs between force methods: "
                + ", ".join(mismatched)
            )
        if not np.isclose(
            analytic_result["initial_temperature_K"],
            numerical_result["initial_temperature_K"],
            rtol=0.0,
            atol=1.0e-10,
        ):
            raise ValueError(f"system {name} does not share the same initial momenta")
    return names


def _common_drift(analytic_series: dict, numerical_series: dict):
    frames = min(
        analytic_series["time_fs"].size,
        numerical_series["time_fs"].size,
    )
    analytic_time = analytic_series["time_fs"][:frames]
    numerical_time = numerical_series["time_fs"][:frames]
    if not np.allclose(analytic_time, numerical_time, rtol=0.0, atol=1.0e-12):
        raise ValueError("paired trajectory times do not share a common grid")
    return (
        analytic_time,
        analytic_series["drift_meV_per_atom"][:frames],
        numerical_series["drift_meV_per_atom"][:frames],
    )


def compare_campaigns(
    analytic: dict[str, tuple[dict, dict]],
    numerical: dict[str, tuple[dict, dict]],
) -> tuple[list[dict], list[dict]]:
    names = validate_campaign_pair(analytic, numerical)
    rows = []
    common_series = []
    for name in names:
        analytic_result, analytic_series = analytic[name]
        numerical_result, numerical_series = numerical[name]
        times, analytic_drift, numerical_drift = _common_drift(
            analytic_series, numerical_series
        )
        analytic_seconds = analytic_result["md_wall_seconds_per_simulated_fs"]
        numerical_seconds = numerical_result["md_wall_seconds_per_simulated_fs"]
        rows.append(
            {
                "system": name,
                "formula": analytic_result["formula"],
                "atoms": analytic_result["atoms"],
                "common_duration_fs": float(times[-1]),
                "analytic_duration_fs": analytic_result["simulated_duration_fs"],
                "numerical_duration_fs": numerical_result[
                    "simulated_duration_fs"
                ],
                "analytic_stable_duration_fs": analytic_result[
                    "stable_duration_at_1meV_per_atom_fs"
                ],
                "numerical_stable_duration_fs": numerical_result[
                    "stable_duration_at_1meV_per_atom_fs"
                ],
                "analytic_common_max_drift_meV_per_atom": float(
                    np.max(np.abs(analytic_drift))
                ),
                "numerical_common_max_drift_meV_per_atom": float(
                    np.max(np.abs(numerical_drift))
                ),
                "analytic_common_rms_drift_meV_per_atom": float(
                    np.sqrt(np.mean(analytic_drift * analytic_drift))
                ),
                "numerical_common_rms_drift_meV_per_atom": float(
                    np.sqrt(np.mean(numerical_drift * numerical_drift))
                ),
                "analytic_md_wall_seconds_per_fs": analytic_seconds,
                "numerical_md_wall_seconds_per_fs": numerical_seconds,
                "numerical_over_analytic_wall_time_ratio": (
                    numerical_seconds / analytic_seconds
                ),
                "finite_difference_step_bohr": numerical_result[
                    "finite_difference_step_bohr"
                ],
            }
        )
        common_series.append(
            {
                "system": name,
                "time_fs": times,
                "analytic_drift_meV_per_atom": analytic_drift,
                "numerical_drift_meV_per_atom": numerical_drift,
            }
        )
    return rows, common_series


def plot_energy_comparison(rows: list[dict], series: list[dict], output: Path):
    columns = min(3, len(rows))
    row_count = math.ceil(len(rows) / columns)
    figure, axes = plt.subplots(
        row_count,
        columns,
        figsize=(4.8 * columns, 3.4 * row_count),
        squeeze=False,
        constrained_layout=True,
    )
    for axis, row, values in zip(axes.flat, rows, series, strict=False):
        axis.plot(
            values["time_fs"],
            values["analytic_drift_meV_per_atom"],
            label="analytic",
            color="#1565c0",
            linewidth=1.4,
        )
        axis.plot(
            values["time_fs"],
            values["numerical_drift_meV_per_atom"],
            label="central FD",
            color="#ef6c00",
            linewidth=1.2,
        )
        axis.axhspan(
            -ENERGY_STABILITY_MEV_PER_ATOM,
            ENERGY_STABILITY_MEV_PER_ATOM,
            color="#2e7d32",
            alpha=0.10,
        )
        axis.axhline(0.0, color="#424242", linewidth=0.7)
        axis.set_title(f"{row['system']} ({row['formula']})")
        axis.set_xlabel("Simulation time (fs)")
        axis.set_ylabel("Total-energy drift (meV/atom)")
        axis.grid(alpha=0.25)
        axis.legend()
    for axis in axes.flat[len(rows) :]:
        axis.set_visible(False)
    figure.suptitle("Analytic versus numerical DeePHF force stability")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_timing_comparison(rows: list[dict], output: Path):
    labels = [row["system"] for row in rows]
    analytic = np.asarray(
        [row["analytic_md_wall_seconds_per_fs"] for row in rows]
    )
    numerical = np.asarray(
        [row["numerical_md_wall_seconds_per_fs"] for row in rows]
    )
    positions = np.arange(len(rows))
    width = 0.38
    figure, axis = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    axis.bar(
        positions - width / 2,
        analytic,
        width,
        label="analytic",
        color="#1565c0",
    )
    axis.bar(
        positions + width / 2,
        numerical,
        width,
        label="central FD",
        color="#ef6c00",
    )
    axis.set_xticks(positions, labels, rotation=35, ha="right")
    axis.set_ylabel("MD wall time / simulated time (s/fs)")
    axis.set_title("DeePHF force-method wall time")
    if numerical.max() / analytic.min() > 20.0:
        axis.set_yscale("log")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def write_report(rows: list[dict], output: Path) -> None:
    step = rows[0]["finite_difference_step_bohr"]
    lines = [
        "# DeePHF force-method comparison",
        "",
        "The analytic method evaluates the complete DeePHF nuclear derivative. The numerical control evaluates the same complete DeePHF total energy at every positive and negative Cartesian displacement and applies a central difference.",
        "",
        f"The numerical displacement is {step:.1e} Bohr. Both methods use identical structures, initial momenta, electronic reference, correction network, SCF controls, temperature, and integration timestep.",
        "",
        "| System | Formula | Atoms | Common duration (fs) | Analytic max drift (meV/atom) | Numerical max drift (meV/atom) | Analytic time (s/fs) | Numerical time (s/fs) | Numerical / analytic |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['system']} | {row['formula']} | {row['atoms']} | "
            f"{row['common_duration_fs']:.2f} | "
            f"{row['analytic_common_max_drift_meV_per_atom']:.6f} | "
            f"{row['numerical_common_max_drift_meV_per_atom']:.6f} | "
            f"{row['analytic_md_wall_seconds_per_fs']:.3f} | "
            f"{row['numerical_md_wall_seconds_per_fs']:.3f} | "
            f"{row['numerical_over_analytic_wall_time_ratio']:.2f}x |"
        )
    lines.extend(
        [
            "",
            "![Total-energy comparison](force_method_energy_comparison.png)",
            "",
            "![Wall-time comparison](force_method_timing_comparison.png)",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("analytic_root", type=Path)
    parser.add_argument("numerical_root", type=Path)
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    analytic = read_campaign(args.analytic_root)
    numerical = read_campaign(args.numerical_root)
    rows, series = compare_campaigns(analytic, numerical)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "force_method_comparison.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (args.output_root / "force_method_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_energy_comparison(
        rows,
        series,
        args.output_root / "force_method_energy_comparison.png",
    )
    plot_timing_comparison(
        rows,
        args.output_root / "force_method_timing_comparison.png",
    )
    write_report(rows, args.output_root / "force_method_comparison.md")
    print(json.dumps(rows, sort_keys=True))


if __name__ == "__main__":
    main()
