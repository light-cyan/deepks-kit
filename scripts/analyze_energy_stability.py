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


plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Droid Sans Fallback", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


ENERGY_STABILITY_MEV_PER_ATOM = 1.0
SYSTEM_NAME = re.compile(r"^(?:(?:large|medium|small)|gram)_\d+")
PROTOCOL_FIELDS = (
    "reference_family",
    "xc",
    "grid_mode",
    "grid_level",
    "small_rho_cutoff",
    "basis",
    "model_name",
    "force_mode",
    "finite_difference_step_bohr",
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


def drift_axis_limit(*drift_series: np.ndarray) -> float:
    """Return a symmetric plot limit that resolves the observed energy drift."""
    maximum = max(
        (float(np.max(np.abs(values), initial=0.0)) for values in drift_series),
        default=0.0,
    )
    return max(0.005, 1.15 * maximum)


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
    force = summary.get("force", {"mode": "analytic"})
    result = {
        "system": system_label(directory),
        "configuration": directory.name,
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
        "model_name": (
            None if summary["model"] is None else Path(summary["model"]).name
        ),
        "force_mode": force["mode"],
        "finite_difference_step_bohr": force.get(
            "finite_difference_step_bohr"
        ),
        "target_temperature_K": summary["temperature_K"],
        "initial_temperature_K": float(energy["temperature_K"][0]),
        "timestep_fs": summary["timestep_fs"],
        "steps": summary["steps"],
        "slurm_job_id": summary["slurm"]["job_id"],
        "simulated_duration_fs": duration,
        "stable_duration_at_1meV_per_atom_fs": stable_for,
        "stable_through_full_run": stable_for == duration,
        "initial_total_energy_eV": float(energy["total_energy_eV"][0]),
        "final_total_energy_eV": float(energy["total_energy_eV"][-1]),
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
        limit = drift_axis_limit(values["drift_meV_per_atom"])
        axis.set_ylim(-limit, limit)
        axis.axhline(0.0, color="#424242", linewidth=0.7)
        if not result["stable_through_full_run"]:
            axis.axvline(
                result["stable_duration_at_1meV_per_atom_fs"],
                color="#c62828",
                linestyle="--",
                linewidth=1.0,
            )
        axis.set_title(
            f"{result['system']}（{result['formula']}，{result['atoms']} 原子）"
        )
        axis.set_xlabel("模拟时间（fs）")
        axis.set_ylabel("总能量漂移（meV/原子）")
        axis.grid(alpha=0.25)
        axis.text(
            0.03,
            0.95,
            f"最大 |漂移| = {result['maximum_absolute_drift_meV_per_atom']:.4f}",
            transform=axis.transAxes,
            va="top",
            fontsize=9,
        )
    figure.suptitle(
        "NVE 总能量守恒（各子图按数据范围缩放；稳定阈值：±1 meV/原子）",
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
    axis.bar(positions - width / 2, md, width, label="MD 循环", color="#1976d2")
    axis.bar(
        positions + width / 2,
        total,
        width,
        label="端到端",
        color="#ef6c00",
    )
    axis.set_xticks(positions, labels, rotation=35, ha="right")
    axis.set_ylabel("每模拟 1 fs 的墙钟时间（s/fs）")
    axis.set_title("GPU Slurm 任务耗时")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def write_report(results: list[dict], output: Path) -> None:
    validate_common_protocol(results)
    protocol = results[0]
    model_name = protocol["model_name"]
    force_description = (
        "完整 DeePHF 解析梯度力"
        if protocol["force_mode"] == "analytic"
        else "完整 DeePHF 总能量的中心有限差分力"
    )
    lines = [
        "# DeePHF NVE 总能量稳定性报告",
        "",
        f"全部轨迹使用{force_description}；电子结构参考为 GPU4PySCF {protocol['reference_family']}，泛函与基组为 {protocol['xc']}/{protocol['basis']}，并使用已发表的 {model_name} DeePHF 修正网络。总势能为参考方法能量与神经网络修正能量之和。",
        "",
        "稳定性只根据总能量判断：总能量绝对漂移首次超过 1 meV/原子之前的最后一帧记为稳定时长。势能和动能虽被记录，但不单独作为稳定性判据。",
        "",
        "## 1. 九个模拟任务的属性与计算条件",
        "",
        "| 任务 | 分子式 | 原子数 | 电荷 | 多重度 | 电子结构参考 | 交换相关泛函 | 基组 | 积分网格 | DeePHF 修正网络 | 力方法 | 目标温度（K） | 第 0 帧温度（K） | 时间步长（fs） | 步数 | Slurm 作业号 |",
        "|---|---:|---:|---:|---:|---|---|---|---|---|---|---:|---:|---:|---:|---|",
    ]
    for result in results:
        model = result["model_name"]
        grid = (
            f"{result['grid_mode']}/level-{result['grid_level']}"
            f"/rho-{result['small_rho_cutoff']:g}"
        )
        force_method = (
            "解析梯度"
            if result["force_mode"] == "analytic"
            else (
                "中心有限差分"
                f"（{result['finite_difference_step_bohr']:.1e} Bohr）"
            )
        )
        lines.append(
            f"| {result['system']} | {result['formula']} | {result['atoms']} | "
            f"{result['charge']} | {result['multiplicity']} | "
            f"GPU4PySCF {result['reference_family']} | {result['xc']} | "
            f"{result['basis']} | {grid} | {model} | {force_method} | "
            f"{result['target_temperature_K']:.1f} | "
            f"{result['initial_temperature_K']:.1f} | "
            f"{result['timestep_fs']:.3f} | "
            f"{result['steps']} | {result['slurm_job_id']} |"
        )
    lines.extend(
        [
            "",
            "## 2. 总能量稳定性与运行时间",
            "",
            "| 任务 | 模拟时长（fs） | 稳定时长（fs） | 初始总能量（eV） | 最终总能量（eV） | 最大漂移（meV/原子） | RMS 漂移（meV/原子） | 最终漂移（meV/原子） | 线性漂移（meV/原子/fs） | MD 耗时（s/fs） | MD 总耗时（h） |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for result in results:
        stable = result["stable_duration_at_1meV_per_atom_fs"]
        if result["stable_through_full_run"]:
            stable_text = f"≥{stable:.2f}"
        else:
            stable_text = f"{stable:.2f}"
        lines.append(
            f"| {result['system']} | {result['simulated_duration_fs']:.2f} | {stable_text} | "
            f"{result['initial_total_energy_eV']:.6f} | "
            f"{result['final_total_energy_eV']:.6f} | "
            f"{result['maximum_absolute_drift_meV_per_atom']:.6f} | "
            f"{result['rms_drift_meV_per_atom']:.6f} | "
            f"{result['final_drift_meV_per_atom']:.6f} | "
            f"{result['linear_drift_meV_per_atom_per_fs']:.6g} | "
            f"{result['md_wall_seconds_per_simulated_fs']:.3f} | "
            f"{result['md_wall_time_seconds'] / 3600.0:.3f} |"
        )
    maximum = max(result["maximum_absolute_drift_meV_per_atom"] for result in results)
    minimum = min(result["maximum_absolute_drift_meV_per_atom"] for result in results)
    duration = min(result["stable_duration_at_1meV_per_atom_fs"] for result in results)
    lines.extend(
        [
            "",
            f"九个体系在全部已模拟区间内均未越过 1 meV/原子阈值，因此观测到的稳定时长均至少为 {duration:.2f} fs；各体系最大绝对漂移范围为 {minimum:.6f}–{maximum:.6f} meV/原子。",
            "",
            "![总能量稳定性](total_energy_stability.png)",
            "",
            "![每模拟飞秒的墙钟时间](wall_time_per_fs.png)",
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
