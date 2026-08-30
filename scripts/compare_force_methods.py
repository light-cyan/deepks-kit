#!/usr/bin/env python
"""Compare analytic and total-energy finite-difference DeePHF NVE runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if __package__:
    from .analyze_energy_stability import (
        ENERGY_STABILITY_MEV_PER_ATOM,
        analyze_system,
        drift_axis_limit,
        validate_common_protocol,
    )
else:
    from analyze_energy_stability import (
        ENERGY_STABILITY_MEV_PER_ATOM,
        analyze_system,
        drift_axis_limit,
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
                "configuration": analytic_result["configuration"],
                "formula": analytic_result["formula"],
                "atoms": analytic_result["atoms"],
                "charge": analytic_result["charge"],
                "multiplicity": analytic_result["multiplicity"],
                "reference_family": analytic_result["reference_family"],
                "xc": analytic_result["xc"],
                "basis": analytic_result["basis"],
                "grid_mode": analytic_result["grid_mode"],
                "grid_level": analytic_result["grid_level"],
                "small_rho_cutoff": analytic_result["small_rho_cutoff"],
                "model_name": analytic_result["model_name"],
                "target_temperature_K": analytic_result["target_temperature_K"],
                "initial_temperature_K": analytic_result["initial_temperature_K"],
                "timestep_fs": analytic_result["timestep_fs"],
                "analytic_steps": analytic_result["steps"],
                "numerical_steps": numerical_result["steps"],
                "analytic_slurm_job_id": analytic_result["slurm_job_id"],
                "numerical_slurm_job_id": numerical_result["slurm_job_id"],
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
                "analytic_initial_total_energy_eV": analytic_result[
                    "initial_total_energy_eV"
                ],
                "analytic_final_total_energy_eV": analytic_result[
                    "final_total_energy_eV"
                ],
                "analytic_full_max_drift_meV_per_atom": analytic_result[
                    "maximum_absolute_drift_meV_per_atom"
                ],
                "analytic_full_rms_drift_meV_per_atom": analytic_result[
                    "rms_drift_meV_per_atom"
                ],
                "analytic_full_final_drift_meV_per_atom": analytic_result[
                    "final_drift_meV_per_atom"
                ],
                "analytic_full_linear_drift_meV_per_atom_per_fs": analytic_result[
                    "linear_drift_meV_per_atom_per_fs"
                ],
                "analytic_md_wall_time_seconds": analytic_result[
                    "md_wall_time_seconds"
                ],
                "numerical_initial_total_energy_eV": numerical_result[
                    "initial_total_energy_eV"
                ],
                "numerical_final_total_energy_eV": numerical_result[
                    "final_total_energy_eV"
                ],
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
            label="解析梯度",
            color="#1565c0",
            linewidth=1.4,
        )
        axis.plot(
            values["time_fs"],
            values["numerical_drift_meV_per_atom"],
            label="中心有限差分",
            color="#ef6c00",
            linewidth=1.2,
        )
        limit = drift_axis_limit(
            values["analytic_drift_meV_per_atom"],
            values["numerical_drift_meV_per_atom"],
        )
        axis.set_ylim(-limit, limit)
        axis.axhline(0.0, color="#424242", linewidth=0.7)
        axis.set_title(f"{row['system']} ({row['formula']})")
        axis.set_xlabel("模拟时间（fs）")
        axis.set_ylabel("总能量漂移（meV/原子）")
        axis.grid(alpha=0.25)
        axis.legend()
    for axis in axes.flat[len(rows) :]:
        axis.set_visible(False)
    figure.suptitle("DeePHF 解析梯度与数值梯度的总能量漂移（各子图按数据范围缩放）")
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
        label="解析梯度",
        color="#1565c0",
    )
    axis.bar(
        positions + width / 2,
        numerical,
        width,
        label="中心有限差分",
        color="#ef6c00",
    )
    axis.set_xticks(positions, labels, rotation=35, ha="right")
    axis.set_ylabel("每模拟 1 fs 的 MD 墙钟时间（s/fs）")
    axis.set_title("DeePHF 力方法运行时间对比")
    if numerical.max() / analytic.min() > 20.0:
        axis.set_yscale("log")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def write_report(rows: list[dict], output: Path, analytic_root: Path) -> None:
    step = rows[0]["finite_difference_step_bohr"]
    analytic_steps = rows[0]["analytic_steps"]
    numerical_steps = rows[0]["numerical_steps"]
    analytic_duration = rows[0]["analytic_duration_fs"]
    numerical_duration = rows[0]["numerical_duration_fs"]
    common_duration = rows[0]["common_duration_fs"]
    maximum_analytic = max(
        row["analytic_full_max_drift_meV_per_atom"] for row in rows
    )
    minimum_analytic = min(
        row["analytic_full_max_drift_meV_per_atom"] for row in rows
    )
    maximum_numerical = max(
        row["numerical_common_max_drift_meV_per_atom"] for row in rows
    )
    ratios = [row["numerical_over_analytic_wall_time_ratio"] for row in rows]
    analytic_energy_plot = Path(
        os.path.relpath(
            analytic_root / "total_energy_stability.png",
            start=output.parent,
        )
    ).as_posix()
    analytic_timing_plot = Path(
        os.path.relpath(
            analytic_root / "wall_time_per_fs.png",
            start=output.parent,
        )
    ).as_posix()
    lines = [
        "# DeePHF GPU 分子动力学与力方法对照报告",
        "",
        "本报告只研究 NVE 轨迹的总能量稳定性。稳定判据为总能量绝对漂移不超过 1 meV/原子；观测区间内未越过阈值时，只报告稳定时长至少达到模拟终点，不对更长时间作外推。",
        "",
        "## 1. 九个模拟任务的属性与计算方法",
        "",
        "九个任务来自已发表 DeePHF 数据中的 GRAM 构型，均为电荷 0、多重度 1 的闭壳层体系。势能采用 `GPU4PySCF RKS B3LYP5/def2-tzvp + b3lyp_gram_t1x.pth DeePHF 神经网络修正`，即神经网络确实参与每一帧总势能及其力的计算。分子动力学使用 ASE Velocity-Verlet NVE 积分器，通过 Slurm 申请 GPU。",
        "",
        f"解析梯度实验直接计算完整 DeePHF 总能量对核坐标的解析导数，其中 GPU4PySCF RKS 梯度包含 DFT 数值积分网格响应，共 {analytic_steps} 步、{analytic_duration:.0f} fs。数值梯度对照对同一完整 DeePHF 总能量作中心有限差分，位移为 {step:.1e} Bohr，共 {numerical_steps} 步、{numerical_duration:.0f} fs；每一帧需要 `1 + 6N` 次总能量计算，即本组体系为 37–85 次。两种方法的初始结构、初始动量、SCF 条件、温度和时间步长完全一致。",
        "本次解析梯度任务中 gram_01、gram_04、gram_07 使用本机 RTX 5090，其余六个体系使用 node2 RTX PRO 6000；数值梯度数据复用此前 node1 RTX 5090 的 10 fs 任务。运行时间比是本批 Slurm 作业的实际吞吐对比，包含 GPU 型号差异，不作为同一硬件上的隔离变量微基准。",
        "",
        "| 任务 | 原始构型 | 分子式 | 原子数 | 电荷 | 多重度 | 参考方法 | 泛函/基组 | 积分网格 | 修正网络 | 目标/第 0 帧温度（K） | 步长（fs） | 解析任务 | 数值任务 |",
        "|---|---|---:|---:|---:|---:|---|---|---|---|---:|---:|---|---|",
    ]
    for row in rows:
        grid = (
            f"{row['grid_mode']}/level-{row['grid_level']}"
            f"/rho-{row['small_rho_cutoff']:g}"
        )
        lines.append(
            f"| {row['system']} | {row['configuration']} | {row['formula']} | "
            f"{row['atoms']} | {row['charge']} | {row['multiplicity']} | "
            f"GPU4PySCF {row['reference_family']} | {row['xc']}/{row['basis']} | "
            f"{grid} | {row['model_name']} | {row['target_temperature_K']:.1f}/"
            f"{row['initial_temperature_K']:.1f} | {row['timestep_fs']:.3f} | "
            f"{row['analytic_steps']} 步/{row['analytic_duration_fs']:.0f} fs，作业 {row['analytic_slurm_job_id']} | "
            f"{row['numerical_steps']} 步/{row['numerical_duration_fs']:.0f} fs，作业 {row['numerical_slurm_job_id']} |"
        )
    lines.extend(
        [
            "",
            f"## 2. 解析梯度 {analytic_duration:.0f} fs 实验结果",
            "",
            "| 任务 | 模拟时长（fs） | 稳定时长（fs） | 初始总能量（eV） | 最终总能量（eV） | 最大漂移（meV/原子） | RMS 漂移（meV/原子） | 最终漂移（meV/原子） | 线性漂移（meV/原子/fs） | MD 耗时（s/fs） | MD 总耗时（h） |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['system']} | {row['analytic_duration_fs']:.2f} | "
            f"≥{row['analytic_stable_duration_fs']:.2f} | "
            f"{row['analytic_initial_total_energy_eV']:.6f} | "
            f"{row['analytic_final_total_energy_eV']:.6f} | "
            f"{row['analytic_full_max_drift_meV_per_atom']:.6f} | "
            f"{row['analytic_full_rms_drift_meV_per_atom']:.6f} | "
            f"{row['analytic_full_final_drift_meV_per_atom']:.6f} | "
            f"{row['analytic_full_linear_drift_meV_per_atom_per_fs']:.6g} | "
            f"{row['analytic_md_wall_seconds_per_fs']:.3f} | "
            f"{row['analytic_md_wall_time_seconds'] / 3600.0:.3f} |"
        )
    lines.extend(
        [
            "",
            f"九个解析梯度轨迹均稳定到 {analytic_duration:.0f} fs 终点，最大绝对漂移范围为 {minimum_analytic:.6f}–{maximum_analytic:.6f} meV/原子，最差值也只有稳定阈值的 {100.0 * maximum_analytic / ENERGY_STABILITY_MEV_PER_ATOM:.2f}%。",
            "",
            f"![解析梯度 {analytic_duration:.0f} fs 总能量漂移]({analytic_energy_plot})",
            "",
            f"![解析梯度 {analytic_duration:.0f} fs 每模拟飞秒耗时]({analytic_timing_plot})",
            "",
            "## 3. 解析梯度与数值梯度对照",
            "",
            f"为保证公平，只比较两组共同拥有的前 {common_duration:.0f} fs 数据。表中的解析梯度漂移也重新限制在前 {common_duration:.0f} fs，而运行时间采用各自完整任务的 MD 阶段平均值。",
            "",
            "| 任务 | 共同区间（fs） | 解析最大漂移（meV/原子） | 数值最大漂移（meV/原子） | 解析耗时（s/fs） | 数值耗时（s/fs） | 数值/解析耗时比 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['system']} | {row['common_duration_fs']:.2f} | "
            f"{row['analytic_common_max_drift_meV_per_atom']:.6f} | "
            f"{row['numerical_common_max_drift_meV_per_atom']:.6f} | "
            f"{row['analytic_md_wall_seconds_per_fs']:.3f} | "
            f"{row['numerical_md_wall_seconds_per_fs']:.3f} | "
            f"{row['numerical_over_analytic_wall_time_ratio']:.2f} 倍 |"
        )
    lines.extend(
        [
            "",
            f"两种力方法在共同的 {common_duration:.0f} fs 区间内均未越过稳定阈值；解析梯度的最大全局漂移为 {max(row['analytic_common_max_drift_meV_per_atom'] for row in rows):.6f} meV/原子，数值梯度为 {maximum_numerical:.6f} meV/原子。数值梯度的实测墙钟时间平均为解析梯度的 {np.mean(ratios):.2f} 倍，逐体系范围为 {min(ratios):.2f}–{max(ratios):.2f} 倍。当前数据支持“解析梯度至少稳定 {analytic_duration:.0f} fs、数值梯度至少稳定 {numerical_duration:.0f} fs”。本次没有重跑数值梯度，{common_duration:.0f} fs 对照直接复用既有 {numerical_duration:.0f} fs 轨迹的相同前缀；不对解析梯度在 {analytic_duration:.0f} fs 之后的稳定性作外推。",
            "",
            "![解析梯度与数值梯度总能量漂移对照](force_method_energy_comparison.png)",
            "",
            "![解析梯度与数值梯度运行时间对照](force_method_timing_comparison.png)",
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
    write_report(
        rows,
        args.output_root / "force_method_comparison.md",
        args.analytic_root,
    )
    print(json.dumps(rows, sort_keys=True))


if __name__ == "__main__":
    main()
