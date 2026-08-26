"""Aggregate child JSON results into machine-readable and Markdown reports."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np

from common import REPORT_DIR, VALIDATION_DIR, load_config, sha256_file, write_json


def _load_results(run_root: Path) -> list[dict[str, Any]]:
    results = []
    for path in sorted(run_root.glob("**/result.json")):
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            result = {"invalid_json": str(error)}
        result["result_path"] = str(path)
        results.append(result)
    return results


def _load_attempts(run_root: Path) -> list[dict[str, Any]]:
    attempts = []
    for path in sorted(run_root.glob("**/result_attempt*.json")):
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            result = {"invalid_json": str(error)}
        result["result_path"] = str(path)
        attempts.append(result)
    return attempts


def _category_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {}
    for category in ("scientific", "integrity", "performance", "resource"):
        counts = Counter()
        failures = []
        for result in results:
            record = result.get("categories", {}).get(category, {})
            passed = record.get("passed")
            counts[str(passed).lower()] += 1
            if passed is False:
                failures.append(
                    {
                        "result_path": result["result_path"],
                        "action": result.get("action"),
                        "workload_id": result.get("workload_id"),
                        "family": result.get("family"),
                        "reasons": record.get("reasons", []),
                    }
                )
        summary[category] = {"counts": dict(counts), "failures": failures}
    return summary


def _accuracy_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        central = result.get("central")
        if central is None:
            continue
        rows.append(
            {
                "workload_id": result.get("workload_id"),
                "family": result.get("family"),
                "zero_energy_max_abs": central["errors"]["zero_energy_native"],
                "zero_gradient_max_abs": central["errors"]["zero_gradient_native"]["max_abs"],
                "direct_zvector_max_abs": central["errors"]["direct_zvector_detailed"]["max_abs"],
                "direct_compact_detailed_max_abs": central["errors"]["direct_compact_detailed"]["max_abs"],
                "zvector_compact_detailed_max_abs": central["errors"]["zvector_compact_detailed"]["max_abs"],
                **result.get("finite_difference_summary", {}),
                "passed": result.get("categories", {}).get("scientific", {}).get("passed"),
            }
        )
    return rows


def _dense_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        dense = result.get("dense")
        if dense is None:
            continue
        rows.append(
            {
                "workload_id": result.get("workload_id"),
                "family": result.get("family"),
                "response_dimension": result.get("reference", {}).get("response_dimension"),
                "condition_number": dense["condition_number"],
                "solution_relative_l2": dense["solution_error"]["relative_l2"],
                "solution_max_abs": dense["solution_error"]["max_abs"],
                "final_gradient_max_abs": dense["final_gradient_error"]["max_abs"],
                "construction_seconds": dense["construction_seconds"],
                "solve_seconds": dense["solve_seconds"],
                "peak_rss_kib": result.get("process", {}).get("resource_measurement", {}).get("peak_rss_kib"),
                "passed": result.get("categories", {}).get("scientific", {}).get("passed"),
            }
        )
    return rows


def _selection_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    subset_results = [result for result in results if result.get("action") == "atom-subset"]
    grouped = defaultdict(dict)
    for result in subset_results:
        key = (
            result.get("profile"),
            result.get("workload_id"),
            result.get("family"),
            result.get("backend"),
            result.get("process", {}).get("source_revision"),
        )
        grouped[key][result.get("subset")] = result
    rows = []
    for key, values in grouped.items():
        full = values.get("full")
        if full is None:
            continue
        full_gradient = np.asarray(full["gradient"])
        for label, result in values.items():
            selected_atoms = result["selected_atoms"]
            error = np.asarray(result["gradient"]) - full_gradient[selected_atoms]
            rows.append(
                {
                    "profile": key[0],
                    "workload_id": key[1],
                    "family": key[2],
                    "backend": key[3],
                    "revision": key[4],
                    "subset": label,
                    "selected_atoms": selected_atoms,
                    "max_abs_error": float(np.max(np.abs(error), initial=0.0)),
                    "timing": result.get("timing"),
                    "peak_rss_kib": result.get("process", {}).get("resource_measurement", {}).get("peak_rss_kib"),
                }
            )
    return rows


def _block_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    block_results = [result for result in results if result.get("action") == "coordinate-block"]
    benchmark = {
        (
            result.get("profile"),
            result.get("workload_id"),
            result.get("family"),
            result.get("process", {}).get("source_revision"),
        ): result
        for result in results
        if result.get("action") == "benchmark"
        and result.get("backend") == "direct"
        and result.get("result_mode") == "compact"
    }
    rows = []
    for result in block_results:
        key = (
            result.get("profile"),
            result.get("workload_id"),
            result.get("family"),
            result.get("process", {}).get("source_revision"),
        )
        expected_result = benchmark.get(key)
        error = None
        if expected_result is not None:
            difference = np.asarray(result["gradient"]) - np.asarray(
                expected_result["measurement"]["gradient"]
            )
            error = float(np.max(np.abs(difference), initial=0.0))
        rows.append(
            {
                "profile": result.get("profile"),
                "workload_id": result.get("workload_id"),
                "family": result.get("family"),
                "coordinate_block_size": result.get("coordinate_block_size"),
                "max_abs_error": error,
                "timing": result.get("timing"),
                "peak_rss_kib": result.get("process", {}).get("resource_measurement", {}).get("peak_rss_kib"),
                "maximum_residual": (result.get("diagnostics") or {}).get("maximum_residual"),
            }
        )
    return rows


def _selection_acceptance(
    selection_rows: list[dict[str, Any]], block_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    tolerance = load_config()["acceptance"]["selected_gradient_max_abs_hartree_per_bohr"]
    selected_passed = bool(selection_rows) and all(
        row["max_abs_error"] <= tolerance for row in selection_rows
    )
    comparable_blocks = [row for row in block_rows if row["max_abs_error"] is not None]
    blocks_passed = bool(comparable_blocks) and all(
        row["max_abs_error"] <= tolerance
        and row["maximum_residual"] <= load_config()["response_controls"]["residual_tolerance"]
        for row in comparable_blocks
    )
    memory_checks = []
    for profile in load_config()["profiles"]:
        x1 = [
            row
            for row in block_rows
            if row["profile"] == profile and row["workload_id"] == "X1-def2-TZVP"
        ]
        if not x1:
            continue
        smallest = min(x1, key=lambda row: row["coordinate_block_size"])
        largest = max(x1, key=lambda row: row["coordinate_block_size"])
        memory_checks.append(
            {
                "profile": profile,
                "smallest_block_size": smallest["coordinate_block_size"],
                "largest_block_size": largest["coordinate_block_size"],
                "smallest_peak_rss_kib": smallest["peak_rss_kib"],
                "largest_peak_rss_kib": largest["peak_rss_kib"],
                "passed": smallest["peak_rss_kib"] < largest["peak_rss_kib"],
            }
        )
    memory_passed = bool(memory_checks) and all(row["passed"] for row in memory_checks)
    checks = {
        "selected_rows": selected_passed,
        "coordinate_blocks": blocks_passed,
        "x1_coordinate_memory_reduction": memory_passed,
    }
    return {"passed": all(checks.values()), "checks": checks, "memory_checks": memory_checks}


def _benchmark_key(result: dict[str, Any]) -> tuple[Any, ...]:
    return (
        result.get("profile"),
        result.get("workload_id"),
        result.get("family"),
        result.get("process", {}).get("source_revision"),
    )


def _benchmark_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        if result.get("action") != "benchmark":
            continue
        measurement = result.get("measurement")
        if measurement is None:
            continue
        resource = result.get("process", {}).get("resource_measurement", {})
        diagnostics = measurement.get("diagnostics") or {}
        rows.append(
            {
                "profile": result.get("profile"),
                "workload_id": result.get("workload_id"),
                "family": result.get("family"),
                "backend": result.get("backend"),
                "result_mode": result.get("result_mode"),
                "revision": result.get("process", {}).get("source_revision"),
                "response_dimension": result.get("reference", {}).get("response_dimension"),
                "ao_count": result.get("reference", {}).get("ao_count"),
                "atom_count": result.get("reference", {}).get("atom_count"),
                "nuclear_rhs_count": 3 * result.get("reference", {}).get("atom_count", 0),
                "median_seconds": measurement["warm_gradient"]["median"],
                "mad_fraction": measurement["warm_gradient"]["mad_fraction"],
                "cold_seconds": measurement["cold_end_to_end_seconds"],
                "peak_rss_kib": resource.get("peak_rss_kib"),
                "python_peak_bytes": measurement.get("python_peak_allocation_bytes"),
                "retained_array_bytes": measurement.get("retained_array_bytes"),
                "iterations": diagnostics.get("iteration_count"),
                "residual": diagnostics.get("maximum_residual"),
                "solver": diagnostics.get("solver"),
                "solve_count": diagnostics.get("solve_count"),
                "matrix_free_action_counts": measurement.get("matrix_free_action_counts"),
                "exit_status": result.get("process", {}).get("exit_status"),
                "performance_passed": result.get("categories", {}).get("performance", {}).get("passed"),
                "result_path": result["result_path"],
            }
        )
    return rows


def _native_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        if result.get("action") != "native-benchmark":
            continue
        measurement = result.get("measurement", {})
        rows.append(
            {
                "profile": result.get("profile"),
                "workload_id": result.get("workload_id"),
                "family": result.get("family"),
                "median_seconds": measurement.get("warm_gradient", {}).get("median"),
                "peak_rss_kib": result.get("process", {}).get("resource_measurement", {}).get("peak_rss_kib"),
                "exit_status": result.get("process", {}).get("exit_status"),
            }
        )
    return rows


def _add_incremental_memory(
    benchmark_rows: list[dict[str, Any]], native_rows: list[dict[str, Any]]
) -> None:
    native = {
        (row["profile"], row["workload_id"], row["family"]): row
        for row in native_rows
    }
    for row in benchmark_rows:
        baseline = native.get((row["profile"], row["workload_id"], row["family"]))
        row["incremental_peak_rss_kib"] = (
            None
            if baseline is None or baseline["peak_rss_kib"] is None or row["peak_rss_kib"] is None
            else row["peak_rss_kib"] - baseline["peak_rss_kib"]
        )


def _add_scientific_acceptance(
    results: list[dict[str, Any]], benchmark_rows: list[dict[str, Any]]
) -> None:
    accepted = {}
    for action in ("preflight", "scientific"):
        for result in results:
            if result.get("action") != action:
                continue
            accepted[(result.get("workload_id"), result.get("family"))] = (
                result.get("categories", {}).get("scientific", {}).get("passed") is True
            )
    for row in benchmark_rows:
        row["scientific_accepted"] = accepted.get(
            (row["workload_id"], row["family"]), False
        )


def _ratios(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = defaultdict(dict)
    for row in rows:
        key = (row["profile"], row["workload_id"], row["family"], row["revision"])
        grouped[key][(row["backend"], row["result_mode"])] = row
    output = []
    for key, values in grouped.items():
        required = {
            ("direct", "compact"),
            ("direct", "detailed"),
            ("zvector", "compact"),
            ("zvector", "detailed"),
        }
        if not required.issubset(values):
            continue
        if not all(values[target].get("scientific_accepted") for target in required):
            continue
        dc = values[("direct", "compact")]
        dd = values[("direct", "detailed")]
        zc = values[("zvector", "compact")]
        zd = values[("zvector", "detailed")]
        output.append(
            {
                "profile": key[0],
                "workload_id": key[1],
                "family": key[2],
                "revision": key[3],
                "zvector_compact_time_over_direct_compact": zc["median_seconds"] / dc["median_seconds"],
                "zvector_compact_rss_over_direct_compact": zc["peak_rss_kib"] / dc["peak_rss_kib"],
                "zvector_compact_time_over_zvector_detailed": zc["median_seconds"] / zd["median_seconds"],
                "zvector_compact_rss_over_zvector_detailed": zc["peak_rss_kib"] / zd["peak_rss_kib"],
                "direct_compact_time_over_direct_detailed": dc["median_seconds"] / dd["median_seconds"],
                "direct_compact_rss_over_direct_detailed": dc["peak_rss_kib"] / dd["peak_rss_kib"],
            }
        )
    return output


def _cross_revision_ratios(rows: list[dict[str, Any]], current_revision: str) -> list[dict[str, Any]]:
    indexed = {
        (
            row["profile"],
            row["workload_id"],
            row["family"],
            row["backend"],
            row["result_mode"],
            row["revision"],
        ): row
        for row in rows
    }
    revisions = sorted({row["revision"] for row in rows if row["revision"]})
    historical = [revision for revision in revisions if revision != current_revision]
    output = []
    for old_revision in historical:
        for row in rows:
            if row["revision"] != current_revision:
                continue
            old_key = (
                row["profile"],
                row["workload_id"],
                row["family"],
                row["backend"],
                row["result_mode"],
                old_revision,
            )
            old = indexed.get(old_key)
            if old is None:
                continue
            output.append(
                {
                    "profile": row["profile"],
                    "workload_id": row["workload_id"],
                    "family": row["family"],
                    "backend": row["backend"],
                    "result_mode": row["result_mode"],
                    "current_revision": current_revision,
                    "historical_revision": old_revision,
                    "current_time_over_historical": row["median_seconds"] / old["median_seconds"],
                    "current_rss_over_historical": row["peak_rss_kib"] / old["peak_rss_kib"],
                }
            )
    return output


def _production_historical_ratios(rows: list[dict[str, Any]], current_revision: str) -> list[dict[str, Any]]:
    revisions = sorted({row["revision"] for row in rows if row["revision"] and row["revision"] != current_revision})
    output = []
    for old_revision in revisions:
        old_index = {
            (row["profile"], row["workload_id"], row["family"], row["backend"]): row
            for row in rows
            if row["revision"] == old_revision and row["result_mode"] == "detailed"
        }
        for current in rows:
            if (
                current["revision"] != current_revision
                or current["result_mode"] != "compact"
                or not current.get("scientific_accepted")
            ):
                continue
            old = old_index.get(
                (current["profile"], current["workload_id"], current["family"], current["backend"])
            )
            if old is None or current["peak_rss_kib"] is None or old["peak_rss_kib"] is None:
                continue
            output.append(
                {
                    "profile": current["profile"],
                    "workload_id": current["workload_id"],
                    "family": current["family"],
                    "backend": current["backend"],
                    "current_revision": current_revision,
                    "historical_revision": old_revision,
                    "current_compact_time_over_historical_detailed": current["median_seconds"] / old["median_seconds"],
                    "current_compact_rss_over_historical_detailed": current["peak_rss_kib"] / old["peak_rss_kib"],
                }
            )
    return output


def _performance_acceptance(
    results: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    ratios: list[dict[str, Any]],
    current_revision: str,
) -> dict[str, Any]:
    current_ratios = [row for row in ratios if row["revision"] == current_revision]
    large_ids = {"L1-def2-SVP", "L1-def2-TZVP", "L2-def2-SVP", "L3-def2-SVP", "X1-def2-TZVP"}
    compact_checks = []
    for row in current_ratios:
        if row["workload_id"] not in large_ids:
            continue
        compact_checks.append(
            {
                **row,
                "time_passed": row["zvector_compact_time_over_zvector_detailed"] <= 1.0 and row["direct_compact_time_over_direct_detailed"] <= 1.0,
                "rss_passed": row["zvector_compact_rss_over_zvector_detailed"] <= 1.0 and row["direct_compact_rss_over_direct_detailed"] <= 1.0,
            }
        )
    closed_shell_speedups = [
        row
        for row in current_ratios
        if row["workload_id"] in {"L1-def2-SVP", "L1-def2-TZVP", "L2-def2-SVP", "X1-def2-TZVP"}
        and row["family"] in {"rhf", "rks"}
        and row["zvector_compact_time_over_direct_compact"] < 1.0
    ]
    zvector_structure = [
        {
            "profile": row["profile"],
            "workload_id": row["workload_id"],
            "family": row["family"],
            "passed": row["solver"] == "scipy.sparse.linalg.gmres(A.T, b)"
            and row["solve_count"] == 1
            and isinstance(row["iterations"], int)
            and row["iterations"] > 0
            and isinstance(row.get("matrix_free_action_counts"), dict)
            and row["matrix_free_action_counts"].get("transpose", 0) > 0
            and row["matrix_free_action_counts"].get("preconditioner", 0) > 0,
        }
        for row in rows
        if row["revision"] == current_revision
        and row["backend"] == "zvector"
        and row["result_mode"] == "compact"
        and row["exit_status"] == 0
    ]
    historical = _production_historical_ratios(rows, current_revision)
    slowdowns = [
        row
        for row in historical
        if row["current_compact_time_over_historical_detailed"] > 1.2
    ]
    historical_revisions = {
        row["revision"] for row in rows if row["revision"] and row["revision"] != current_revision
    }
    dense_rss = {
        (result.get("workload_id"), result.get("family")): result.get("process", {}).get("resource_measurement", {}).get("peak_rss_kib")
        for result in results
        if result.get("action") == "dense-replay"
        and result.get("process", {}).get("exit_status") == 0
    }
    memory_candidates = []
    for old_revision in historical_revisions:
        old_index = {
            (row["profile"], row["workload_id"], row["family"]): row
            for row in rows
            if row["revision"] == old_revision
            and row["backend"] == "zvector"
            and row["result_mode"] == "detailed"
            and row["exit_status"] == 0
        }
        for current in rows:
            if (
                current["revision"] != current_revision
                or current["profile"] != "deterministic-1t"
                or current["backend"] != "zvector"
                or current["result_mode"] != "compact"
                or current["exit_status"] != 0
            ):
                continue
            old = old_index.get((current["profile"], current["workload_id"], current["family"]))
            dense = dense_rss.get((current["workload_id"], current["family"]))
            if old is None or dense is None:
                continue
            memory_candidates.append(
                {
                    "workload_id": current["workload_id"],
                    "family": current["family"],
                    "response_dimension": current["response_dimension"],
                    "current_peak_rss_kib": current["peak_rss_kib"],
                    "historical_peak_rss_kib": old["peak_rss_kib"],
                    "dense_peak_rss_kib": dense,
                }
            )
    largest_memory_comparison = (
        max(memory_candidates, key=lambda row: row["response_dimension"])
        if memory_candidates
        else None
    )
    largest_memory_passed = bool(
        largest_memory_comparison
        and largest_memory_comparison["current_peak_rss_kib"]
        < largest_memory_comparison["historical_peak_rss_kib"]
        and largest_memory_comparison["current_peak_rss_kib"]
        < largest_memory_comparison["dense_peak_rss_kib"]
    )
    checks = {
        "all_large_compact_time": bool(compact_checks) and all(row["time_passed"] for row in compact_checks),
        "all_large_compact_rss": bool(compact_checks) and all(row["rss_passed"] for row in compact_checks),
        "two_large_closed_shell_speedups": len(closed_shell_speedups) >= 2,
        "zvector_scalar_gmres_structure": bool(zvector_structure) and all(row["passed"] for row in zvector_structure),
        "no_unattributed_historical_slowdown": not slowdowns,
        "largest_common_case_memory": largest_memory_passed,
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "compact_checks": compact_checks,
        "large_closed_shell_speedups": closed_shell_speedups,
        "zvector_structure": zvector_structure,
        "historical_slowdowns_over_20_percent": slowdowns,
        "largest_common_case_memory_comparison": largest_memory_comparison,
    }


def _scaling_fits(rows: list[dict[str, Any]], current_revision: str) -> list[dict[str, Any]]:
    selected = [
        row
        for row in rows
        if row["revision"] == current_revision
        and row["backend"] == "zvector"
        and row["result_mode"] == "compact"
        and row["exit_status"] == 0
    ]
    fits = []
    for profile in load_config()["profiles"]:
        profile_rows = [row for row in selected if row["profile"] == profile]
        for predictor in ("response_dimension", "ao_count", "atom_count", "nuclear_rhs_count"):
            valid = [
                row
                for row in profile_rows
                if row[predictor] and row["median_seconds"] > 0 and row["peak_rss_kib"]
            ]
            if len(valid) < 3:
                continue
            x = np.log(np.asarray([row[predictor] for row in valid], dtype=np.float64))
            for outcome in ("median_seconds", "peak_rss_kib", "incremental_peak_rss_kib"):
                valid_outcome = [row for row in valid if row.get(outcome) is not None and row[outcome] > 0]
                if len(valid_outcome) < 3:
                    continue
                x = np.log(np.asarray([row[predictor] for row in valid_outcome], dtype=np.float64))
                y = np.log(np.asarray([row[outcome] for row in valid_outcome], dtype=np.float64))
                slope, intercept = np.polyfit(x, y, 1)
                prediction = slope * x + intercept
                residual = y - prediction
                fits.append(
                    {
                        "profile": profile,
                        "predictor": predictor,
                        "outcome": outcome,
                        "sample_count": len(valid_outcome),
                        "log_log_slope": float(slope),
                        "log_log_intercept": float(intercept),
                        "residual_l2": float(np.linalg.norm(residual)),
                        "workloads": [row["workload_id"] for row in valid_outcome],
                    }
                )
    return fits


def _artifact_manifest() -> list[dict[str, Any]]:
    paths = [VALIDATION_DIR / "README.md", VALIDATION_DIR / "configs" / "campaign.json"]
    paths.extend(sorted((VALIDATION_DIR / "geometries").glob("*.xyz")))
    paths.extend(sorted((VALIDATION_DIR / "scripts").glob("*.py")))
    checkpoint = VALIDATION_DIR / "checkpoints" / "frozen_corrnet.pth"
    if checkpoint.exists():
        paths.append(checkpoint)
    return [
        {
            "path": str(path.relative_to(VALIDATION_DIR)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in paths
    ]


def _campaign_integrity(results: list[dict[str, Any]]) -> dict[str, Any]:
    validation_hashes = sorted(
        {
            result.get("environment", {}).get("validation_hash")
            for result in results
            if result.get("environment", {}).get("validation_hash")
        }
    )
    source_snapshots = {
        (
            environment.get("source_root"),
            environment.get("base_revision"),
            environment.get("tracked_diff_sha256"),
            environment.get("config_hash"),
            environment.get("validation_hash"),
        )
        for result in results
        if (environment := result.get("environment")) is not None
    }
    source_provenance_complete = bool(source_snapshots) and all(
        all(value is not None for value in snapshot) for snapshot in source_snapshots
    )
    source_clean_states = sorted(
        {
            result.get("environment", {}).get("source_clean")
            for result in results
            if "environment" in result
        },
        key=str,
    )
    profile_controls = []
    config = load_config()
    for result in results:
        environment = result.get("environment")
        profile = result.get("profile")
        if environment is None or profile not in config["profiles"]:
            continue
        expected = config["profiles"][profile]
        actual_affinity = environment.get("process_affinity", [])
        if "-" in expected["cores"]:
            start, stop = (int(value) for value in expected["cores"].split("-"))
            expected_affinity = list(range(start, stop + 1))
        else:
            expected_affinity = [int(expected["cores"])]
        profile_controls.append(
            {
                "result_path": result.get("result_path"),
                "profile": profile,
                "affinity_passed": actual_affinity == expected_affinity,
                "threads_passed": environment.get("declared_threads") == expected["threads"],
            }
        )
    checks = {
        "one_validation_hash": len(validation_hashes) == 1,
        "one_source_snapshot": len(source_snapshots) == 1,
        "source_provenance_complete": source_provenance_complete,
        "profile_affinity_and_threads": bool(profile_controls)
        and all(item["affinity_passed"] and item["threads_passed"] for item in profile_controls),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "validation_hashes": validation_hashes,
        "source_snapshots": [
            {
                "source_root": snapshot[0],
                "base_revision": snapshot[1],
                "tracked_diff_sha256": snapshot[2],
                "config_hash": snapshot[3],
                "validation_hash": snapshot[4],
            }
            for snapshot in sorted(source_snapshots, key=str)
        ],
        "source_clean_states": source_clean_states,
        "profile_controls": profile_controls,
    }


def aggregate(run_root: Path) -> dict[str, Any]:
    results = _load_results(run_root)
    manifest_path = run_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    benchmark_rows = _benchmark_rows(results)
    native_rows = _native_rows(results)
    _add_incremental_memory(benchmark_rows, native_rows)
    _add_scientific_acceptance(results, benchmark_rows)
    current_revision = manifest["current_revision"]
    backend_ratios = _ratios(benchmark_rows)
    selection_rows = _selection_rows(results)
    block_rows = _block_rows(results)
    aggregate_result = {
        "schema_version": 1,
        "run_root": str(run_root),
        "run_manifest": manifest,
        "result_count": len(results),
        "preserved_retry_attempts": _load_attempts(run_root),
        "category_summary": _category_summary(results),
        "campaign_integrity": _campaign_integrity(results),
        "accuracy": _accuracy_rows(results),
        "dense_replay": _dense_rows(results),
        "benchmarks": benchmark_rows,
        "native_benchmarks": native_rows,
        "backend_ratios": backend_ratios,
        "cross_revision_ratios": _cross_revision_ratios(benchmark_rows, current_revision),
        "production_historical_ratios": _production_historical_ratios(benchmark_rows, current_revision),
        "performance_acceptance": _performance_acceptance(
            results, benchmark_rows, backend_ratios, current_revision
        ),
        "scaling_fits": _scaling_fits(benchmark_rows, current_revision),
        "conditioning": [result for result in results if result.get("action") == "conditioning-sweep"],
        "selection": selection_rows,
        "coordinate_blocking": block_rows,
        "selection_acceptance": _selection_acceptance(selection_rows, block_rows),
        "dft": [result for result in results if result.get("action") == "dft-sequence"],
        "scanner": [result for result in results if result.get("action") == "scanner"],
        "force_data": [result for result in results if str(result.get("action", "")).startswith("force-data")],
        "artifacts": _artifact_manifest(),
    }
    return aggregate_result


def _format_number(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _format_coordinate(record: dict[str, Any] | None) -> str:
    if not record:
        return "—"
    return f"atom {record['atom']} {record['axis_name']}"


def _format_descriptor_location(record: dict[str, Any] | None) -> str:
    if not record:
        return "—"
    index = ",".join(str(value) for value in record["descriptor_index"])
    return f"{_format_coordinate(record)}; q[{index}]"


def _format_direction(record: dict[str, Any] | None) -> str:
    if not record:
        return "—"
    return str(record["direction_index"])


def write_markdown(aggregate_result: dict[str, Any], path: Path) -> None:
    categories = aggregate_result["category_summary"]
    lines = [
        "# Scientific Correctness and Performance Validation Report",
        "",
        f"Run: `{aggregate_result['run_manifest']['run_id']}`",
        "",
        f"Revision: `{aggregate_result['run_manifest']['current_revision']}`",
        "",
        f"Results: {aggregate_result['result_count']}",
        "",
        "## Outcome categories",
        "",
        "| Category | Passed | Failed | Not applicable or pending |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name in ("scientific", "integrity", "performance", "resource"):
        counts = categories[name]["counts"]
        lines.append(
            f"| {name} | {counts.get('true', 0)} | {counts.get('false', 0)} | {counts.get('none', 0)} |"
        )
    lines.extend(
        [
            "",
            "## Scientific accuracy",
            "",
            "| Workload | Family | Direct/Z-vector max abs | FD component max abs | FD direction max abs | Descriptor FD max abs | Passed |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in aggregate_result["accuracy"]:
        lines.append(
            "| {workload_id} | {family} | {direct_zvector_max_abs} | {maximum_component_error} | {maximum_directional_error} | {maximum_relaxed_descriptor_error} | {passed} |".format(
                **{key: _format_number(value) for key, value in row.items()}
            )
        )
    lines.extend(
        [
            "",
            "## Finite-difference accuracy by predeclared step",
            "",
            "| Workload | Family | Step (Bohr) | Force max abs | Worst coordinate | Descriptor max abs | Worst descriptor | Direction max abs | Worst direction | Passed |",
            "| --- | --- | ---: | ---: | --- | ---: | --- | ---: | ---: | --- |",
        ]
    )
    for row in aggregate_result["accuracy"]:
        for step_summary in row.get("per_step", {}).values():
            lines.append(
                "| {workload} | {family} | {step} | {force} | {coordinate} | {descriptor} | {descriptor_location} | {direction} | {direction_index} | {passed} |".format(
                    workload=row["workload_id"],
                    family=row["family"],
                    step=_format_number(step_summary["step_bohr"]),
                    force=_format_number(step_summary["maximum_component_error"]),
                    coordinate=_format_coordinate(step_summary["worst_component"]),
                    descriptor=_format_number(
                        step_summary["maximum_relaxed_descriptor_error"]
                    ),
                    descriptor_location=_format_descriptor_location(
                        step_summary["worst_descriptor"]
                    ),
                    direction=_format_number(
                        step_summary["maximum_directional_error"]
                    ),
                    direction_index=_format_direction(
                        step_summary["worst_direction"]
                    ),
                    passed=step_summary.get("passed"),
                )
            )
    lines.extend(
        [
            "",
            "## Dense replay",
            "",
            "| Workload | Family | Dimension | Condition | Solution relative L2 | Solution max abs | Gradient max abs | Passed |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in aggregate_result["dense_replay"]:
        lines.append(
            "| {workload_id} | {family} | {response_dimension} | {condition_number} | {solution_relative_l2} | {solution_max_abs} | {final_gradient_max_abs} | {passed} |".format(
                **{key: _format_number(value) for key, value in row.items()}
            )
        )
    lines.extend(
        [
            "",
            "## Backend ratios",
            "",
            "| Profile | Workload | Family | Z compact/direct compact time | Z compact/direct compact RSS | Z compact/Z detailed time | Z compact/Z detailed RSS |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in aggregate_result["backend_ratios"]:
        lines.append(
            "| {profile} | {workload_id} | {family} | {zvector_compact_time_over_direct_compact} | {zvector_compact_rss_over_direct_compact} | {zvector_compact_time_over_zvector_detailed} | {zvector_compact_rss_over_zvector_detailed} |".format(
                **{key: _format_number(value) for key, value in row.items()}
            )
        )
    failures = [
        failure
        for category in categories.values()
        for failure in category["failures"]
    ]
    performance_acceptance = aggregate_result["performance_acceptance"]
    for name, passed in performance_acceptance["checks"].items():
        if not passed:
            failures.append(
                {
                    "action": "campaign-performance-acceptance",
                    "workload_id": None,
                    "family": None,
                    "reasons": [f"campaign performance rule failed: {name}"],
                }
            )
    for name, passed in aggregate_result["selection_acceptance"]["checks"].items():
        if not passed:
            failures.append(
                {
                    "action": "campaign-selection-acceptance",
                    "workload_id": None,
                    "family": None,
                    "reasons": [f"campaign selection rule failed: {name}"],
                }
            )
    for name, passed in aggregate_result["campaign_integrity"]["checks"].items():
        if not passed:
            failures.append(
                {
                    "action": "campaign-integrity",
                    "workload_id": None,
                    "family": None,
                    "reasons": [f"campaign integrity rule failed: {name}"],
                }
            )
    lines.extend(["", "## Unresolved limits", ""])
    if failures:
        for failure in failures:
            reason = "; ".join(failure["reasons"]) or "unspecified category failure"
            lines.append(
                f"- `{failure['action']}` `{failure.get('workload_id')}` `{failure.get('family')}`: {reason}"
            )
    else:
        lines.append("All recorded category checks passed.")
    lines.extend(
        [
            "",
            "## Reproduction",
            "",
            "```bash",
            "uv sync --locked --python 3.11",
            f"uv run python validation/scientific_performance/scripts/run_campaign.py --run-root {aggregate_result['run_root']}",
            f"uv run python validation/scientific_performance/scripts/aggregate.py {aggregate_result['run_root']}",
            "```",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output", type=Path, default=REPORT_DIR / "aggregate.json")
    parser.add_argument("--markdown", type=Path, default=REPORT_DIR / "summary.md")
    arguments = parser.parse_args()
    result = aggregate(arguments.run_root.resolve())
    write_json(arguments.output, result)
    write_markdown(result, arguments.markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
