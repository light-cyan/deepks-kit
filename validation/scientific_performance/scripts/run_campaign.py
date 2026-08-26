"""Orchestrate the complete scientific-performance campaign in isolated children."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

from common import (
    REPOSITORY_DIR,
    RUN_DIR,
    VALIDATION_DIR,
    load_config,
    sha256_bytes,
    sha256_file,
    workload_by_id,
    write_json,
)


WORKER = Path(__file__).with_name("worker.py")
PYTHON = REPOSITORY_DIR / ".venv" / "bin" / "python"
TIME = Path("/usr/bin/time")
PHASES = (
    "verification",
    "setup",
    "preflight",
    "scientific",
    "dense",
    "conditioning",
    "benchmark",
    "selection",
    "invariance",
    "dft",
    "scanner",
    "force-data",
    "cross-revision",
    "aggregate",
)


@dataclass(frozen=True)
class ChildCase:
    phase: str
    action: str
    profile: str
    source: str
    workload: str | None = None
    family: str | None = None
    backend: str | None = None
    mode: str | None = None
    label: str | None = None

    @property
    def name(self) -> str:
        fields = [self.action]
        for value in (
            self.workload,
            self.family,
            self.backend,
            self.mode,
            self.label,
        ):
            if value:
                fields.append(value)
        return "__".join(fields).replace("/", "-")


def _revision(root: Path, revision: str = "HEAD") -> str:
    return subprocess.check_output(
        ["git", "rev-parse", revision], cwd=root, text=True
    ).strip()


def ensure_worktree(revision: str, label: str) -> Path:
    """Create or reuse one detached, tracked-clean source worktree."""
    target = VALIDATION_DIR / "worktrees" / label
    expected = _revision(REPOSITORY_DIR, revision)
    if target.exists():
        actual = _revision(target)
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=target,
            text=True,
        ).strip()
        if actual != expected or status:
            raise RuntimeError(
                f"existing validation worktree {target} is not the required clean revision"
            )
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "worktree", "add", "--detach", str(target), expected],
        cwd=REPOSITORY_DIR,
        check=True,
    )
    return target


def _validation_hash_at(source_root: Path) -> str:
    """Hash validation inputs and executable scripts in one source tree."""
    validation_dir = source_root / "validation" / "scientific_performance"
    paths = [validation_dir / "configs" / "campaign.json"]
    paths.extend(sorted((validation_dir / "geometries").glob("*.xyz")))
    paths.extend(sorted((validation_dir / "scripts").glob("*.py")))
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(validation_dir)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def source_snapshot(source_root: Path) -> dict[str, Any]:
    """Capture a stable source snapshot before campaign outputs are created."""
    tracked_status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=source_root,
        text=True,
    ).strip()
    working_tree_status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=source_root,
        text=True,
    ).strip()
    tracked_patch = subprocess.check_output(
        ["git", "diff", "--binary", "HEAD", "--"], cwd=source_root
    )
    config_path = (
        source_root
        / "validation"
        / "scientific_performance"
        / "configs"
        / "campaign.json"
    )
    return {
        "source_root": source_root.resolve(),
        "base_revision": _revision(source_root),
        "source_clean": working_tree_status == "",
        "working_tree_status": working_tree_status,
        "tracked_source_clean": tracked_status == "",
        "tracked_diff_status": tracked_status,
        "tracked_diff_sha256": sha256_bytes(tracked_patch),
        "config_hash": sha256_file(config_path),
        "validation_hash": _validation_hash_at(source_root),
    }


def _parse_time(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    patterns = {
        "user_seconds": r"User time \(seconds\):\s+(.+)",
        "system_seconds": r"System time \(seconds\):\s+(.+)",
        "cpu_percent": r"Percent of CPU this job got:\s+(.+)",
        "elapsed": r"Elapsed \(wall clock\) time.*:\s+(.+)",
        "peak_rss_kib": r"Maximum resident set size \(kbytes\):\s+(\d+)",
        "major_page_faults": r"Major \(requiring I/O\) page faults:\s+(\d+)",
        "minor_page_faults": r"Minor \(reclaiming a frame\) page faults:\s+(\d+)",
        "file_system_inputs": r"File system inputs:\s+(\d+)",
        "file_system_outputs": r"File system outputs:\s+(\d+)",
        "time_exit_status": r"Exit status:\s+(-?\d+)",
    }
    values: dict[str, Any] = {}
    for name, pattern in patterns.items():
        match = re.search(pattern, text)
        if not match:
            continue
        value = match.group(1).strip()
        if name in {
            "peak_rss_kib",
            "major_page_faults",
            "minor_page_faults",
            "file_system_inputs",
            "file_system_outputs",
            "time_exit_status",
        }:
            values[name] = int(value)
        elif name in {"user_seconds", "system_seconds"}:
            values[name] = float(value)
        else:
            values[name] = value
    values["raw"] = text
    return values


def _case_tier(case: ChildCase) -> dict[str, Any]:
    config = load_config()
    if case.workload:
        tier_name = workload_by_id(case.workload)["tier"]
    elif case.action == "force-data" and case.label == "large":
        tier_name = "maximum"
    else:
        tier_name = "bounded"
    return {"name": tier_name, **config["resource_tiers"][tier_name]}


def _category_state(result: dict[str, Any], category: str) -> str:
    passed = result.get("categories", {}).get(category, {}).get("passed")
    if passed is True:
        return "PASS"
    if passed is False:
        return "FAIL"
    return "NA"


def _process_state(result: dict[str, Any]) -> str:
    process = result.get("process")
    if not isinstance(process, dict) or process.get("exit_status") is None:
        return "INCOMPLETE"
    if process.get("exit_status") == 0 and not process.get("timeout"):
        return "PASS"
    return "FAIL"


def _print_result_state(prefix: str, case: ChildCase, result: dict[str, Any]) -> None:
    states = " ".join(
        f"{category}={_category_state(result, category)}"
        for category in ("scientific", "integrity", "performance", "resource")
    )
    print(
        f"{prefix} {case.phase} {case.name} process={_process_state(result)} {states}",
        flush=True,
    )
    for category in ("scientific", "integrity", "performance", "resource"):
        record = result.get("categories", {}).get(category, {})
        if record.get("passed") is False:
            print(
                f"REASON {case.phase} {case.name} {category}: "
                + "; ".join(record.get("reasons", []) or ["unspecified failure"]),
                flush=True,
            )


def run_child(
    case: ChildCase,
    run_root: Path,
    source_roots: dict[str, Path],
    source_snapshots: dict[str, dict[str, Any]],
    rerun: bool,
) -> dict[str, Any]:
    """Run one timed child and preserve every exit mode and resource counter."""
    case_root = run_root / case.phase / case.profile / case.source / case.name
    result_path = case_root / "result.json"
    stdout_path = case_root / "stdout.log"
    stderr_path = case_root / "stderr.log"
    time_path = case_root / "time.txt"
    if result_path.exists() and not rerun:
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        wrapper = existing.get("process", {})
        if wrapper.get("exit_status") == 0 and not wrapper.get("timeout"):
            _print_result_state("SKIP", case, existing)
            return existing
    case_root.mkdir(parents=True, exist_ok=True)
    source_root = source_roots[case.source]
    snapshot = source_snapshots[case.source]
    profile_config = load_config()["profiles"][case.profile]
    threads = str(profile_config["threads"])
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(source_root),
            "DEEPKS_SOURCE_ROOT": str(source_root),
            "VALIDATION_BASE_REVISION": snapshot["base_revision"],
            "VALIDATION_SOURCE_CLEAN": "1" if snapshot["source_clean"] else "0",
            "VALIDATION_WORKING_TREE_STATUS": snapshot["working_tree_status"],
            "VALIDATION_TRACKED_DIFF_STATUS": snapshot["tracked_diff_status"],
            "VALIDATION_TRACKED_DIFF_SHA256": snapshot["tracked_diff_sha256"],
            "VALIDATION_CONFIG_HASH": snapshot["config_hash"],
            "VALIDATION_INPUT_HASH": snapshot["validation_hash"],
            "VALIDATION_PROFILE": case.profile,
            "VALIDATION_ENV_LOCK_HASH": sha256_file(REPOSITORY_DIR / "uv.lock"),
            "VIRTUAL_ENV": str(REPOSITORY_DIR / ".venv"),
            "OPENBLAS_NUM_THREADS": threads,
            "OMP_NUM_THREADS": threads,
            "MKL_NUM_THREADS": threads,
            "NUMEXPR_NUM_THREADS": threads,
            "VECLIB_MAXIMUM_THREADS": threads,
        }
    )
    tier = _case_tier(case)
    command = [
        str(TIME),
        "-v",
        "-o",
        str(time_path),
        "timeout",
        "--signal=TERM",
        "--kill-after=30s",
        str(tier["timeout_seconds"]),
        "taskset",
        "-c",
        profile_config["cores"],
        "uv",
        "run",
        "--project",
        str(source_root),
        "--active",
        "--no-sync",
        "python",
        str(
            source_root
            / "validation"
            / "scientific_performance"
            / "scripts"
            / "worker.py"
        ),
        "--action",
        case.action,
        "--output",
        str(result_path),
    ]
    for option, value in (
        ("--workload", case.workload),
        ("--family", case.family),
        ("--backend", case.backend),
        ("--mode", case.mode),
        ("--label", case.label),
    ):
        if value is not None:
            command.extend((option, value))
    print(f"RUN  {case.phase} {case.name} {case.profile} {case.source}", flush=True)
    controller_path = case_root / "controller.json"
    controller = {
        "state": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "phase": case.phase,
        "action": case.action,
        "name": case.name,
        "profile": case.profile,
        "source": case.source,
        "source_revision": _revision(source_root),
        "source_snapshot": snapshot,
        "command": command,
    }
    write_json(controller_path, controller)
    timed_out = False
    return_code = None
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        try:
            completed = subprocess.run(
                command,
                cwd=source_root,
                env=environment,
                stdout=stdout,
                stderr=stderr,
                timeout=tier["timeout_seconds"] + 60,
                check=False,
            )
            return_code = completed.returncode
            timed_out = return_code == 124
        except subprocess.TimeoutExpired:
            timed_out = True
            return_code = 124
    if result_path.exists():
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            result = {"invalid_result": str(error)}
    else:
        result = {}
    resource = _parse_time(time_path)
    peak_rss_kib = resource.get("peak_rss_kib")
    rss_limit_kib = int(tier["rss_limit_gib"] * 1024 * 1024)
    resource_passed = (
        return_code == 0
        and not timed_out
        and (peak_rss_kib is None or peak_rss_kib <= rss_limit_kib)
    )
    wrapper = {
        "command": command,
        "source_root": source_root,
        "source_revision": _revision(source_root),
        "source_snapshot": snapshot,
        "stdout_path": stdout_path,
        "stderr_path": stderr_path,
        "time_path": time_path,
        "exit_status": return_code,
        "timeout": timed_out,
        "resource_tier": tier,
        "resource_measurement": resource,
    }
    result["process"] = wrapper
    categories = result.setdefault("categories", {})
    categories["resource"] = {
        "passed": resource_passed,
        "reasons": []
        if resource_passed
        else [
            "child timed out"
            if timed_out
            else "child exited unsuccessfully"
            if return_code != 0
            else "peak RSS exceeded the declared tier limit"
        ],
    }
    write_json(result_path, result)
    controller.update(
        {
            "state": "completed",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "process_state": _process_state(result),
            "category_states": {
                category: _category_state(result, category)
                for category in ("scientific", "integrity", "performance", "resource")
            },
        }
    )
    write_json(controller_path, controller)
    _print_result_state("DONE", case, result)
    return result


def archive_attempt(case: ChildCase, run_root: Path, attempt: int) -> dict[str, str]:
    """Preserve one failed or noisy performance attempt before its single retry."""
    case_root = run_root / case.phase / case.profile / case.source / case.name
    archived = {}
    for stem, suffix in (
        ("result", ".json"),
        ("stdout", ".log"),
        ("stderr", ".log"),
        ("time", ".txt"),
    ):
        source = case_root / f"{stem}{suffix}"
        if not source.exists():
            continue
        target = case_root / f"{stem}_attempt{attempt}{suffix}"
        source.replace(target)
        archived[stem] = str(target)
    return archived


def phase_cases(phase: str) -> list[ChildCase]:
    """Expand one plan phase into isolated child cases."""
    config = load_config()
    workloads = config["workloads"]
    if phase == "setup":
        return [ChildCase(phase, "checkpoint", "deterministic-1t", "current")]
    if phase == "verification":
        return [
            ChildCase(phase, "verification", "deterministic-1t", "current", label=label)
            for label in ("focused", "complete")
        ]
    if phase == "preflight":
        return [
            ChildCase(phase, "preflight", "deterministic-1t", "current", workload["id"], family)
            for workload in workloads
            for family in workload["families"]
        ]
    if phase == "scientific":
        return [
            ChildCase(phase, "scientific", "deterministic-1t", "current", workload["id"], family)
            for workload in workloads
            if workload["finite_difference"] != "none"
            for family in workload["families"]
        ]
    if phase == "dense":
        dense_workloads = {"S1-6-31G", "S2-def2-TZVP", "S3-def2-SVP"}
        return [
            ChildCase(phase, "dense", "deterministic-1t", "current", workload["id"], family)
            for workload in workloads
            if workload["id"] in dense_workloads
            for family in workload["families"]
        ]
    if phase == "conditioning":
        return [ChildCase(phase, "conditioning", "deterministic-1t", "current")]
    if phase == "benchmark":
        native = [
            ChildCase(phase, "native-benchmark", profile, "current", workload["id"], family)
            for profile in config["profiles"]
            for workload in workloads
            for family in workload["families"]
        ]
        corrected = [
            ChildCase(phase, "benchmark", profile, "current", workload["id"], family, backend, mode)
            for profile in config["profiles"]
            for workload in workloads
            for family in workload["families"]
            for backend in ("direct", "zvector")
            for mode in ("compact", "detailed")
        ]
        return native + corrected
    if phase == "selection":
        selected = {
            "L1-def2-SVP": ("rhf", "rks"),
            "L1-def2-TZVP": ("rhf", "rks"),
            "X1-def2-TZVP": ("rhf",),
        }
        subsets = [
            ChildCase(phase, "atom-subset", profile, "current", workload, family, backend, label=label)
            for profile in config["profiles"]
            for workload, families in selected.items()
            for family in families
            for backend in ("direct", "zvector")
            for label in ("full", "one_atom", "one_monomer", "half_atoms", "all_permuted")
        ]
        blocks = [
            ChildCase(phase, "coordinate-block", profile, "current", workload, "rhf", label=label)
            for profile in config["profiles"]
            for workload in ("L1-def2-SVP", "L1-def2-TZVP", "X1-def2-TZVP")
            for label in ("1", "2", "4", "8", "full")
        ]
        return subsets + blocks
    if phase == "invariance":
        selected = {"S1-6-31G", "L1-def2-SVP", "L1-def2-TZVP", "L2-def2-SVP"}
        return [
            ChildCase(phase, "invariance", "deterministic-1t", "current", workload["id"], family)
            for workload in workloads
            if workload["id"] in selected
            for family in workload["families"]
        ]
    if phase == "dft":
        selected = (
            ("S2-def2-TZVP", "rks"),
            ("L1-def2-SVP", "rks"),
            ("L1-def2-TZVP", "rks"),
            ("S3-def2-SVP", "uks"),
            ("L3-def2-SVP", "uks"),
        )
        return [
            ChildCase(phase, "dft-sequence", profile, "current", workload, family)
            for profile in config["profiles"]
            for workload, family in selected
        ]
    if phase == "scanner":
        return [
            ChildCase(phase, "scanner", profile, "current", config["scanner"]["workload"], "rhf")
            for profile in config["profiles"]
        ]
    if phase == "force-data":
        return [
            ChildCase(phase, "force-data", "deterministic-1t", "current", label=payload["id"])
            for payload in config["force_data"]["payloads"]
        ] + [ChildCase(phase, "force-data-physical", "deterministic-1t", "current", label="physical")]
    if phase == "cross-revision":
        selected = {"S1-6-31G", "S2-def2-TZVP", "S3-def2-SVP", "L1-def2-SVP", "L1-def2-TZVP", "L2-def2-SVP"}
        return [
            ChildCase(phase, "benchmark", profile, "historical", workload["id"], family, backend, mode)
            for profile in config["profiles"]
            for workload in workloads
            if workload["id"] in selected
            for family in workload["families"]
            for backend in ("direct", "zvector")
            for mode in ("detailed",)
        ]
    if phase == "aggregate":
        return []
    raise ValueError(f"unknown phase {phase!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phases", nargs="+", choices=PHASES, default=list(PHASES))
    parser.add_argument("--run-id")
    parser.add_argument(
        "--run-root",
        type=Path,
        help="Exact output directory; defaults to validation/scientific_performance/runs/<run-id>",
    )
    parser.add_argument(
        "--source-mode",
        choices=("current-worktree", "detached-worktree"),
        default="current-worktree",
        help="Execute the current working tree by default; detached mode must be explicit",
    )
    parser.add_argument(
        "--current-revision",
        help="Revision for explicit detached-worktree mode, or an assertion for current-worktree mode",
    )
    parser.add_argument(
        "--scientific-case",
        action="append",
        default=[],
        metavar="WORKLOAD:FAMILY",
        help="Run only a predeclared scientific case; repeat for multiple cases",
    )
    parser.add_argument("--rerun", action="store_true")
    arguments = parser.parse_args()
    validation_driver_revision = _revision(REPOSITORY_DIR)
    if arguments.source_mode == "current-worktree":
        current_revision = validation_driver_revision
        if (
            arguments.current_revision is not None
            and _revision(REPOSITORY_DIR, arguments.current_revision)
            != current_revision
        ):
            parser.error(
                "--current-revision does not match the current worktree HEAD; "
                "use --source-mode detached-worktree to test another revision"
            )
        current_source_root = REPOSITORY_DIR
    else:
        current_revision = _revision(
            REPOSITORY_DIR, arguments.current_revision or "HEAD"
        )
        current_source_root = ensure_worktree(
            current_revision, f"current-{current_revision[:12]}"
        )
    run_id = arguments.run_id or (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + current_revision[:12]
    )
    run_root = (
        arguments.run_root.resolve()
        if arguments.run_root is not None
        else RUN_DIR / run_id
    )
    if arguments.run_root is not None and arguments.run_id is None:
        run_id = run_root.name
    source_roots = {"current": current_source_root}
    if "cross-revision" in arguments.phases:
        historical_revision = load_config()["historical_revision"]
        historical_full = _revision(REPOSITORY_DIR, historical_revision)
        source_roots["historical"] = ensure_worktree(
            historical_full, f"historical-{historical_full[:12]}"
        )
    source_snapshots = {
        name: source_snapshot(root) for name, root in source_roots.items()
    }
    available_scientific_cases = {
        f"{case.workload}:{case.family}"
        for case in phase_cases("scientific")
    }
    unknown_scientific_cases = set(arguments.scientific_case).difference(
        available_scientific_cases
    )
    if unknown_scientific_cases:
        parser.error(
            "unknown --scientific-case values: "
            + ", ".join(sorted(unknown_scientific_cases))
        )
    run_root.mkdir(parents=True, exist_ok=True)
    current_snapshot = source_snapshots["current"]
    manifest = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "current_revision": current_revision,
        "base_revision": current_snapshot["base_revision"],
        "source_clean": current_snapshot["source_clean"],
        "working_tree_status": current_snapshot["working_tree_status"],
        "tracked_source_clean": current_snapshot["tracked_source_clean"],
        "tracked_diff_status": current_snapshot["tracked_diff_status"],
        "tracked_diff_sha256": current_snapshot["tracked_diff_sha256"],
        "validation_hash": current_snapshot["validation_hash"],
        "validation_driver_revision": validation_driver_revision,
        "validation_driver_tracked_diff": current_snapshot[
            "tracked_diff_status"
        ],
        "config_hash": current_snapshot["config_hash"],
        "environment_lock_hash": sha256_file(REPOSITORY_DIR / "uv.lock"),
        "phases": arguments.phases,
        "source_mode": arguments.source_mode,
        "source_roots": source_roots,
        "source_snapshots": source_snapshots,
        "scientific_cases": arguments.scientific_case,
    }
    write_json(run_root / "manifest.json", manifest)
    recorded_results = []
    for phase in arguments.phases:
        if phase == "aggregate":
            continue
        for case in phase_cases(phase):
            selector = f"{case.workload}:{case.family}"
            if (
                phase == "scientific"
                and arguments.scientific_case
                and selector not in arguments.scientific_case
            ):
                continue
            result = run_child(
                case,
                run_root,
                source_roots,
                source_snapshots,
                arguments.rerun,
            )
            process = result.get("process", {})
            performance_failed = (
                case.action in {"benchmark", "native-benchmark", "atom-subset", "coordinate-block", "dft-sequence"}
                and result.get("categories", {}).get("performance", {}).get("passed") is False
            )
            process_failed = process.get("exit_status") != 0 or process.get("timeout")
            if case.action in {"benchmark", "native-benchmark", "atom-subset", "coordinate-block", "dft-sequence"} and (performance_failed or process_failed):
                archived = archive_attempt(case, run_root, 1)
                result = run_child(
                    case, run_root, source_roots, source_snapshots, True
                )
                result["previous_attempt_artifacts"] = archived
                result_path = (
                    run_root
                    / case.phase
                    / case.profile
                    / case.source
                    / case.name
                    / "result.json"
                )
                write_json(result_path, result)
                process = result.get("process", {})
            recorded_results.append(result)
    if "aggregate" in arguments.phases:
        subprocess.run(
            [
                str(PYTHON),
                str(Path(__file__).with_name("aggregate.py")),
                str(run_root),
                "--output",
                str(run_root / "aggregate.json"),
                "--markdown",
                str(run_root / "SUMMARY.md"),
            ],
            cwd=REPOSITORY_DIR,
            check=True,
        )
    process_failures = sum(_process_state(result) != "PASS" for result in recorded_results)
    category_failures = Counter(
        category
        for result in recorded_results
        for category in ("scientific", "integrity", "performance", "resource")
        if _category_state(result, category) == "FAIL"
    )
    print(
        "Campaign children complete: "
        f"process_failures={process_failures} "
        + " ".join(
            f"{category}_failures={category_failures[category]}"
            for category in ("scientific", "integrity", "performance", "resource")
        ),
        flush=True,
    )
    return 1 if process_failures or category_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
