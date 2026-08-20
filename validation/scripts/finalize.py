"""Assemble the release-validation manifest and concise human-readable report."""

from __future__ import annotations

import json
from pathlib import Path
import sys

from common import (
    REPORT_DIR,
    REPOSITORY_DIR,
    VALIDATION_DIR,
    environment_metadata,
    sha256_file,
    write_json,
)


STAGE_REPORTS = (
    "scientific_rhf.json",
    "scientific_rks.json",
    "scientific_uhf.json",
    "scientific_uks.json",
    "training_teacher.json",
    "training_mp2.json",
    "performance.json",
    "verification.json",
)


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _artifact_manifest() -> list[dict]:
    artifacts = []
    excluded = {
        REPORT_DIR / "artifact_manifest.json",
        REPORT_DIR / "summary.md",
        REPORT_DIR / "master.json",
    }
    for path in sorted(VALIDATION_DIR.rglob("*")):
        if not path.is_file() or path in excluded or path.name in {"draft.md", "TODO.md"}:
            continue
        artifacts.append(
            {
                "path": str(path.relative_to(REPOSITORY_DIR)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return artifacts


def _scientific_line(report: dict) -> str:
    if "finite_difference" not in report:
        return f"- {report['stage']}: failed closed with {report.get('exception_type', 'Error')}: {report.get('message', 'unknown error')}"
    steps = report["finite_difference"]["steps"]
    gradient_error = max(item["total_gradient"]["max_abs"] for item in steps.values())
    descriptor_error = max(
        item["relaxed_descriptor_jacobian"]["max_abs"]
        for item in steps.values()
    )
    backend_error = report["central"]["direct_vs_zvector_total"]["max_abs"]
    signal = report["finite_difference"]["anti_vacuity"][
        "response_descriptor_max_abs_bohr_inverse"
    ]
    status = "PASS" if report["passed"] else "FAIL"
    return f"- {report['family']} {report['system']}: {status}; direct/Z-vector max error {backend_error:.3e} Eh/Bohr; finite-difference gradient max error {gradient_error:.3e} Eh/Bohr; relaxed-descriptor max error {descriptor_error:.3e} Bohr^-1; response signal {signal:.3e} Bohr^-1."


def _training_line(report: dict) -> str:
    if "heldout_comparison" not in report:
        return f"- {report['stage']}: failed closed with {report.get('exception_type', 'Error')}: {report.get('message', 'unknown error')}"
    comparison = report["heldout_comparison"]
    baseline = comparison["zero_correction_rhf"]
    energy_only = comparison["energy_only"]
    combined = comparison["energy_plus_force"]
    status = "PASS" if report["passed"] else "FAIL"
    return f"- {report['workflow']} water-dimer workflow: {status}; held-out RHF baseline energy/force RMSE {baseline['energy']['rmse']:.3e} Eh and {baseline['force']['rmse']:.3e} Eh/Bohr; energy-only {energy_only['energy']['rmse']:.3e} and {energy_only['force']['rmse']:.3e}; energy-plus-force {combined['energy']['rmse']:.3e} and {combined['force']['rmse']:.3e}."


def _performance_lines(report: dict) -> list[str]:
    if "samples" not in report:
        return [f"- Performance collection failed closed with {report.get('exception_type', 'Error')}: {report.get('message', 'unknown error')}"]
    lines = []
    for name, sample in report["samples"].items():
        lines.append(
            f"- {name}: median {sample['median_seconds']:.6f} s, minimum {sample['minimum_seconds']:.6f} s, maximum {sample['maximum_seconds']:.6f} s, MAD {sample['median_absolute_deviation_seconds']:.6f} s."
        )
    return lines


def finalize() -> dict:
    reports = {}
    for filename in STAGE_REPORTS:
        path = REPORT_DIR / filename
        if path.is_file():
            reports[filename] = _load(path)
        else:
            reports[filename] = {
                "stage": filename.removesuffix(".json"),
                "passed": False,
                "exception_type": "MissingStageReport",
                "message": f"{filename} was not published",
            }
    artifacts = _artifact_manifest()
    write_json(REPORT_DIR / "artifact_manifest.json", artifacts)
    stages = {
        report["stage"]: bool(report.get("passed", False))
        for report in reports.values()
    }
    master = {
        "stage": "release_validation",
        "passed": all(stages.values()),
        "environment": environment_metadata(),
        "stages": stages,
        "stage_reports": {
            report["stage"]: str((REPORT_DIR / filename).relative_to(REPOSITORY_DIR))
            for filename, report in reports.items()
        },
        "artifact_count": len(artifacts),
        "artifact_manifest": str(
            (REPORT_DIR / "artifact_manifest.json").relative_to(REPOSITORY_DIR)
        ),
    }
    write_json(REPORT_DIR / "master.json", master)

    scientific = [
        _scientific_line(reports[f"scientific_{family}.json"])
        for family in ("rhf", "rks", "uhf", "uks")
    ]
    training = [
        _training_line(reports[f"training_{workflow}.json"])
        for workflow in ("teacher", "mp2")
    ]
    performance = _performance_lines(reports["performance.json"])
    verification_report = reports["verification.json"]
    verification = []
    for command in verification_report.get("commands", []):
        status = "PASS" if command["passed"] else "FAIL"
        verification.append(
            f"- {status}: `{' '.join(command['command'])}`; log `{command['log']}`."
        )
    if not verification:
        verification.append(
            f"- Verification failed closed with {verification_report.get('exception_type', 'MissingResults')}: {verification_report.get('message', 'no command results were published')}"
        )
    overall = "PASS" if master["passed"] else "FAIL"
    content = "\n".join(
        [
            "# DeePHF Release Validation Report",
            "",
            f"Overall status: {overall}.",
            "",
            "## Scientific matrices",
            "",
            *scientific,
            "",
            "## Water-dimer training",
            "",
            *training,
            "",
            "## Performance samples",
            "",
            *performance,
            "",
            "## Verification",
            "",
            *verification,
            "",
            "## Archive",
            "",
            f"The validation archive contains {len(artifacts)} hashed files listed in `validation/reports/artifact_manifest.json`; machine-readable stage status is in `validation/reports/master.json`.",
            "",
        ]
    )
    (REPORT_DIR / "summary.md").write_text(content, encoding="utf-8")
    return master


if __name__ == "__main__":
    result = finalize()
    print(json.dumps({"stage": result["stage"], "passed": result["passed"]}))
    sys.exit(0 if result["passed"] else 1)

