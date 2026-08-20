"""Run locked-environment, test-suite, and build verification commands."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

from common import REPORT_DIR, REPOSITORY_DIR, environment_metadata, write_json


COMMANDS = (
    ("locked_sync", ["uv", "sync", "--locked", "--python", "3.11"]),
    ("baseline_tests", ["uv", "run", "pytest", "tests/baseline"]),
    ("complete_tests", ["uv", "run", "pytest"]),
    ("build", ["uv", "build", "--out-dir", "validation/outputs/build"]),
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        choices=tuple(name for name, _command in COMMANDS),
    )
    arguments = parser.parse_args()
    environment = dict(os.environ)
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        environment[name] = "1"
    report_path = REPORT_DIR / "verification.json"
    previous = {}
    if arguments.only is not None and report_path.is_file():
        with report_path.open("r", encoding="utf-8") as stream:
            previous_report = json.load(stream)
        previous = {
            item["name"]: item for item in previous_report.get("commands", [])
        }
    selected = [
        (stage, command)
        for stage, command in COMMANDS
        if arguments.only is None or stage == arguments.only
    ]
    current = {}
    for stage, command in selected:
        process = subprocess.run(
            command,
            cwd=REPOSITORY_DIR,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        log_path = REPORT_DIR / f"verification_{stage}.log"
        log_path.write_text(process.stdout, encoding="utf-8")
        result = {
            "name": stage,
            "command": command,
            "return_code": process.returncode,
            "passed": process.returncode == 0,
            "log": str(log_path.relative_to(REPOSITORY_DIR)),
        }
        current[stage] = result
        print(json.dumps(result), flush=True)
    results = [
        current.get(stage, previous.get(stage))
        for stage, _command in COMMANDS
    ]
    if any(item is None for item in results):
        raise RuntimeError("partial verification requires a complete existing report")
    report = {
        "stage": "verification",
        "passed": all(item["passed"] for item in results),
        "environment": environment_metadata(),
        "commands": results,
    }
    write_json(report_path, report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
