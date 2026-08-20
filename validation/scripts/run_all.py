"""Execute every validation tier and publish verification and summary reports."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

from common import REPORT_DIR, REPOSITORY_DIR


def _environment() -> dict[str, str]:
    environment = dict(os.environ)
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        environment[name] = "1"
    return environment


def _run(command: list[str], log_name: str) -> dict:
    process = subprocess.run(
        command,
        cwd=REPOSITORY_DIR,
        env=_environment(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log = REPORT_DIR / log_name
    log.write_text(process.stdout, encoding="utf-8")
    return {
        "command": command,
        "return_code": process.returncode,
        "passed": process.returncode == 0,
        "log": str(log.relative_to(REPOSITORY_DIR)),
    }


def main() -> int:
    python = sys.executable
    stages = [
        ([python, "validation/scripts/run_scientific.py", "--family", family], f"run_scientific_{family}.log")
        for family in ("rhf", "rks", "uhf", "uks")
    ]
    stages.extend(
        ([python, "validation/scripts/run_training.py", "--workflow", workflow], f"run_training_{workflow}.log")
        for workflow in ("teacher", "mp2")
    )
    stages.append(
        ([python, "validation/scripts/run_performance.py"], "run_performance.log")
    )
    stage_results = [_run(command, log) for command, log in stages]
    verification = _run(
        [python, "validation/scripts/run_verification.py"],
        "run_verification.log",
    )
    final = _run(
        [python, "validation/scripts/finalize.py"], "finalize.log"
    )
    return 0 if all(item["passed"] for item in stage_results) and verification["passed"] and final["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
