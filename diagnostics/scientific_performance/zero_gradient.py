"""Reproduce one zero-correction DeePHF versus native-gradient comparison."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

for _thread_environment_name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_thread_environment_name] = "1"

import numpy as np


REPOSITORY_DIR = Path(__file__).resolve().parents[2]
VALIDATION_DIR = REPOSITORY_DIR / "validation" / "scientific_performance"
sys.path.insert(0, str(VALIDATION_DIR / "scripts"))

from common import configure_threads, error_norms, fresh_reference, make_method, workload_by_id, write_json  # noqa: E402


AXES = ("x", "y", "z")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", required=True)
    parser.add_argument("--family", required=True, choices=("rhf", "uhf", "rks", "uks"))
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    configure_threads(1)
    workload = workload_by_id(arguments.workload)
    reference = fresh_reference(workload, arguments.family)
    default_native_gradient = np.asarray(reference.nuc_grad_method().kernel(), dtype=np.float64)
    full_grid_driver = reference.nuc_grad_method()
    if arguments.family in {"rks", "uks"}:
        full_grid_driver.grids = reference.grids
        full_grid_driver.grid_response = True
    full_grid_native_gradient = np.asarray(full_grid_driver.kernel(), dtype=np.float64)
    method = make_method(reference, None)
    zero_energy = float(method.kernel())
    driver = method.nuc_grad_method(backend="direct", retain_details=True)
    zero_gradient = np.asarray(driver.kernel(), dtype=np.float64)
    difference = zero_gradient - full_grid_native_gradient
    flat_index = int(np.argmax(np.abs(difference)))
    atom, axis = (int(value) for value in np.unravel_index(flat_index, difference.shape))
    output = arguments.output or Path(__file__).with_name("results") / f"zero_gradient__{arguments.workload}__{arguments.family}.json"
    result = {
        "experiment": "zero_gradient",
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "workload_id": arguments.workload,
        "family": arguments.family,
        "native_energy": float(reference.e_tot),
        "zero_correction_energy": zero_energy,
        "energy_error_hartree": zero_energy - float(reference.e_tot),
        "zero_vs_default_native_error": error_norms(zero_gradient, default_native_gradient),
        "zero_vs_full_grid_native_error": error_norms(zero_gradient, full_grid_native_gradient),
        "worst_component": {
            "atom": atom,
            "axis": axis,
            "axis_name": AXES[axis],
            "full_grid_native_gradient": float(full_grid_native_gradient[atom, axis]),
            "zero_correction_gradient": float(zero_gradient[atom, axis]),
            "signed_error_hartree_per_bohr": float(difference[atom, axis]),
        },
        "default_native_gradient": default_native_gradient,
        "full_grid_native_gradient": full_grid_native_gradient,
        "zero_correction_gradient": zero_gradient,
        "difference": difference,
    }
    write_json(output, result)
    print(json.dumps({"output": str(output), "zero_vs_default_native_error": result["zero_vs_default_native_error"], "zero_vs_full_grid_native_error": result["zero_vs_full_grid_native_error"], "worst_component": result["worst_component"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
