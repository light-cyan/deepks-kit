"""Compare one relaxed descriptor derivative with fresh-reference finite differences."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

for _thread_environment_name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_thread_environment_name] = "1"

import numpy as np


REPOSITORY_DIR = Path(__file__).resolve().parents[2]
VALIDATION_DIR = REPOSITORY_DIR / "validation" / "scientific_performance"
sys.path.insert(0, str(VALIDATION_DIR / "scripts"))

from common import (  # noqa: E402
    configure_threads,
    diagnostics_dict,
    error_norms,
    fresh_reference,
    hash_array,
    make_method,
    state_continuity,
    workload_by_id,
    workload_geometry,
    write_json,
)


AXES = ("x", "y", "z")


def _point(workload: dict[str, Any], family: str, central_reference, model, coordinates: np.ndarray) -> dict[str, Any]:
    started = time.perf_counter()
    reference = fresh_reference(workload, family, coordinates)
    continuity = state_continuity(central_reference, reference)
    if not continuity["accepted"]:
        raise RuntimeError("the displaced reference changed the intended electronic state")
    method = make_method(reference, model)
    return {
        "coordinates_hash": hash_array(coordinates),
        "energy": float(method.kernel()),
        "descriptor": np.asarray(method.descriptor(), dtype=np.float64),
        "state_continuity": continuity,
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", required=True)
    parser.add_argument("--family", required=True, choices=("rhf", "uhf", "rks", "uks"))
    parser.add_argument("--atom", required=True, type=int)
    parser.add_argument("--axis", required=True, choices=AXES)
    parser.add_argument("--steps", required=True, nargs="+", type=float)
    parser.add_argument("--conv-tol", type=float)
    parser.add_argument("--conv-tol-grad", type=float)
    parser.add_argument("--max-cycle", type=int)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    configure_threads(1)
    workload = workload_by_id(arguments.workload)
    scf_controls = dict(workload.get("scf_overrides", {}))
    for name in ("conv_tol", "conv_tol_grad", "max_cycle"):
        value = getattr(arguments, name)
        if value is not None:
            scf_controls[name] = value
    workload["scf_overrides"] = scf_controls
    _, central_coordinates = workload_geometry(workload)
    axis = AXES.index(arguments.axis)
    central_reference = fresh_reference(workload, arguments.family)
    from common import deterministic_model

    model = deterministic_model()
    method = make_method(central_reference, model)
    central_energy = float(method.kernel())
    driver = method.nuc_grad_method(backend="direct", retain_details=True)
    central_gradient = np.asarray(driver.kernel(), dtype=np.float64)
    analytic_relaxed = np.asarray(driver.dq_dR_relaxed[arguments.atom, axis], dtype=np.float64)
    analytic_explicit = np.asarray(driver.dq_dR_explicit[arguments.atom, axis], dtype=np.float64)
    analytic_response = np.asarray(driver.dq_dR_response[arguments.atom, axis], dtype=np.float64)
    output = arguments.output or Path(__file__).with_name("results") / f"descriptor__{arguments.workload}__{arguments.family}__atom{arguments.atom}_{arguments.axis}.json"
    result = {
        "experiment": "descriptor_point",
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPOSITORY_DIR, text=True).strip(),
        "workload_id": arguments.workload,
        "family": arguments.family,
        "scf_overrides": scf_controls,
        "coordinate": {"atom": arguments.atom, "axis": axis, "axis_name": arguments.axis},
        "central": {
            "coordinates_hash": hash_array(central_coordinates),
            "energy": central_energy,
            "gradient_component": float(central_gradient[arguments.atom, axis]),
            "analytic_descriptor_relaxed": analytic_relaxed,
            "analytic_descriptor_explicit": analytic_explicit,
            "analytic_descriptor_response": analytic_response,
            "response_diagnostics": diagnostics_dict(driver),
        },
        "steps": [],
    }
    write_json(output, result)
    for step in arguments.steps:
        points = []
        for sign in (-1, 1):
            coordinates = central_coordinates.copy()
            coordinates[arguments.atom, axis] += sign * step
            points.append(_point(workload, arguments.family, central_reference, model, coordinates))
        energy_derivative = (points[1]["energy"] - points[0]["energy"]) / (2.0 * step)
        descriptor_derivative = (points[1]["descriptor"] - points[0]["descriptor"]) / (2.0 * step)
        difference = analytic_relaxed - descriptor_derivative
        flat_index = int(np.argmax(np.abs(difference)))
        descriptor_atom, descriptor_feature = (int(value) for value in np.unravel_index(flat_index, difference.shape))
        step_result = {
            "step_bohr": step,
            "minus": points[0],
            "plus": points[1],
            "energy_derivative": energy_derivative,
            "energy_gradient_error": float(central_gradient[arguments.atom, axis] - energy_derivative),
            "descriptor_derivative": descriptor_derivative,
            "descriptor_error": error_norms(analytic_relaxed, descriptor_derivative),
            "worst_descriptor_element": {
                "descriptor_atom": descriptor_atom,
                "descriptor_feature": descriptor_feature,
                "analytic_relaxed": float(analytic_relaxed[descriptor_atom, descriptor_feature]),
                "finite_difference": float(descriptor_derivative[descriptor_atom, descriptor_feature]),
                "signed_error_per_bohr": float(difference[descriptor_atom, descriptor_feature]),
            },
        }
        result["steps"].append(step_result)
        write_json(output, result)
        print(json.dumps({"step_bohr": step, "energy_gradient_error": step_result["energy_gradient_error"], "descriptor_error": step_result["descriptor_error"], "worst_descriptor_element": step_result["worst_descriptor_element"]}, sort_keys=True), flush=True)
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
