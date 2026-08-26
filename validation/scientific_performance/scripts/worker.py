"""Execute one isolated scientific-performance validation child action."""

from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack, nullcontext
import gc
import inspect
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
import tracemalloc
from typing import Any
from unittest import mock

import numpy as np
import pyscf
import torch
from pyscf import dft, gto, mp, scf

from common import (
    CHECKPOINT_DIR,
    REPORT_DIR,
    RUN_DIR,
    base_result,
    configure_threads,
    deterministic_directions,
    deterministic_model,
    diagnostics_dict,
    effective_scf_controls,
    error_norms,
    finite_difference_components,
    finite_difference_steps,
    fresh_reference,
    gradient_partitions,
    hash_array,
    json_safe,
    load_config,
    make_method,
    max_abs,
    minimum_orbital_gaps,
    model_hash,
    report_exception,
    response_dimensions,
    sha256_file,
    state_continuity,
    statistics,
    workload_by_id,
    workload_geometry,
    write_json,
)


def _profile() -> tuple[str, int]:
    profile = os.environ.get("VALIDATION_PROFILE", "deterministic-1t")
    threads = int(load_config()["profiles"][profile]["threads"])
    return profile, threads


def _pyscf_native_gradient(reference, family: str) -> np.ndarray:
    """Evaluate the PySCF gradient on the same finite-grid energy surface."""
    driver = reference.nuc_grad_method()
    if family in {"rks", "uks"}:
        driver.grids = reference.grids
        driver.grid_response = True
    return np.asarray(driver.kernel(), dtype=np.float64)


@contextmanager
def _profile_call_counters(counts: dict[str, int], targets):
    """Count Python call events without changing guarded implementation objects."""
    labels_by_code: dict[Any, list[str]] = {}
    for function, label in targets:
        code = getattr(function, "__code__", None)
        if code is not None:
            labels_by_code.setdefault(code, []).append(label)
    previous = sys.getprofile()

    def profiler(frame, event, argument):
        if event == "call":
            for label in labels_by_code.get(frame.f_code, ()):
                counts[label] = counts.get(label, 0) + 1
        if previous is not None:
            previous(frame, event, argument)

    sys.setprofile(profiler)
    try:
        yield
    finally:
        sys.setprofile(previous)


def _stability_status(reference) -> dict[str, Any]:
    start = time.perf_counter()
    output = reference.stability(
        internal=True,
        external=False,
        return_status=True,
        verbose=0,
    )
    booleans = [bool(value) for value in output if isinstance(value, (bool, np.bool_))]
    return {
        "elapsed_seconds": time.perf_counter() - start,
        "reported_status": booleans,
        "stable": bool(booleans and booleans[0]),
    }


def _grid_provenance(reference) -> dict[str, Any] | None:
    if not hasattr(reference, "grids"):
        return None
    grid = reference.grids
    return {
        "point_count": int(np.asarray(grid.weights).size),
        "coordinate_hash": hash_array(np.asarray(grid.coords)),
        "weight_hash": hash_array(np.asarray(grid.weights)),
        "atom_grid": grid.atom_grid,
        "prune": grid.prune,
        "alignment": grid.alignment,
        "cutoff": grid.cutoff,
        "xc": reference.xc,
        "small_rho_cutoff": reference.small_rho_cutoff,
    }


def _reference_summary(reference) -> dict[str, Any]:
    return {
        "converged": bool(reference.converged),
        "energy": float(reference.e_tot),
        "ao_count": int(reference.mol.nao),
        "atom_count": int(reference.mol.natm),
        **response_dimensions(reference),
        "minimum_orbital_gaps": minimum_orbital_gaps(reference),
        "occupations": np.asarray(reference.mo_occ),
        "ao_labels": tuple(reference.mol.ao_labels()),
        "grid": _grid_provenance(reference),
    }


def _driver(method, backend: str, detailed: bool, atmlst=None, **options):
    parameters = inspect.signature(method.nuc_grad_method).parameters
    if "retain_details" in parameters:
        driver = method.nuc_grad_method(
            backend=backend,
            retain_details=detailed,
            **options,
        )
    else:
        driver = method.nuc_grad_method(backend=backend, **options)
    gradient = np.asarray(driver.kernel(atmlst=atmlst), dtype=np.float64)
    return driver, gradient


def action_checkpoint(output: Path, _workload_id: str | None, _family: str | None) -> dict[str, Any]:
    profile, threads = _profile()
    result = base_result("checkpoint", profile, threads)
    model = deterministic_model()
    checkpoint = CHECKPOINT_DIR / "frozen_corrnet.pth"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model.save(checkpoint, validation_model_hash=model_hash(model))
    loaded = type(model).load(checkpoint, strict=True).double().eval()
    result["checkpoint"] = checkpoint
    result["checkpoint_hash"] = sha256_file(checkpoint)
    result["model_hash"] = model_hash(model)
    result["reload_model_hash"] = model_hash(loaded)
    passed = result["model_hash"] == result["reload_model_hash"]
    result["categories"]["integrity"] = {
        "passed": passed,
        "reasons": [] if passed else ["checkpoint reload changed the model state"],
    }
    write_json(output, result)
    return result


def action_verification(output: Path, label: str) -> dict[str, Any]:
    profile, threads = _profile()
    result = base_result("verification", profile, threads)
    focused = [
        "tests/response_scalability",
        "tests/analytic_forces",
        "tests/uhf_analytic_forces",
        "tests/rks_analytic_forces",
        "tests/uks_analytic_forces",
        "tests/zvector_inference",
        "tests/uhf_zvector_inference",
        "tests/rks_zvector_inference",
        "tests/uks_zvector_inference",
        "tests/baseline",
    ]
    targets = focused if label == "focused" else ["tests"]
    command = [sys.executable, "-m", "pytest", *targets]
    start = time.perf_counter()
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    result["verification"] = {
        "label": label,
        "command": command,
        "elapsed_seconds": time.perf_counter() - start,
        "exit_status": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    passed = completed.returncode == 0
    result["categories"]["integrity"] = {
        "passed": passed,
        "reasons": [] if passed else [f"{label} pytest verification failed"],
    }
    write_json(output, result)
    return result


def action_preflight(output: Path, workload_id: str, family: str) -> dict[str, Any]:
    profile, threads = _profile()
    workload = workload_by_id(workload_id)
    result = base_result("preflight", profile, threads, workload, family)
    result["comparison_targets"] = ["direct-detailed", "zvector-detailed"]
    reference = fresh_reference(workload, family)
    model = deterministic_model()
    method = make_method(reference, model)
    result["reference"] = _reference_summary(reference)
    result["stability"] = _stability_status(reference)
    descriptor_diagnostics = method.validate_force_compatibility()
    result["descriptor_diagnostics"] = descriptor_diagnostics
    direct_driver, direct_gradient = _driver(method, "direct", True)
    zvector_driver, zvector_gradient = _driver(method, "zvector", True)
    result["pilot"] = {
        "direct_diagnostics": diagnostics_dict(direct_driver),
        "zvector_diagnostics": diagnostics_dict(zvector_driver),
        "direct_partitions": gradient_partitions(direct_driver),
        "zvector_partitions": gradient_partitions(zvector_driver),
        "gradient_error": error_norms(zvector_gradient, direct_gradient),
        "correction_gradient_response_max_abs": max_abs(
            direct_driver.correction_gradient_response
        ),
        "descriptor_response_max_abs": max_abs(direct_driver.dq_dR_response),
    }
    acceptance = load_config()["acceptance"]
    reasons = []
    if not result["stability"]["stable"]:
        reasons.append("native internal stability analysis rejected the geometry")
    if result["pilot"]["gradient_error"]["max_abs"] > acceptance["direct_zvector_max_abs_hartree_per_bohr"]:
        reasons.append("direct and Z-vector pilot gradients disagree")
    if result["pilot"]["correction_gradient_response_max_abs"] < acceptance["response_signal_min_hartree_per_bohr"]:
        reasons.append("correction response signal is below the anti-vacuity floor")
    if result["pilot"]["descriptor_response_max_abs"] < acceptance["descriptor_response_signal_min_per_bohr"]:
        reasons.append("descriptor response signal is below the anti-vacuity floor")
    result["categories"]["scientific"] = {"passed": not reasons, "reasons": reasons}
    result["categories"]["integrity"] = {"passed": True, "reasons": []}
    write_json(output, result)
    return result


def _finite_difference_point(
    workload: dict[str, Any],
    family: str,
    central_reference,
    coordinates: np.ndarray,
    model,
    diagnostic_context: dict[str, Any],
) -> dict[str, Any]:
    context = {
        **diagnostic_context,
        "coordinates_hash": hash_array(coordinates),
    }
    try:
        reference = fresh_reference(workload, family, coordinates)
    except Exception as error:
        context["failure_stage"] = "fresh_reference"
        raise FiniteDifferencePointError(context, error) from error
    try:
        continuity = state_continuity(central_reference, reference)
    except Exception as error:
        context["failure_stage"] = "state_continuity"
        raise FiniteDifferencePointError(context, error) from error
    if not continuity["accepted"]:
        context["failure_stage"] = "state_continuity"
        context["state_continuity"] = continuity
        raise FiniteDifferencePointError(
            context,
            RuntimeError("a displaced reference changed the intended electronic state"),
        )
    try:
        method = make_method(reference, model)
    except Exception as error:
        context["failure_stage"] = "method_validation"
        raise FiniteDifferencePointError(context, error) from error
    try:
        energy = float(method.kernel())
        descriptor = np.asarray(method.descriptor(), dtype=np.float64)
    except Exception as error:
        context["failure_stage"] = "energy_or_descriptor"
        raise FiniteDifferencePointError(context, error) from error
    return {
        "energy": energy,
        "descriptor": descriptor,
        "state": continuity,
    }


class FiniteDifferencePointError(RuntimeError):
    """Bind one displaced-point failure to its exact finite-difference input."""

    def __init__(self, diagnostic_context: dict[str, Any], error: BaseException):
        self.diagnostic_context = diagnostic_context
        label = ", ".join(
            f"{name}={value}"
            for name, value in diagnostic_context.items()
            if name != "state_continuity"
        )
        super().__init__(f"finite-difference point failed ({label}): {type(error).__name__}: {error}")


def _complete_finite_differences(
    workload: dict[str, Any],
    family: str,
    reference,
    model,
) -> dict[str, Any]:
    _, central_coordinates = workload_geometry(workload)
    components = finite_difference_components(workload)
    directions = deterministic_directions(workload)
    result: dict[str, Any] = {"steps": {}}
    for step in finite_difference_steps(workload, family):
        step_record: dict[str, Any] = {
            "step_bohr": step,
            "components": [],
            "directions": [],
        }
        for atom, axis in components:
            points = []
            for sign in (-1, 1):
                coordinates = central_coordinates.copy()
                coordinates[atom, axis] += sign * step
                points.append(
                    _finite_difference_point(
                        workload,
                        family,
                        reference,
                        coordinates,
                        model,
                        {
                            "kind": "component",
                            "step_bohr": step,
                            "atom": atom,
                            "axis": axis,
                            "axis_name": ("x", "y", "z")[axis],
                            "sign": sign,
                        },
                    )
                )
            energy_derivative = (points[1]["energy"] - points[0]["energy"]) / (2.0 * step)
            descriptor_derivative = (
                np.asarray(points[1]["descriptor"])
                - np.asarray(points[0]["descriptor"])
            ) / (2.0 * step)
            step_record["components"].append(
                {
                    "atom": atom,
                    "axis": axis,
                    "axis_name": ("x", "y", "z")[axis],
                    "energy_derivative": energy_derivative,
                    "descriptor_derivative": descriptor_derivative,
                    "minus_state": points[0]["state"],
                    "plus_state": points[1]["state"],
                }
            )
        for direction_index, direction in enumerate(directions):
            points = []
            for sign in (-1, 1):
                coordinates = central_coordinates + sign * step * direction
                points.append(
                    _finite_difference_point(
                        workload,
                        family,
                        reference,
                        coordinates,
                        model,
                        {
                            "kind": "direction",
                            "step_bohr": step,
                            "direction_index": direction_index,
                            "sign": sign,
                        },
                    )
                )
            step_record["directions"].append(
                {
                    "index": direction_index,
                    "direction": direction,
                    "energy_derivative": (points[1]["energy"] - points[0]["energy"]) / (2.0 * step),
                    "minus_state": points[0]["state"],
                    "plus_state": points[1]["state"],
                }
            )
        result["steps"][f"{step:.1e}"] = step_record
    return result


def _assess_finite_differences(
    finite_difference: dict[str, Any],
    analytic_gradient: np.ndarray,
    explicit_only_gradient: np.ndarray,
    relaxed_descriptor: np.ndarray,
) -> dict[str, Any]:
    """Annotate and summarize every predeclared finite-difference step."""
    per_step: dict[str, Any] = {}
    for step_label, step_record in finite_difference["steps"].items():
        component_errors = []
        descriptor_errors = []
        explicit_errors = []
        directional_errors = []
        worst_component = None
        worst_descriptor = None
        worst_direction = None
        for item in step_record["components"]:
            atom = item["atom"]
            axis = item["axis"]
            item["analytic_gradient"] = analytic_gradient[atom, axis]
            item["gradient_error"] = float(
                item["analytic_gradient"] - item["energy_derivative"]
            )
            item["explicit_only_gradient"] = float(
                explicit_only_gradient[atom, axis]
            )
            item["explicit_only_error"] = float(
                item["explicit_only_gradient"] - item["energy_derivative"]
            )
            descriptor_delta = np.asarray(relaxed_descriptor[atom, axis]) - np.asarray(
                item["descriptor_derivative"]
            )
            item["relaxed_descriptor_error"] = error_norms(
                relaxed_descriptor[atom, axis], item["descriptor_derivative"]
            )
            component_error = abs(item["gradient_error"])
            descriptor_error = item["relaxed_descriptor_error"]["max_abs"]
            component_errors.append(component_error)
            descriptor_errors.append(descriptor_error)
            explicit_errors.append(abs(item["explicit_only_error"]))
            if worst_component is None or component_error > worst_component["absolute_error"]:
                worst_component = {
                    "atom": atom,
                    "axis": axis,
                    "axis_name": item["axis_name"],
                    "signed_error": item["gradient_error"],
                    "absolute_error": component_error,
                }
            flat_index = int(np.argmax(np.abs(descriptor_delta)))
            descriptor_index = tuple(
                int(value)
                for value in np.unravel_index(flat_index, descriptor_delta.shape)
            )
            descriptor_signed_error = float(descriptor_delta[descriptor_index])
            if worst_descriptor is None or descriptor_error > worst_descriptor["absolute_error"]:
                worst_descriptor = {
                    "atom": atom,
                    "axis": axis,
                    "axis_name": item["axis_name"],
                    "descriptor_index": descriptor_index,
                    "signed_error": descriptor_signed_error,
                    "absolute_error": descriptor_error,
                }
        for item in step_record["directions"]:
            analytic = float(
                np.einsum("ax,ax->", analytic_gradient, item["direction"])
            )
            item["analytic_derivative"] = analytic
            item["error"] = analytic - item["energy_derivative"]
            directional_error = abs(item["error"])
            directional_errors.append(directional_error)
            if worst_direction is None or directional_error > worst_direction["absolute_error"]:
                worst_direction = {
                    "direction_index": item["index"],
                    "signed_error": item["error"],
                    "absolute_error": directional_error,
                }
        maximum_component_error = max(component_errors, default=0.0)
        maximum_explicit_only_error = max(explicit_errors, default=0.0)
        step_summary = {
            "step_bohr": step_record["step_bohr"],
            "component_count": len(step_record["components"]),
            "direction_count": len(step_record["directions"]),
            "maximum_component_error": maximum_component_error,
            "maximum_directional_error": max(directional_errors, default=0.0),
            "maximum_relaxed_descriptor_error": max(descriptor_errors, default=0.0),
            "maximum_explicit_only_error": maximum_explicit_only_error,
            "explicit_to_complete_error_ratio": (
                maximum_explicit_only_error
                / max(maximum_component_error, np.finfo(float).tiny)
            ),
            "worst_component": worst_component,
            "worst_descriptor": worst_descriptor,
            "worst_direction": worst_direction,
        }
        step_record["summary"] = step_summary
        per_step[step_label] = step_summary

    def aggregate_worst(metric: str, record: str):
        if not per_step:
            return None
        _, summary = max(per_step.items(), key=lambda item: item[1][metric])
        worst = summary[record]
        if worst is None:
            return None
        return {"step_bohr": summary["step_bohr"], **worst}

    maximum_component_error = max(
        (item["maximum_component_error"] for item in per_step.values()), default=0.0
    )
    maximum_explicit_only_error = max(
        (item["maximum_explicit_only_error"] for item in per_step.values()),
        default=0.0,
    )
    return {
        "per_step": per_step,
        "maximum_component_error": maximum_component_error,
        "maximum_directional_error": max(
            (item["maximum_directional_error"] for item in per_step.values()),
            default=0.0,
        ),
        "maximum_relaxed_descriptor_error": max(
            (
                item["maximum_relaxed_descriptor_error"]
                for item in per_step.values()
            ),
            default=0.0,
        ),
        "maximum_explicit_only_error": maximum_explicit_only_error,
        "explicit_to_complete_error_ratio": (
            maximum_explicit_only_error
            / max(maximum_component_error, np.finfo(float).tiny)
        ),
        "worst_component": aggregate_worst(
            "maximum_component_error", "worst_component"
        ),
        "worst_descriptor": aggregate_worst(
            "maximum_relaxed_descriptor_error", "worst_descriptor"
        ),
        "worst_direction": aggregate_worst(
            "maximum_directional_error", "worst_direction"
        ),
    }


def action_scientific(output: Path, workload_id: str, family: str) -> dict[str, Any]:
    profile, threads = _profile()
    workload = workload_by_id(workload_id)
    result = base_result("scientific", profile, threads, workload, family)
    result["comparison_targets"] = [
        "pyscf-native",
        "fresh-fd",
        "direct-compact",
        "direct-detailed",
        "zvector-compact",
        "zvector-detailed",
    ]
    reference = fresh_reference(workload, family)
    result["reference"] = _reference_summary(reference)
    native_gradient = _pyscf_native_gradient(reference, family)
    zero_method = make_method(reference, None)
    zero_energy = float(zero_method.kernel())
    zero_gradient = np.asarray(zero_method.gradient(backend="direct"), dtype=np.float64)
    model = deterministic_model()
    method = make_method(reference, model)
    energy = float(method.kernel())
    direct_detailed, direct_detailed_gradient = _driver(method, "direct", True)
    direct_compact, direct_compact_gradient = _driver(method, "direct", False)
    zvector_detailed, zvector_detailed_gradient = _driver(method, "zvector", True)
    zvector_compact, zvector_compact_gradient = _driver(method, "zvector", False)
    repeated_energy = float(method.kernel())
    repeated_gradient = np.asarray(method.gradient(backend="zvector"), dtype=np.float64)
    loaded_model = type(model).load(
        CHECKPOINT_DIR / "frozen_corrnet.pth", strict=True
    ).double().eval()
    loaded_method = make_method(reference, loaded_model)
    loaded_energy = float(loaded_method.kernel())
    loaded_gradient = np.asarray(
        loaded_method.gradient(backend="zvector"), dtype=np.float64
    )
    result["central"] = {
        "native_energy": float(reference.e_tot),
        "native_gradient": native_gradient,
        "native_gradient_grid_response": family in {"rks", "uks"},
        "zero_correction_energy": zero_energy,
        "zero_correction_gradient": zero_gradient,
        "energy": energy,
        "correction_energy": energy - float(reference.e_tot),
        "direct_detailed": {
            "gradient": direct_detailed_gradient,
            "diagnostics": diagnostics_dict(direct_detailed),
            "partitions": gradient_partitions(direct_detailed),
        },
        "direct_compact": {
            "gradient": direct_compact_gradient,
            "diagnostics": diagnostics_dict(direct_compact),
            "retained_fields": sorted(vars(direct_compact)),
        },
        "zvector_detailed": {
            "gradient": zvector_detailed_gradient,
            "diagnostics": diagnostics_dict(zvector_detailed),
            "partitions": gradient_partitions(zvector_detailed),
        },
        "zvector_compact": {
            "gradient": zvector_compact_gradient,
            "diagnostics": diagnostics_dict(zvector_compact),
            "retained_fields": sorted(vars(zvector_compact)),
        },
        "errors": {
            "zero_energy_native": abs(zero_energy - float(reference.e_tot)),
            "zero_gradient_native": error_norms(zero_gradient, native_gradient),
            "direct_zvector_detailed": error_norms(zvector_detailed_gradient, direct_detailed_gradient),
            "direct_compact_detailed": error_norms(direct_compact_gradient, direct_detailed_gradient),
            "zvector_compact_detailed": error_norms(zvector_compact_gradient, zvector_detailed_gradient),
            "repeated_gradient": error_norms(repeated_gradient, zvector_compact_gradient),
            "repeated_energy": abs(repeated_energy - energy),
            "checkpoint_gradient": error_norms(loaded_gradient, zvector_compact_gradient),
            "checkpoint_energy": abs(loaded_energy - energy),
        },
        "total_force_sum": -np.sum(direct_detailed_gradient, axis=0),
    }
    finite_difference = _complete_finite_differences(
        workload, family, reference, model
    )
    result["finite_difference"] = finite_difference
    result["finite_difference_summary"] = _assess_finite_differences(
        finite_difference,
        direct_detailed_gradient,
        native_gradient + direct_detailed.correction_gradient_explicit,
        direct_detailed.dq_dR_relaxed,
    )
    acceptance = load_config()["acceptance"]
    finite_difference_step_checks = {}
    for step_label, summary in result["finite_difference_summary"]["per_step"].items():
        step_checks = {
            "component": summary["maximum_component_error"]
            <= acceptance["finite_difference_gradient_max_abs_hartree_per_bohr"],
            "direction": summary["maximum_directional_error"]
            <= acceptance["finite_difference_gradient_max_abs_hartree_per_bohr"],
            "descriptor": summary["maximum_relaxed_descriptor_error"]
            <= acceptance["finite_difference_descriptor_max_abs_per_bohr"],
        }
        summary["acceptance_checks"] = step_checks
        summary["passed"] = all(step_checks.values())
        finite_difference_step_checks[step_label] = step_checks
    result["finite_difference_step_checks"] = finite_difference_step_checks
    reasons = []
    zero_gradient_tolerance = (
        acceptance["zero_hf_gradient_max_abs_hartree_per_bohr"]
        if family in {"rhf", "uhf"}
        else acceptance["zero_dft_gradient_max_abs_hartree_per_bohr"]
    )
    checks = {
        "zero_energy": result["central"]["errors"]["zero_energy_native"] <= acceptance["zero_energy_max_abs_hartree"],
        "zero_gradient": result["central"]["errors"]["zero_gradient_native"]["max_abs"] <= zero_gradient_tolerance,
        "direct_zvector": result["central"]["errors"]["direct_zvector_detailed"]["max_abs"] <= acceptance["direct_zvector_max_abs_hartree_per_bohr"],
        "direct_compact": result["central"]["errors"]["direct_compact_detailed"]["max_abs"] <= acceptance["compact_detailed_max_abs_hartree_per_bohr"],
        "zvector_compact": result["central"]["errors"]["zvector_compact_detailed"]["max_abs"] <= acceptance["compact_detailed_max_abs_hartree_per_bohr"],
        "finite_difference_component": all(
            item["component"] for item in finite_difference_step_checks.values()
        ),
        "finite_difference_direction": all(
            item["direction"] for item in finite_difference_step_checks.values()
        ),
        "finite_difference_descriptor": all(
            item["descriptor"] for item in finite_difference_step_checks.values()
        ),
        "anti_vacuity_response": max_abs(direct_detailed.correction_gradient_response) >= acceptance["response_signal_min_hartree_per_bohr"],
        "anti_vacuity_descriptor": max_abs(direct_detailed.dq_dR_response) >= acceptance["descriptor_response_signal_min_per_bohr"],
        "anti_vacuity_explicit_error": result["finite_difference_summary"]["explicit_to_complete_error_ratio"] >= acceptance["explicit_error_ratio_min"],
        "repeat_identity": result["central"]["errors"]["repeated_gradient"]["max_abs"] == 0.0 and result["central"]["errors"]["repeated_energy"] == 0.0,
        "checkpoint_identity": result["central"]["errors"]["checkpoint_gradient"]["max_abs"] == 0.0 and result["central"]["errors"]["checkpoint_energy"] == 0.0,
    }
    if family == "rhf":
        checks["force_sum"] = max_abs(result["central"]["total_force_sum"]) <= acceptance["force_sum_max_abs_hartree_per_bohr"]
    for name, passed in checks.items():
        if not passed:
            reasons.append(f"acceptance check failed: {name}")
    result["acceptance_checks"] = checks
    result["categories"]["scientific"] = {"passed": all(checks.values()), "reasons": reasons}
    integrity_passed = checks["repeat_identity"] and checks["checkpoint_identity"]
    result["categories"]["integrity"] = {"passed": integrity_passed, "reasons": [] if integrity_passed else ["unchanged input or checkpoint reload was not bitwise reproducible"]}
    write_json(output, result)
    return result


def _dense_operator_and_vectors(method, family: str):
    with _forbid_dense_adjoint(method) as action_counts:
        driver, zvector_gradient = _driver(method, "zvector", True)
    adjoint = driver.adjoint_result
    adapter_type = method._adjoint_adapter_type
    available = {
        **load_config()["adjoint_controls"],
        **load_config()["validation_operator_controls"],
    }
    parameters = inspect.signature(adapter_type.__init__).parameters
    if "controls" not in parameters:
        available = {name: value for name, value in available.items() if name in parameters}
    adapter = adapter_type(method.reference, **available)
    core = getattr(adapter, "_core", adapter)
    coefficient, energy, occupation, occupied, virtual, _gap = core._state()
    dimension = response_dimensions(method.reference)["response_dimension"]
    if dimension > load_config()["dense_dimension_limit"]:
        raise RuntimeError(
            f"response dimension {dimension} exceeds dense replay limit"
        )
    matrix = np.empty((dimension, dimension), dtype=np.float64)
    identity = np.eye(dimension, dtype=np.float64)
    batch_size = min(32, dimension)
    for start in range(0, dimension, batch_size):
        stop = min(start + batch_size, dimension)
        if np.asarray(occupation).ndim == 1:
            nocc = int(np.count_nonzero(occupied))
            nvir = int(np.count_nonzero(virtual))
            roots = identity[start:stop].reshape(-1, nvir, nocc)
            images = core._apply_occupied_virtual_operator(
                roots, coefficient, energy, occupation, occupied, virtual
            ).reshape(stop - start, dimension)
        else:
            images = core._apply_occupied_virtual_operator(
                identity[start:stop], coefficient, energy, occupied, virtual
            ).reshape(stop - start, dimension)
        matrix[:, start:stop] = images.T
    if family in {"rhf", "rks"}:
        objective = np.asarray(adjoint.objective_orbital_gradient).reshape(-1)
        matrix_free = np.asarray(adjoint.zvector).reshape(-1)
    else:
        objective = np.concatenate(
            (
                np.asarray(adjoint.alpha_objective_orbital_gradient).reshape(-1),
                np.asarray(adjoint.beta_objective_orbital_gradient).reshape(-1),
            )
        )
        matrix_free = np.concatenate(
            (
                np.asarray(adjoint.alpha_zvector).reshape(-1),
                np.asarray(adjoint.beta_zvector).reshape(-1),
            )
        )
    return driver, zvector_gradient, matrix, objective, matrix_free, action_counts


def action_dense(output: Path, workload_id: str, family: str) -> dict[str, Any]:
    profile, threads = _profile()
    workload = workload_by_id(workload_id)
    result = base_result("dense-replay", profile, threads, workload, family)
    result["comparison_targets"] = [
        "dense-replay",
        "direct-detailed",
        "zvector-detailed",
    ]
    reference = fresh_reference(workload, family)
    method = make_method(reference, deterministic_model())
    method.kernel()
    construction_start = time.perf_counter()
    driver, zvector_gradient, matrix, objective, matrix_free, action_counts = _dense_operator_and_vectors(method, family)
    construction_time = time.perf_counter() - construction_start
    solve_start = time.perf_counter()
    dense_solution = np.linalg.solve(matrix.T, objective)
    solve_time = time.perf_counter() - solve_start
    eigen_start = time.perf_counter()
    eigenvalues = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
    eigen_time = time.perf_counter() - eigen_start
    direct_driver, direct_gradient = _driver(method, "direct", True)
    result["reference"] = _reference_summary(reference)
    result["dense"] = {
        "matrix_shape": matrix.shape,
        "matrix_bytes": matrix.nbytes,
        "matrix_hash": hash_array(matrix),
        "explicit_forward_action_batches": int(np.ceil(matrix.shape[0] / 32)),
        "construction_seconds": construction_time,
        "solve_seconds": solve_time,
        "eigensolve_seconds": eigen_time,
        "symmetry_residual": max_abs(matrix - matrix.T),
        "minimum_eigenvalue": float(eigenvalues[0]),
        "maximum_eigenvalue": float(eigenvalues[-1]),
        "condition_number": float(eigenvalues[-1] / eigenvalues[0]),
        "dense_residual": error_norms(matrix.T @ dense_solution, objective),
        "matrix_free_residual": error_norms(matrix.T @ matrix_free, objective),
        "solution_error": error_norms(matrix_free, dense_solution),
        "response_gradient_error": error_norms(
            driver.correction_gradient_response,
            direct_driver.correction_gradient_response,
        ),
        "final_gradient_error": error_norms(zvector_gradient, direct_gradient),
        "matrix_free_diagnostics": diagnostics_dict(driver),
        "matrix_free_action_counts": action_counts,
    }
    acceptance = load_config()["acceptance"]
    checks = {
        "symmetry": result["dense"]["symmetry_residual"] <= load_config()["validation_operator_controls"]["operator_symmetry_tolerance"],
        "positive": result["dense"]["minimum_eigenvalue"] > load_config()["validation_operator_controls"]["operator_stability_tolerance"],
        "solution_relative_l2": result["dense"]["solution_error"]["relative_l2"] <= acceptance["dense_relative_l2"],
        "solution_max_abs": result["dense"]["solution_error"]["max_abs"] <= acceptance["dense_max_abs"],
        "gradient": result["dense"]["final_gradient_error"]["max_abs"] <= acceptance["direct_zvector_max_abs_hartree_per_bohr"],
    }
    result["acceptance_checks"] = checks
    result["categories"]["scientific"] = {
        "passed": all(checks.values()),
        "reasons": [f"acceptance check failed: {name}" for name, passed in checks.items() if not passed],
    }
    write_json(output, result)
    return result


def _retained_array_bytes(driver) -> int:
    return sum(
        value.nbytes
        for value in vars(driver).values()
        if isinstance(value, np.ndarray)
    )


@contextmanager
def _forbid_dense_adjoint(method):
    """Fail a production Z-vector transaction on any dense adjoint operation."""
    import deepks.deephf.pyscf_uks as pyscf_uks
    import deepks.deephf.pyscf_rhf as pyscf_rhf
    import deepks.deephf.pyscf_rks as pyscf_rks
    import deepks.deephf.pyscf_uhf as pyscf_uhf

    dimension = response_dimensions(method.reference)["response_dimension"]

    def forbidden_matrix(*_args, **_kwargs):
        raise AssertionError("production Z-vector attempted explicit response-matrix construction")

    def forbidden_solve(*_args, **_kwargs):
        raise AssertionError("production Z-vector attempted numpy.linalg.solve")

    allocation_functions = {}
    for name in ("empty", "zeros", "ones"):
        original = getattr(np, name)

        def checked(shape, *args, _original=original, **kwargs):
            if tuple(shape) == (dimension, dimension) if hasattr(shape, "__iter__") else False:
                raise AssertionError("production Z-vector allocated a square response matrix")
            return _original(shape, *args, **kwargs)

        allocation_functions[name] = checked
    adapter_type = method._adjoint_adapter_type
    core_type = (
        pyscf_uks._UKSInternalAdjointAdapter
        if not hasattr(adapter_type, "_response_operator_matrix_and_diagnostics")
        else adapter_type
    )
    family = type(method.reference).__name__.lower()
    if family == "rhf":
        problem_type = pyscf_rhf._RHFScalarAdjointProblem
    elif family == "rks":
        problem_type = pyscf_rks._RKSLinearResponseProblem
    else:
        problem_type = pyscf_uhf._UHFScalarAdjointProblem
    action_counts = {"forward": 0, "transpose": 0, "preconditioner": 0}
    with ExitStack() as stack:
        stack.enter_context(mock.patch.object(core_type, "_response_operator_matrix_and_diagnostics", forbidden_matrix))
        stack.enter_context(mock.patch.object(np.linalg, "solve", forbidden_solve))
        stack.enter_context(mock.patch.object(np, "empty", allocation_functions["empty"]))
        stack.enter_context(mock.patch.object(np, "zeros", allocation_functions["zeros"]))
        stack.enter_context(mock.patch.object(np, "ones", allocation_functions["ones"]))
        for name, label in (
            ("apply", "forward"),
            ("apply_transpose", "transpose"),
            ("precondition", "preconditioner"),
        ):
            original = getattr(problem_type, name)

            def counted(self, *args, _original=original, _label=label, **kwargs):
                action_counts[_label] += 1
                return _original(self, *args, **kwargs)

            stack.enter_context(mock.patch.object(problem_type, name, counted))
        yield action_counts


@contextmanager
def _telemetry(method):
    counts: dict[str, int] = {}
    patches = []

    def wrap(owner, name: str, label: str):
        if not hasattr(owner, name):
            return
        original = getattr(owner, name)

        def counted(*args, **kwargs):
            counts[label] = counts.get(label, 0) + 1
            return original(*args, **kwargs)

        patches.append(mock.patch.object(owner, name, counted))

    wrap(method, "_correction_sensitivity", "model_sensitivity")
    wrap(method, "_current_science_state_fingerprint", "science_state_fingerprints")
    wrap(method, "_validate_reference_object", "reference_validations")
    wrap(method._descriptor, "correction_derivatives", "contracted_descriptor_differential")
    wrap(method._descriptor, "dq_dP", "complete_dq_dP")
    wrap(method._descriptor, "dq_dR_explicit", "complete_dq_dR_explicit")
    wrap(method._descriptor, "projected_density", "descriptor_projections")
    wrap(method._descriptor, "torch_descriptor", "torch_descriptor_evaluations")
    wrap(getattr(method, "_response_adapter_type", None), "solve", "direct_response_solves")
    wrap(
        getattr(method, "_response_adapter_type", None),
        "_solve_for_gradient",
        "compact_direct_response_solves",
    )
    wrap(getattr(method, "_adjoint_adapter_type", None), "solve", "adjoint_solves")
    for patcher in patches:
        patcher.start()
    try:
        with _profile_call_counters(
            counts,
            ((type(method.reference).nuc_grad_method, "native_gradient_factories"),),
        ):
            yield counts
    finally:
        for patcher in reversed(patches):
            patcher.stop()


def action_benchmark(
    output: Path,
    workload_id: str,
    family: str,
    backend: str,
    detailed: bool,
) -> dict[str, Any]:
    profile, threads = _profile()
    workload = workload_by_id(workload_id)
    result = base_result("benchmark", profile, threads, workload, family)
    result["backend"] = backend
    result["result_mode"] = "detailed" if detailed else "compact"
    repetitions = int(load_config()["resource_tiers"][workload["tier"]]["repetitions"])
    cold_start = time.perf_counter()
    reference = fresh_reference(workload, family)
    model = deterministic_model()
    method = make_method(reference, model)
    energy = float(method.kernel())
    current_compact_api = "retain_details" in inspect.signature(method.nuc_grad_method).parameters
    result["comparison_targets"] = [
        f"{backend}-{'detailed' if detailed else 'compact'}"
        if current_compact_api
        else "pre-matrix-free"
    ]
    dense_guard = _forbid_dense_adjoint(method) if backend == "zvector" and current_compact_api else nullcontext()
    with dense_guard:
        driver, cold_gradient = _driver(method, backend, detailed)
    cold_seconds = time.perf_counter() - cold_start
    native_start = time.perf_counter()
    native_gradient = _pyscf_native_gradient(reference, family)
    native_seconds = time.perf_counter() - native_start
    dense_guard = _forbid_dense_adjoint(method) if backend == "zvector" and current_compact_api else nullcontext()
    with dense_guard:
        _driver(method, backend, detailed)
    samples = []
    cpu_samples = []
    tracemalloc.start()
    telemetry_counts = None
    dense_action_counts = None
    latest_driver = driver
    latest_gradient = cold_gradient
    for _ in range(repetitions):
        gc.collect()
        wall_start = time.perf_counter()
        cpu_start = time.process_time()
        dense_guard = _forbid_dense_adjoint(method) if backend == "zvector" and current_compact_api else nullcontext()
        with dense_guard as action_counts, _telemetry(method) as counts:
            latest_driver, latest_gradient = _driver(method, backend, detailed)
        cpu_samples.append(time.process_time() - cpu_start)
        samples.append(time.perf_counter() - wall_start)
        telemetry_counts = counts
        dense_action_counts = action_counts
    _, python_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    result["reference"] = _reference_summary(reference)
    result["measurement"] = {
        "cold_end_to_end_seconds": cold_seconds,
        "native_gradient_seconds": native_seconds,
        "warm_gradient": statistics(samples),
        "cpu_seconds": statistics(cpu_samples),
        "python_peak_allocation_bytes": python_peak,
        "retained_array_bytes": _retained_array_bytes(latest_driver),
        "retained_fields": sorted(vars(latest_driver)),
        "telemetry": telemetry_counts,
        "matrix_free_action_counts": dense_action_counts,
        "diagnostics": diagnostics_dict(latest_driver),
        "gradient": latest_gradient,
        "force": -latest_gradient,
        "energy": energy,
        "descriptor": np.asarray(method.descriptor()),
        "native_gradient": native_gradient,
    }
    acceptance = load_config()["acceptance"]
    mad_pass = result["measurement"]["warm_gradient"]["mad_fraction"] <= acceptance["performance_mad_fraction"]
    result["categories"]["scientific"] = {"passed": bool(np.isfinite(latest_gradient).all()), "reasons": []}
    result["categories"]["performance"] = {
        "passed": mad_pass,
        "reasons": [] if mad_pass else ["MAD divided by median exceeds 0.05"],
    }
    write_json(output, result)
    return result


def action_native_benchmark(output: Path, workload_id: str, family: str) -> dict[str, Any]:
    """Measure the shared native energy-gradient cost in its own fresh process."""
    profile, threads = _profile()
    workload = workload_by_id(workload_id)
    result = base_result("native-benchmark", profile, threads, workload, family)
    result["comparison_targets"] = ["pyscf-native"]
    repetitions = int(load_config()["resource_tiers"][workload["tier"]]["repetitions"])
    cold_start = time.perf_counter()
    reference = fresh_reference(workload, family)
    cold_gradient = _pyscf_native_gradient(reference, family)
    cold_seconds = time.perf_counter() - cold_start
    _pyscf_native_gradient(reference, family)
    samples = []
    gradients = []
    for _ in range(repetitions):
        start = time.perf_counter()
        gradients.append(_pyscf_native_gradient(reference, family))
        samples.append(time.perf_counter() - start)
    result["reference"] = _reference_summary(reference)
    result["measurement"] = {
        "cold_end_to_end_seconds": cold_seconds,
        "warm_gradient": statistics(samples),
        "gradient": gradients[-1],
        "repeat_max_abs": max(
            (max_abs(gradient - gradients[0]) for gradient in gradients[1:]),
            default=0.0,
        ),
        "cold_gradient_error": error_norms(cold_gradient, gradients[-1]),
    }
    passed = result["measurement"]["repeat_max_abs"] == 0.0
    result["categories"]["scientific"] = {
        "passed": passed,
        "reasons": [] if passed else ["native repeated gradients changed"],
    }
    mad_passed = (
        result["measurement"]["warm_gradient"]["mad_fraction"]
        <= load_config()["acceptance"]["performance_mad_fraction"]
    )
    result["categories"]["performance"] = {
        "passed": mad_passed,
        "reasons": [] if mad_passed else ["MAD divided by median exceeds 0.05"],
    }
    write_json(output, result)
    return result


def action_selection(output: Path, workload_id: str, family: str) -> dict[str, Any]:
    profile, threads = _profile()
    workload = workload_by_id(workload_id)
    result = base_result("atom-selection", profile, threads, workload, family)
    result["comparison_targets"] = ["direct-compact", "zvector-compact"]
    reference = fresh_reference(workload, family)
    method = make_method(reference, deterministic_model())
    atom_count = reference.mol.natm
    subsets = {
        "one_atom": (0,),
        "one_monomer": tuple(range(min(3, atom_count))),
        "half_atoms": tuple(range((atom_count + 1) // 2)),
        "all_permuted": tuple(reversed(range(atom_count))),
    }
    records = {}
    checks = []
    for backend in ("direct", "zvector"):
        full_driver, full_gradient = _driver(method, backend, False)
        backend_records = {}
        for name, subset in subsets.items():
            start = time.perf_counter()
            driver, selected = _driver(method, backend, False, atmlst=subset)
            elapsed = time.perf_counter() - start
            expected = full_gradient[list(subset)]
            error = error_norms(selected, expected)
            checks.append(error["max_abs"])
            backend_records[name] = {
                "atom_indices": subset,
                "gradient": selected,
                "error": error,
                "elapsed_seconds": elapsed,
                "diagnostics": diagnostics_dict(driver),
            }
        records[backend] = {
            "full_gradient": full_gradient,
            "full_diagnostics": diagnostics_dict(full_driver),
            "subsets": backend_records,
        }
    if family == "rhf":
        block_records = {}
        for block_size in (1, 2, 4, 8, atom_count):
            start = time.perf_counter()
            driver, gradient = _driver(
                method,
                "direct",
                False,
                coordinate_block_size=block_size,
            )
            block_records[str(block_size)] = {
                "gradient": gradient,
                "error": error_norms(gradient, records["direct"]["full_gradient"]),
                "elapsed_seconds": time.perf_counter() - start,
                "diagnostics": diagnostics_dict(driver),
                "block_summary": getattr(driver, "blocked_response_summary", None),
            }
        result["coordinate_blocking"] = block_records
    result["selection"] = records
    passed = max(checks, default=0.0) <= load_config()["acceptance"]["selected_gradient_max_abs_hartree_per_bohr"]
    result["categories"]["scientific"] = {
        "passed": passed,
        "reasons": [] if passed else ["selected gradient rows disagree with full rows"],
    }
    write_json(output, result)
    return result


def action_subset(
    output: Path,
    workload_id: str,
    family: str,
    backend: str,
    label: str,
) -> dict[str, Any]:
    """Measure one atom subset in a fresh process for attributable peak RSS."""
    profile, threads = _profile()
    workload = workload_by_id(workload_id)
    result = base_result("atom-subset", profile, threads, workload, family)
    result["comparison_targets"] = [f"{backend}-compact"]
    reference = fresh_reference(workload, family)
    method = make_method(reference, deterministic_model())
    atom_count = reference.mol.natm
    subsets = {
        "full": tuple(range(atom_count)),
        "one_atom": (0,),
        "one_monomer": tuple(range(min(3, atom_count))),
        "half_atoms": tuple(range((atom_count + 1) // 2)),
        "all_permuted": tuple(reversed(range(atom_count))),
    }
    atom_indices = subsets[label]
    _driver(method, backend, False, atmlst=atom_indices)
    samples = []
    repetitions = load_config()["resource_tiers"][workload["tier"]]["repetitions"]
    for _ in range(repetitions):
        start = time.perf_counter()
        driver, gradient = _driver(method, backend, False, atmlst=atom_indices)
        samples.append(time.perf_counter() - start)
    result["backend"] = backend
    result["subset"] = label
    result["selected_atoms"] = atom_indices
    result["gradient"] = gradient
    result["timing"] = statistics(samples)
    result["diagnostics"] = diagnostics_dict(driver)
    result["retained_array_bytes"] = _retained_array_bytes(driver)
    result["categories"]["scientific"] = {
        "passed": bool(np.isfinite(gradient).all()),
        "reasons": [],
    }
    mad_passed = result["timing"]["mad_fraction"] <= load_config()["acceptance"]["performance_mad_fraction"]
    result["categories"]["performance"] = {
        "passed": mad_passed,
        "reasons": [] if mad_passed else ["MAD divided by median exceeds 0.05"],
    }
    write_json(output, result)
    return result


def action_block(
    output: Path,
    workload_id: str,
    family: str,
    label: str,
) -> dict[str, Any]:
    """Measure one RHF coordinate block size in an isolated child."""
    profile, threads = _profile()
    workload = workload_by_id(workload_id)
    result = base_result("coordinate-block", profile, threads, workload, family)
    result["comparison_targets"] = ["direct-compact"]
    reference = fresh_reference(workload, family)
    method = make_method(reference, deterministic_model())
    block_size = reference.mol.natm if label == "full" else int(label)
    _driver(
        method,
        "direct",
        False,
        coordinate_block_size=block_size,
    )
    samples = []
    repetitions = load_config()["resource_tiers"][workload["tier"]]["repetitions"]
    for _ in range(repetitions):
        start = time.perf_counter()
        driver, gradient = _driver(
            method,
            "direct",
            False,
            coordinate_block_size=block_size,
        )
        samples.append(time.perf_counter() - start)
    result["coordinate_block_size"] = block_size
    result["gradient"] = gradient
    result["timing"] = statistics(samples)
    result["diagnostics"] = diagnostics_dict(driver)
    result["blocked_response_summary"] = getattr(driver, "blocked_response_summary", None)
    result["categories"]["scientific"] = {
        "passed": bool(np.isfinite(gradient).all()),
        "reasons": [],
    }
    mad_passed = result["timing"]["mad_fraction"] <= load_config()["acceptance"]["performance_mad_fraction"]
    result["categories"]["performance"] = {
        "passed": mad_passed,
        "reasons": [] if mad_passed else ["MAD divided by median exceeds 0.05"],
    }
    write_json(output, result)
    return result


def _rotation_matrix() -> np.ndarray:
    axis = np.asarray([0.37, -0.53, 0.76], dtype=np.float64)
    axis /= np.linalg.norm(axis)
    angle = 0.413
    cross = np.asarray(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]],
        dtype=np.float64,
    )
    return np.eye(3) * np.cos(angle) + (1.0 - np.cos(angle)) * np.outer(axis, axis) + np.sin(angle) * cross


def _fresh_energy_gradient(workload, family, atoms, coordinates, backend="zvector"):
    reference = fresh_reference(workload, family, coordinates, atoms)
    method = make_method(reference, deterministic_model())
    return float(method.kernel()), np.asarray(method.gradient(backend=backend))


def _fresh_energy_gradient_angstrom(workload, family, atoms, coordinates_bohr):
    from deepks.deephf import build_reference
    from common import ANGSTROM_PER_BOHR

    molecule = gto.M(
        atom=list(zip(atoms, coordinates_bohr * ANGSTROM_PER_BOHR)),
        basis=workload["basis"],
        unit="Angstrom",
        charge=workload["charge"],
        spin=workload["spin"],
        symmetry=False,
        cart=False,
        verbose=0,
    )
    reference = build_reference(
        molecule,
        family,
        scf_args=effective_scf_controls(workload, family),
        verbose=0,
    )
    method = make_method(reference, deterministic_model())
    return float(method.kernel()), np.asarray(method.gradient(backend="zvector"))


def action_invariance(output: Path, workload_id: str, family: str) -> dict[str, Any]:
    profile, threads = _profile()
    workload = workload_by_id(workload_id)
    result = base_result("invariance", profile, threads, workload, family)
    result["comparison_targets"] = ["zvector-compact"]
    atoms, coordinates = workload_geometry(workload)
    central_energy, central_gradient = _fresh_energy_gradient(workload, family, atoms, coordinates)
    translation = np.asarray([0.41, -0.27, 0.19])
    translated_energy, translated_gradient = _fresh_energy_gradient(workload, family, atoms, coordinates + translation)
    rotation = _rotation_matrix()
    center = coordinates.mean(axis=0)
    rotated_coordinates = (coordinates - center) @ rotation.T + center
    rotated_energy, rotated_gradient = _fresh_energy_gradient(workload, family, atoms, rotated_coordinates)
    permutation = tuple(reversed(range(len(atoms))))
    permuted_atoms = tuple(atoms[index] for index in permutation)
    permuted_coordinates = coordinates[list(permutation)]
    permuted_energy, permuted_gradient = _fresh_energy_gradient(workload, family, permuted_atoms, permuted_coordinates)
    inverse = np.argsort(permutation)
    angstrom_energy, angstrom_gradient = _fresh_energy_gradient_angstrom(
        workload, family, atoms, coordinates
    )
    result["invariance"] = {
        "translation_energy_error": abs(translated_energy - central_energy),
        "translation_gradient_error": error_norms(translated_gradient, central_gradient),
        "rotation_energy_error": abs(rotated_energy - central_energy),
        "rotation_gradient_error": error_norms(rotated_gradient, central_gradient @ rotation.T),
        "permutation_energy_error": abs(permuted_energy - central_energy),
        "permutation_gradient_error": error_norms(permuted_gradient[inverse], central_gradient),
        "angstrom_bohr_energy_error": abs(angstrom_energy - central_energy),
        "angstrom_bohr_gradient_error": error_norms(angstrom_gradient, central_gradient),
        "force_sum": -np.sum(central_gradient, axis=0),
    }
    tolerance = 1.0e-8 if family == "rhf" else 1.0e-6
    values = [
        result["invariance"]["translation_gradient_error"]["max_abs"],
        result["invariance"]["rotation_gradient_error"]["max_abs"],
        result["invariance"]["permutation_gradient_error"]["max_abs"],
        result["invariance"]["angstrom_bohr_gradient_error"]["max_abs"],
    ]
    passed = max(values) <= tolerance
    result["categories"]["scientific"] = {
        "passed": passed,
        "reasons": [] if passed else ["rigid-motion or atom-permutation invariance exceeded tolerance"],
    }
    write_json(output, result)
    return result


def action_conditioning(output: Path, _workload_id: str | None, _family: str | None) -> dict[str, Any]:
    profile, threads = _profile()
    result = base_result("conditioning-sweep", profile, threads)
    result["comparison_targets"] = ["dense-replay", "direct-detailed", "zvector-detailed"]
    sweep = load_config()["conditioning_sweep"]
    records = []
    for distance in sweep["bond_lengths_bohr"]:
        coordinates = np.asarray(
            [[0.0, 0.0, 0.0], [distance, 0.07, -0.03], [2.0 * distance, -0.05, 0.06], [3.0 * distance, 0.03, -0.08]],
            dtype=np.float64,
        )
        molecule = gto.M(
            atom=list(zip(("H",) * 4, coordinates)),
            basis=sweep["basis"],
            unit="Bohr",
            charge=0,
            spin=0,
            symmetry=False,
            cart=False,
            verbose=0,
        )
        reference = scf.RHF(molecule).set(**load_config()["scf_controls"])
        reference.kernel()
        record: dict[str, Any] = {"bond_length_bohr": distance, "converged": bool(reference.converged)}
        if not reference.converged:
            record["accepted"] = False
            record["reason"] = "SCF did not converge"
            records.append(record)
            continue
        method = make_method(reference, deterministic_model())
        try:
            direct_start = time.perf_counter()
            direct_driver, direct_gradient = _driver(method, "direct", True)
            direct_seconds = time.perf_counter() - direct_start
            z_start = time.perf_counter()
            z_driver, z_gradient = _driver(method, "zvector", True)
            z_seconds = time.perf_counter() - z_start
            _, _, matrix, objective, matrix_free, action_counts = _dense_operator_and_vectors(method, "rhf")
            eigenvalues = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
            record.update(
                {
                    "accepted": True,
                    "minimum_orbital_gap": minimum_orbital_gaps(reference)[0],
                    "minimum_eigenvalue": float(eigenvalues[0]),
                    "maximum_eigenvalue": float(eigenvalues[-1]),
                    "condition_number": float(eigenvalues[-1] / eigenvalues[0]),
                    "gmres_iterations": z_driver.response_diagnostics.iteration_count,
                    "gmres_residual": z_driver.response_diagnostics.maximum_residual,
                    "physical_residual": error_norms(matrix.T @ matrix_free, objective),
                    "matrix_free_action_counts": action_counts,
                    "gradient_error": error_norms(z_gradient, direct_gradient),
                    "direct_seconds": direct_seconds,
                    "zvector_seconds": z_seconds,
                    "direct_residual": direct_driver.response_diagnostics.maximum_residual,
                }
            )
        except Exception as error:
            record["accepted"] = False
            record["reason"] = f"{type(error).__name__}: {error}"
        records.append(record)
    result["records"] = records
    accepted = [record for record in records if record["accepted"]]
    passed = bool(accepted) and all(
        record["gradient_error"]["max_abs"] <= load_config()["acceptance"]["direct_zvector_max_abs_hartree_per_bohr"]
        for record in accepted
    )
    result["categories"]["scientific"] = {
        "passed": passed,
        "reasons": [] if passed else ["no conditioning state passed strict scientific acceptance"],
    }
    write_json(output, result)
    return result


@contextmanager
def _dft_counters():
    """Count expensive grid, NumInt, LibXC, and native-gradient entry points."""
    from pyscf.dft import gen_grid, libxc, numint

    counts: dict[str, int] = {}
    targets = (
        (gen_grid.Grids.build, "grid_builds"),
        (numint.NumInt.eval_ao, "numint_eval_ao"),
        (numint.NumInt.eval_xc_eff, "numint_eval_xc_eff"),
        (libxc.eval_xc1, "libxc_eval_xc1"),
        (dft.rks.RKS.nuc_grad_method, "native_rks_gradient_factories"),
        (dft.uks.UKS.nuc_grad_method, "native_uks_gradient_factories"),
    )
    with _profile_call_counters(counts, targets):
        yield counts


def action_dft_sequence(output: Path, workload_id: str, family: str) -> dict[str, Any]:
    profile, threads = _profile()
    workload = workload_by_id(workload_id)
    result = base_result("dft-sequence", profile, threads, workload, family)
    result["comparison_targets"] = ["pyscf-native", "direct-detailed", "zvector-detailed", "zvector-compact"]
    with _dft_counters() as counters:
        reference_start = time.perf_counter()
        reference = fresh_reference(workload, family)
        reference_seconds = time.perf_counter() - reference_start
        model = deterministic_model()
        cold_start = time.perf_counter()
        method = make_method(reference, model)
        cold_validation_seconds = time.perf_counter() - cold_start
        warm_start = time.perf_counter()
        method.validate_force_compatibility()
        warm_validation_seconds = time.perf_counter() - warm_start
        native_start = time.perf_counter()
        native_gradient = _pyscf_native_gradient(reference, family)
        native_seconds = time.perf_counter() - native_start
        direct_start = time.perf_counter()
        direct_driver, direct_gradient = _driver(method, "direct", True)
        direct_seconds = time.perf_counter() - direct_start
        zvector_start = time.perf_counter()
        zvector_driver, zvector_gradient = _driver(method, "zvector", True)
        zvector_seconds = time.perf_counter() - zvector_start
        unchanged_times = []
        unchanged_gradients = []
        for _ in range(10):
            start = time.perf_counter()
            unchanged_gradients.append(np.asarray(method.gradient(backend="zvector")))
            unchanged_times.append(time.perf_counter() - start)
        fresh_times = []
        fresh_gradients = []
        for _ in range(10):
            start = time.perf_counter()
            new_reference = fresh_reference(workload, family)
            fresh_method = make_method(new_reference, model)
            fresh_gradients.append(np.asarray(fresh_method.gradient(backend="zvector")))
            fresh_times.append(time.perf_counter() - start)
    unchanged_errors = [
        error_norms(gradient, unchanged_gradients[0])
        for gradient in unchanged_gradients[1:]
    ]
    fresh_errors = [
        error_norms(gradient, unchanged_gradients[0])
        for gradient in fresh_gradients
    ]
    result["reference"] = _reference_summary(reference)
    result["dft_measurement"] = {
        "reference_construction_seconds": reference_seconds,
        "cold_validation_seconds": cold_validation_seconds,
        "warm_validation_seconds": warm_validation_seconds,
        "native_gradient_seconds": native_seconds,
        "direct_gradient_seconds": direct_seconds,
        "zvector_gradient_seconds": zvector_seconds,
        "native_gradient": native_gradient,
        "direct_gradient": direct_gradient,
        "zvector_gradient": zvector_gradient,
        "direct_zvector_error": error_norms(zvector_gradient, direct_gradient),
        "direct_diagnostics": diagnostics_dict(direct_driver),
        "zvector_diagnostics": diagnostics_dict(zvector_driver),
        "unchanged_sequence": statistics(unchanged_times),
        "fresh_sequence": statistics(fresh_times),
        "unchanged_max_error": max((item["max_abs"] for item in unchanged_errors), default=0.0),
        "fresh_max_error": max((item["max_abs"] for item in fresh_errors), default=0.0),
        "counters": counters,
    }
    passed = (
        result["dft_measurement"]["direct_zvector_error"]["max_abs"]
        <= load_config()["acceptance"]["direct_zvector_max_abs_hartree_per_bohr"]
        and result["dft_measurement"]["unchanged_max_error"] == 0.0
        and result["dft_measurement"]["fresh_max_error"]
        <= load_config()["acceptance"]["compact_detailed_max_abs_hartree_per_bohr"]
    )
    result["categories"]["scientific"] = {
        "passed": passed,
        "reasons": [] if passed else ["DFT sequence gradients were not reproducible"],
    }
    result["categories"]["integrity"] = {
        "passed": result["dft_measurement"]["unchanged_max_error"] == 0.0,
        "reasons": [],
    }
    performance_passed = (
        result["dft_measurement"]["unchanged_sequence"]["mad_fraction"]
        <= load_config()["acceptance"]["performance_mad_fraction"]
        and result["dft_measurement"]["fresh_sequence"]["mad_fraction"]
        <= load_config()["acceptance"]["performance_mad_fraction"]
    )
    result["categories"]["performance"] = {
        "passed": performance_passed,
        "reasons": [] if performance_passed else ["DFT sequence MAD divided by median exceeds 0.05"],
    }
    write_json(output, result)
    return result


def _scanner_trajectory(workload: dict[str, Any]) -> list[np.ndarray]:
    _, central = workload_geometry(workload)
    generator = np.random.default_rng(load_config()["seed"] + 91)
    deformation = generator.normal(size=central.shape)
    deformation -= deformation.mean(axis=0, keepdims=True)
    deformation /= np.max(np.abs(deformation))
    forward = []
    for index in range(50):
        fraction = index / 49.0
        angle = 0.012 * np.sin(np.pi * fraction)
        cross = np.asarray(
            [[0.0, -1.0, 0.0], [1.0, 0.0, -0.2], [0.0, 0.2, 0.0]],
            dtype=np.float64,
        )
        rotation = np.eye(3) + angle * cross
        translation = np.asarray([0.015 * fraction, -0.009 * fraction, 0.006 * fraction])
        coordinates = central @ rotation.T + 0.035 * fraction * deformation + translation
        forward.append(coordinates)
    forward[10] = forward[9].copy()
    reverse = [value.copy() for value in reversed(forward[:48])]
    trajectory = forward + reverse
    trajectory.append(forward[-1].copy())
    trajectory.append(forward[0].copy())
    if len(trajectory) != 100:
        raise AssertionError("scanner trajectory must contain exactly 100 frames")
    return trajectory


def action_scanner(output: Path, workload_id: str, family: str) -> dict[str, Any]:
    profile, threads = _profile()
    workload = workload_by_id(workload_id)
    result = base_result("scanner", profile, threads, workload, family)
    result["comparison_targets"] = ["fresh-fd", "zvector-compact"]
    reference = fresh_reference(workload, family)
    model = deterministic_model()
    method = make_method(reference, model)
    scanner = method.nuc_grad_method(backend="zvector", retain_details=False).as_scanner()
    frames = []
    trajectory = _scanner_trajectory(workload)
    first_seen: dict[str, tuple[float, np.ndarray]] = {}
    repeated_errors = []
    for frame_index, coordinates in enumerate(trajectory):
        start = time.perf_counter()
        energy, gradient = scanner(coordinates)
        elapsed = time.perf_counter() - start
        geometry_key = hash_array(coordinates)
        if geometry_key in first_seen:
            previous_energy, previous_gradient = first_seen[geometry_key]
            repeated_errors.append(
                {
                    "energy": abs(energy - previous_energy),
                    "gradient": error_norms(gradient, previous_gradient),
                }
            )
        else:
            first_seen[geometry_key] = (energy, gradient.copy())
        diagnostics = scanner.gradient_driver.response_diagnostics
        frames.append(
            {
                "index": frame_index,
                "geometry_hash": geometry_key,
                "energy": energy,
                "gradient": gradient,
                "elapsed_seconds": elapsed,
                "scf_cycles": getattr(scanner.reference, "cycles", None),
                "gmres_iterations": diagnostics.iteration_count,
                "response_residual": diagnostics.maximum_residual,
                "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            }
        )
    fresh_checks = []
    for frame_index in load_config()["scanner"]["check_frames"]:
        start = time.perf_counter()
        energy, gradient = _fresh_energy_gradient(
            workload, family, workload_geometry(workload)[0], trajectory[frame_index]
        )
        fresh_checks.append(
            {
                "index": frame_index,
                "elapsed_seconds": time.perf_counter() - start,
                "energy_error": abs(energy - frames[frame_index]["energy"]),
                "gradient_error": error_norms(gradient, frames[frame_index]["gradient"]),
            }
        )
    times = [frame["elapsed_seconds"] for frame in frames]
    result["scanner"] = {
        "frames": frames,
        "first_frame_seconds": times[0],
        "subsequent_frame_statistics": statistics(times[1:]),
        "repeated_frame_errors": repeated_errors,
        "fresh_checks": fresh_checks,
        "fresh_reconstruction_statistics": statistics(
            [item["elapsed_seconds"] for item in fresh_checks]
        ),
        "rss_growth_kib": frames[-1]["process_peak_rss_kib"] - frames[0]["process_peak_rss_kib"],
    }
    worst_fresh = max(
        (item["gradient_error"]["max_abs"] for item in fresh_checks), default=0.0
    )
    worst_repeat = max(
        (item["gradient"]["max_abs"] for item in repeated_errors), default=0.0
    )
    passed = worst_fresh <= 1.0e-8 and worst_repeat == 0.0
    result["categories"]["scientific"] = {
        "passed": passed,
        "reasons": [] if passed else ["scanner fresh or revisited geometry mismatch"],
    }
    write_json(output, result)
    return result


def _water_cluster_molecule(water_count: int):
    atoms = []
    coordinates = []
    base = np.asarray(
        [[0.0, 0.0, 0.0], [1.795, 0.02, 0.07], [-0.46, 1.72, -0.09]],
        dtype=np.float64,
    )
    for index in range(water_count):
        grid = np.asarray(
            [index % 4, (index // 4) % 4, index // 16], dtype=np.float64
        )
        shift = 5.1 * grid + np.asarray(
            [0.013 * index, -0.009 * index, 0.007 * index], dtype=np.float64
        )
        local = base.copy()
        local[1, 1] += 0.002 * index
        local[2, 2] -= 0.0015 * index
        atoms.extend(("O", "H", "H"))
        coordinates.extend(local + shift)
    return gto.M(
        atom=list(zip(atoms, np.asarray(coordinates))),
        basis="sto-3g",
        unit="Bohr",
        charge=0,
        spin=0,
        symmetry=False,
        cart=False,
        verbose=0,
    )


def _expand_force_dataset(base_directory: Path, target_directory: Path, frame_count: int):
    from deepks.data.force_schema import (
        CANONICAL_FORCE_FIELDS,
        _write_force_dataset,
        load_force_dataset,
    )

    contract, arrays = load_force_dataset(base_directory)
    expanded = {
        name: np.repeat(value, frame_count, axis=0)
        for name, value in arrays.items()
    }
    manifest = contract.manifest
    frame = {
        key: value
        for key, value in manifest["frames"][0].items()
        if key not in {"field_sha256", "sample_id"}
    }
    provenance = {
        name: manifest[name]
        for name in (
            "atom_mapping",
            "descriptor",
            "reference",
            "response",
            "target",
            "generation",
        )
    }
    provenance["frames"] = [frame for _ in range(frame_count)]
    start = time.perf_counter()
    expanded_contract = _write_force_dataset(
        target_directory,
        arrays=expanded,
        provenance=provenance,
    )
    serialization_seconds = time.perf_counter() - start
    return expanded_contract, expanded, serialization_seconds


def _force_epoch(reader, model, *, training: bool) -> dict[str, Any]:
    from deepks.model.reader import _force_batch_error

    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-4) if training else None
    batch_count = int(np.ceil(reader.nframes / reader.batch_size))
    hash_seconds = 0.0
    bytes_hashed = 0
    host_device_conversion_seconds = 0.0
    model_forward_seconds = 0.0
    autograd_sensitivity_seconds = 0.0
    jacobian_contraction_seconds = 0.0
    optimizer_seconds = 0.0
    metric_seconds = 0.0
    losses = []
    for _ in range(batch_count):
        batch = reader.sample_train()
        start = time.perf_counter()
        integrity_error = _force_batch_error(batch, (reader.force_contract,))
        hash_seconds += time.perf_counter() - start
        bytes_hashed += sum(
            tensor.nelement() * tensor.element_size() for tensor in batch.values()
        )
        if integrity_error is not None:
            raise RuntimeError(integrity_error)
        start = time.perf_counter()
        descriptor = batch["descriptor"].to("cpu")
        jacobian = batch["dq_dR_relaxed"].to("cpu")
        energy_target = batch["energy"].to("cpu")
        force_target = batch["force"].to("cpu")
        host_device_conversion_seconds += time.perf_counter() - start
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        values = descriptor.detach().requires_grad_(True)
        start = time.perf_counter()
        energy = model(values)
        model_forward_seconds += time.perf_counter() - start
        start = time.perf_counter()
        (sensitivity,) = torch.autograd.grad(
            energy,
            values,
            grad_outputs=torch.ones_like(energy),
            retain_graph=training,
            create_graph=training,
            only_inputs=True,
        )
        autograd_sensitivity_seconds += time.perf_counter() - start
        start = time.perf_counter()
        force = -torch.einsum("fbxik,fik->fbx", jacobian, sensitivity)
        jacobian_contraction_seconds += time.perf_counter() - start
        loss = torch.mean((energy - energy_target) ** 2) + torch.mean(
            (force - force_target) ** 2
        )
        if optimizer is not None:
            start = time.perf_counter()
            loss.backward()
            optimizer.step()
            optimizer_seconds += time.perf_counter() - start
        start = time.perf_counter()
        losses.append(float(loss.detach()))
        metric_seconds += time.perf_counter() - start
    return {
        "batch_count": batch_count,
        "integrity_hash_seconds": hash_seconds,
        "bytes_hashed": bytes_hashed,
        "host_device_conversion_seconds": host_device_conversion_seconds,
        "model_forward_seconds": model_forward_seconds,
        "autograd_sensitivity_seconds": autograd_sensitivity_seconds,
        "jacobian_contraction_seconds": jacobian_contraction_seconds,
        "optimizer_seconds": optimizer_seconds,
        "metric_seconds": metric_seconds,
        "mean_loss": float(np.mean(losses)),
    }


def _energy_epoch(reader, model) -> dict[str, Any]:
    batch_count = int(np.ceil(reader.nframes / reader.batch_size))
    seconds = 0.0
    errors = []
    for _ in range(batch_count):
        batch = reader.sample_train()
        start = time.perf_counter()
        prediction = model(batch["descriptor"])
        seconds += time.perf_counter() - start
        errors.append(float(torch.mean((prediction - batch["energy"]) ** 2)))
    return {
        "batch_count": batch_count,
        "model_forward_seconds": seconds,
        "mean_energy_mse": float(np.mean(errors)),
    }


def action_force_data(output: Path, _workload_id: str | None, _family: str | None, label: str) -> dict[str, Any]:
    from deepks.data.force_schema import force_checkpoint_metadata
    from deepks.deephf import write_rhf_force_dataset
    from deepks.model.evaluate import predict_correction
    from deepks.model.model import CorrNet
    from deepks.model.reader import Reader

    profile, threads = _profile()
    result = base_result("force-data", profile, threads)
    result["comparison_targets"] = ["direct-detailed"]
    payload = next(
        item for item in load_config()["force_data"]["payloads"] if item["id"] == label
    )
    directory = output.parent / "dataset"
    base_directory = output.parent / "base-frame"
    for generated_directory in (directory, base_directory):
        if generated_directory.exists():
            shutil.rmtree(generated_directory)
    molecule = _water_cluster_molecule(int(payload["water_count"]))
    reference_start = time.perf_counter()
    reference = scf.RHF(molecule).set(**load_config()["scf_controls"])
    reference.kernel()
    if not reference.converged:
        raise RuntimeError("force-data RHF reference did not converge")
    reference_seconds = time.perf_counter() - reference_start
    model = deterministic_model()
    teacher = make_method(reference, model)
    target_energy = float(teacher.kernel())
    target_force = np.asarray(teacher.forces(backend="direct"))
    generation_start = time.perf_counter()
    write_rhf_force_dataset(
        base_directory,
        reference,
        projector_basis=load_config()["projector_basis"],
        e_target=target_energy,
        f_target=target_force,
        target={
            "method": "deterministic DeePHF CorrNet teacher",
            "basis": "sto-3g",
            "software": "deepks-kit scientific-performance validation",
            "version": "1",
            "frozen_core": False,
            "relativistic": "none",
            "state": "closed-shell singlet ground state",
            "energy_force_consistent": True,
            "settings": {"model": "frozen deterministic tanh CorrNet"},
        },
        response_options=load_config()["response_controls"],
    )
    generation_seconds = time.perf_counter() - generation_start
    contract, expanded_arrays, serialization_seconds = _expand_force_dataset(
        base_directory, directory, int(payload["frames"])
    )
    jacobian_bytes = expanded_arrays["dq_dR_relaxed"].nbytes
    del expanded_arrays
    gc.collect()
    reader_records = {}
    checkpoint_error = None
    for batch_size in (*load_config()["force_data"]["batch_sizes"], 64):
        load_start = time.perf_counter()
        reader = Reader(
            directory,
            batch_size=min(int(batch_size), int(payload["frames"])),
            force_mode="deephf_relaxed",
        )
        load_seconds = time.perf_counter() - load_start
        evaluation_model = deterministic_model()
        energy_evaluation = _energy_epoch(reader, evaluation_model)
        evaluation = _force_epoch(reader, evaluation_model, training=False)
        training_model = deterministic_model()
        training = _force_epoch(reader, training_model, training=True)
        checkpoint = output.parent / f"checkpoint-b{batch_size}.pth"
        training_model.save(
            checkpoint,
            force_training=force_checkpoint_metadata(reader.force_contract),
        )
        loaded_model = CorrNet.load(
            checkpoint,
            strict=True,
            require_force_metadata=True,
            expected_force_contract=reader.force_contract,
        ).double().eval()
        sample = reader.sample_train()
        before = predict_correction(
            training_model.eval(), sample["descriptor"], sample["dq_dR_relaxed"], require_force=True
        )
        after = predict_correction(
            loaded_model, sample["descriptor"], sample["dq_dR_relaxed"], require_force=True
        )
        checkpoint_error = {
            "energy": error_norms(after.energy.detach().numpy(), before.energy.detach().numpy()),
            "force": error_norms(after.force.detach().numpy(), before.force.detach().numpy()),
        }
        reader_records[str(batch_size)] = {
            "load_seconds": load_seconds,
            "evaluation": evaluation,
            "energy_only_evaluation": energy_evaluation,
            "training": training,
            "checkpoint_hash": sha256_file(checkpoint),
            "checkpoint_reload_error": checkpoint_error,
        }
        del reader
        gc.collect()
    from deepks.model.test import main as saved_data_test

    saved_test_start = time.perf_counter()
    saved_test_result = saved_data_test(
        [str(directory)],
        model_file=str(checkpoint),
        output_prefix=None,
        force_mode="deephf_relaxed",
    )
    saved_test_seconds = time.perf_counter() - saved_test_start
    total_bytes = sum(path.stat().st_size for path in directory.iterdir())
    for record in reader_records.values():
        record["bytes_read"] = total_bytes
    result["force_data"] = {
        "payload": payload,
        "reference_seconds": reference_seconds,
        "single_frame_generation_seconds": generation_seconds,
        "generation_seconds_per_frame": generation_seconds,
        "serialization_seconds": serialization_seconds,
        "dataset_bytes": total_bytes,
        "jacobian_bytes": jacobian_bytes,
        "jacobian_bytes_per_frame": jacobian_bytes // int(payload["frames"]),
        "contract_fingerprint": contract.compatibility_fingerprint,
        "batch_records": reader_records,
        "saved_data_test_seconds": saved_test_seconds,
        "saved_data_test_result": saved_test_result,
    }
    large_requirement = label != "large" or jacobian_bytes > 2**30
    reload_pass = bool(
        checkpoint_error
        and checkpoint_error["energy"]["max_abs"] == 0.0
        and checkpoint_error["force"]["max_abs"] == 0.0
    )
    result["categories"]["scientific"] = {
        "passed": reload_pass,
        "reasons": [] if reload_pass else ["checkpoint restart changed predictions"],
    }
    result["categories"]["integrity"] = {
        "passed": large_requirement,
        "reasons": [] if large_requirement else ["large Jacobian payload did not exceed one GiB"],
    }
    write_json(output, result)
    return result


def action_force_data_physical(output: Path, _workload_id: str | None, _family: str | None) -> dict[str, Any]:
    from deepks.deephf import build_reference, write_rhf_force_dataset
    from deepks.model.reader import Reader

    profile, threads = _profile()
    result = base_result("force-data-physical", profile, threads)
    result["comparison_targets"] = ["pyscf-native", "direct-detailed"]
    workload = workload_by_id("L1-def2-SVP")
    atoms, coordinates = workload_geometry(workload)
    atoms = atoms[:6]
    coordinates = coordinates[:6]
    references = []
    energies = []
    forces = []
    timings = []
    for frame_index in range(load_config()["force_data"]["physical_frames"]):
        displacement = 0.004 * np.sin(frame_index + np.arange(18)).reshape(6, 3)
        molecule = gto.M(
            atom=list(zip(atoms, coordinates + displacement)),
            basis="6-31G",
            unit="Bohr",
            charge=0,
            spin=0,
            symmetry=False,
            cart=False,
            verbose=0,
        )
        start = time.perf_counter()
        reference = build_reference(
            molecule, "rhf", scf_args=load_config()["scf_controls"], verbose=0
        )
        rmp2 = mp.MP2(reference).run(verbose=0)
        gradient = np.asarray(rmp2.nuc_grad_method().kernel())
        timings.append(time.perf_counter() - start)
        references.append(reference)
        energies.append(float(reference.e_tot + rmp2.e_corr))
        forces.append(-gradient)
    directory = output.parent / "dataset"
    generation_start = time.perf_counter()
    contract = write_rhf_force_dataset(
        directory,
        references,
        projector_basis=load_config()["projector_basis"],
        e_target=np.asarray(energies),
        f_target=np.asarray(forces),
        target={
            "method": "RMP2",
            "basis": "6-31G",
            "software": "PySCF",
            "version": pyscf.__version__,
            "frozen_core": False,
            "relativistic": "none",
            "state": "closed-shell singlet ground state",
            "energy_force_consistent": True,
            "settings": {"frozen": None},
        },
        response_options=load_config()["response_controls"],
    )
    generation_seconds = time.perf_counter() - generation_start
    reader_start = time.perf_counter()
    reader = Reader(directory, batch_size=4, force_mode="deephf_relaxed")
    reader_seconds = time.perf_counter() - reader_start
    result["physical_force_data"] = {
        "frame_count": len(references),
        "rmp2_target_statistics": statistics(timings),
        "generation_seconds": generation_seconds,
        "reader_seconds": reader_seconds,
        "contract_fingerprint": contract.compatibility_fingerprint,
        "energy_targets": energies,
        "force_targets": forces,
        "all_finite": bool(np.isfinite(energies).all() and np.isfinite(forces).all()),
    }
    passed = result["physical_force_data"]["all_finite"] and reader.nframes == len(references)
    result["categories"]["scientific"] = {
        "passed": passed,
        "reasons": [] if passed else ["physical RMP2 force dataset is incomplete or nonfinite"],
    }
    write_json(output, result)
    return result


ACTIONS = {
    "checkpoint": action_checkpoint,
    "preflight": action_preflight,
    "scientific": action_scientific,
    "dense": action_dense,
    "selection": action_selection,
    "invariance": action_invariance,
    "conditioning": action_conditioning,
    "dft-sequence": action_dft_sequence,
    "scanner": action_scanner,
    "force-data-physical": action_force_data_physical,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", choices=sorted((*ACTIONS, "benchmark", "native-benchmark", "force-data", "atom-subset", "coordinate-block", "verification")), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workload")
    parser.add_argument("--family")
    parser.add_argument("--backend", choices=("direct", "zvector"))
    parser.add_argument("--mode", choices=("compact", "detailed"))
    parser.add_argument("--label")
    arguments = parser.parse_args()
    profile, threads = _profile()
    configure_threads(threads)
    result = None
    try:
        if arguments.action == "verification":
            result = action_verification(arguments.output, arguments.label)
        elif arguments.action == "native-benchmark":
            result = action_native_benchmark(
                arguments.output,
                arguments.workload,
                arguments.family,
            )
        elif arguments.action == "benchmark":
            result = action_benchmark(
                arguments.output,
                arguments.workload,
                arguments.family,
                arguments.backend,
                arguments.mode == "detailed",
            )
        elif arguments.action == "force-data":
            result = action_force_data(
                arguments.output,
                arguments.workload,
                arguments.family,
                arguments.label,
            )
        elif arguments.action == "atom-subset":
            result = action_subset(
                arguments.output,
                arguments.workload,
                arguments.family,
                arguments.backend,
                arguments.label,
            )
        elif arguments.action == "coordinate-block":
            result = action_block(
                arguments.output,
                arguments.workload,
                arguments.family,
                arguments.label,
            )
        else:
            result = ACTIONS[arguments.action](
                arguments.output,
                arguments.workload,
                arguments.family,
            )
    except Exception as error:
        workload = workload_by_id(arguments.workload) if arguments.workload else None
        result = base_result(arguments.action, profile, threads, workload, arguments.family)
        report_exception(result, "scientific", error)
        write_json(arguments.output, result)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
