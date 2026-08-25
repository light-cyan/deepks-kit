"""Identify exact displaced points that fail SCF, state continuity, or strict audits."""

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
from pyscf import dft, scf


REPOSITORY_DIR = Path(__file__).resolve().parents[2]
VALIDATION_DIR = REPOSITORY_DIR / "validation" / "scientific_performance"
sys.path.insert(0, str(VALIDATION_DIR / "scripts"))

from common import (  # noqa: E402
    configure_threads,
    deterministic_directions,
    effective_scf_controls,
    finite_difference_components,
    finite_difference_steps,
    hash_array,
    make_method,
    make_molecule,
    max_abs,
    state_continuity,
    workload_by_id,
    workload_geometry,
    write_json,
)


AXES = ("x", "y", "z")


def _reference(workload: dict[str, Any], family: str, coordinates: np.ndarray, scf_controls: dict[str, Any]):
    molecule = make_molecule(workload, coordinates)
    constructors = {
        "rhf": scf.RHF,
        "uhf": scf.UHF,
        "rks": dft.RKS,
        "uks": dft.UKS,
    }
    reference = constructors[family](molecule)
    reference.verbose = 0
    reference.set(**scf_controls)
    if family in {"rks", "uks"}:
        reference.xc = "LDA_X + LDA_C_VWN"
        reference.grids.atom_grid = {symbol: (20, 50) for symbol in set(molecule.elements)}
        reference.grids.prune = None
        reference.grids.alignment = 1
        reference.grids.cutoff = 1.0e-15
        reference.grids.build(with_non0tab=True, sort_grids=False)
        reference.small_rho_cutoff = 0.0
    history = []

    def callback(environment):
        history.append(
            {
                "cycle": int(environment.get("cycle", len(history))),
                "energy": float(environment["e_tot"]) if environment.get("e_tot") is not None else None,
                "orbital_gradient_norm": float(environment["norm_gorb"]) if environment.get("norm_gorb") is not None else None,
                "density_change_norm": float(environment["norm_ddm"]) if environment.get("norm_ddm") is not None else None,
            }
        )

    reference.callback = callback
    reference.kernel(dm0=None)
    reference.callback = None
    if reference.converged:
        density = reference.make_rdm1(reference.mo_coeff, reference.mo_occ)
        fock = reference.get_fock(dm=density)
        reference.mo_energy, reference.mo_coeff = reference.canonicalize(
            reference.mo_coeff,
            reference.mo_occ,
            fock=fock,
        )
    try:
        orbital_gradient_max_abs = max_abs(reference.get_grad())
    except Exception:
        orbital_gradient_max_abs = None
    return reference, history, orbital_gradient_max_abs


def _history_summary(history: list[dict[str, Any]], scf_controls: dict[str, Any]) -> dict[str, Any]:
    energy_changes = [abs(current["energy"] - previous["energy"]) for previous, current in zip(history, history[1:])]
    joint_matches = []
    for index, energy_change in enumerate(energy_changes, start=1):
        gradient_norm = history[index]["orbital_gradient_norm"]
        if gradient_norm is not None and energy_change < scf_controls["conv_tol"] and gradient_norm < scf_controls["conv_tol_grad"]:
            joint_matches.append(index)
    return {
        "cycles": len(history),
        "last": history[-1] if history else None,
        "last_energy_change_abs": energy_changes[-1] if energy_changes else None,
        "minimum_energy_change_abs": min(energy_changes, default=None),
        "minimum_orbital_gradient_norm": min((item["orbital_gradient_norm"] for item in history if item["orbital_gradient_norm"] is not None), default=None),
        "minimum_density_change_norm": min((item["density_change_norm"] for item in history if item["density_change_norm"] is not None), default=None),
        "joint_energy_and_gradient_matches": joint_matches,
    }


def _uks_canonical_residuals(reference) -> dict[str, float]:
    from pyscf.scf import hf as scf_hf
    from deepks.deephf.audits.unrestricted_reference import _dense_uks_quadrature

    molecule = reference.mol
    coefficient = np.asarray(reference.mo_coeff)
    energy = np.asarray(reference.mo_energy)
    overlap = np.asarray(reference.get_ovlp())
    hcore = np.asarray(reference.get_hcore())
    density = np.asarray(reference.make_rdm1())
    coulomb, _exchange = scf_hf.get_jk(molecule, density, hermi=1)
    total_coulomb = np.asarray(coulomb[0] + coulomb[1])
    _electron_counts, _xc_energy, xc_potential = _dense_uks_quadrature(reference, density)
    fock = hcore[None] + total_coulomb[None] + xc_potential
    return {
        name: max_abs(fock[index] @ coefficient[index] - overlap @ (coefficient[index] * energy[index]))
        for index, name in enumerate(("alpha", "beta"))
    }


def _configured_points(workload: dict[str, Any]):
    _, central = workload_geometry(workload)
    for step in finite_difference_steps(workload):
        for atom, axis in finite_difference_components(workload):
            for sign in (-1, 1):
                coordinates = central.copy()
                coordinates[atom, axis] += sign * step
                yield {
                    "kind": "component",
                    "step_bohr": step,
                    "atom": atom,
                    "axis": axis,
                    "axis_name": AXES[axis],
                    "sign": sign,
                }, coordinates
        for direction_index, direction in enumerate(deterministic_directions(workload)):
            for sign in (-1, 1):
                yield {
                    "kind": "direction",
                    "step_bohr": step,
                    "direction_index": direction_index,
                    "sign": sign,
                }, central + sign * step * direction


def _single_point(workload: dict[str, Any], arguments):
    _, central = workload_geometry(workload)
    if arguments.kind == "component":
        axis = AXES.index(arguments.axis)
        coordinates = central.copy()
        coordinates[arguments.atom, axis] += arguments.sign * arguments.step
        context = {
            "kind": "component",
            "step_bohr": arguments.step,
            "atom": arguments.atom,
            "axis": axis,
            "axis_name": arguments.axis,
            "sign": arguments.sign,
        }
    else:
        direction = deterministic_directions(workload)[arguments.direction_index]
        coordinates = central + arguments.sign * arguments.step * direction
        context = {
            "kind": "direction",
            "step_bohr": arguments.step,
            "direction_index": arguments.direction_index,
            "sign": arguments.sign,
        }
    return [(context, coordinates)]


def _point_result(workload, family, central_reference, context, coordinates, scf_controls):
    started = time.perf_counter()
    result = {**context, "coordinates_hash": hash_array(coordinates)}
    try:
        reference, history, orbital_gradient = _reference(workload, family, coordinates, scf_controls)
    except Exception as error:
        return {
            **result,
            "passed": False,
            "failure_stage": "scf_execution",
            "exception": {"type": type(error).__name__, "message": str(error)},
            "elapsed_seconds": time.perf_counter() - started,
        }
    result.update(
        {
            "scf_converged": bool(reference.converged),
            "scf_energy": float(reference.e_tot),
            "scf_cycle_history": history,
            "scf_cycle_summary": _history_summary(history, scf_controls),
            "orbital_gradient_max_abs": orbital_gradient,
        }
    )
    if not reference.converged:
        result.update(
            {
                "passed": False,
                "failure_stage": "scf_convergence",
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        return result
    try:
        continuity = state_continuity(central_reference, reference)
        result["state_continuity"] = continuity
    except Exception as error:
        result.update(
            {
                "passed": False,
                "failure_stage": "state_continuity",
                "exception": {"type": type(error).__name__, "message": str(error)},
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        return result
    if not continuity["accepted"]:
        result.update({"passed": False, "failure_stage": "state_continuity", "elapsed_seconds": time.perf_counter() - started})
        return result
    if family == "uks":
        try:
            result["uks_canonical_residuals"] = _uks_canonical_residuals(reference)
        except Exception as error:
            result["uks_canonical_residual_error"] = {"type": type(error).__name__, "message": str(error)}
    try:
        make_method(reference, None)
    except Exception as error:
        result.update(
            {
                "passed": False,
                "failure_stage": "method_validation",
                "exception": {"type": type(error).__name__, "message": str(error)},
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        return result
    result.update({"passed": True, "failure_stage": None, "elapsed_seconds": time.perf_counter() - started})
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", required=True)
    parser.add_argument("--family", required=True, choices=("rhf", "uhf", "rks", "uks"))
    parser.add_argument("--scope", choices=("configured", "point"), default="configured")
    parser.add_argument("--kind", choices=("component", "direction"), default="component")
    parser.add_argument("--step", type=float)
    parser.add_argument("--atom", type=int)
    parser.add_argument("--axis", choices=AXES)
    parser.add_argument("--direction-index", type=int)
    parser.add_argument("--sign", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--conv-tol", type=float)
    parser.add_argument("--conv-tol-grad", type=float)
    parser.add_argument("--max-cycle", type=int)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.scope == "point":
        if arguments.step is None:
            parser.error("--step is required for --scope point")
        if arguments.kind == "component" and (arguments.atom is None or arguments.axis is None):
            parser.error("--atom and --axis are required for a component point")
        if arguments.kind == "direction" and arguments.direction_index is None:
            parser.error("--direction-index is required for a direction point")
    configure_threads(1)
    workload = workload_by_id(arguments.workload)
    frozen_scf_controls = effective_scf_controls(workload)
    scf_controls = dict(frozen_scf_controls)
    for name in ("conv_tol", "conv_tol_grad", "max_cycle"):
        value = getattr(arguments, name)
        if value is not None:
            scf_controls[name] = value
    _, central_coordinates = workload_geometry(workload)
    central_reference, central_history, central_orbital_gradient = _reference(workload, arguments.family, central_coordinates, scf_controls)
    if not central_reference.converged:
        raise RuntimeError("the central diagnostic reference did not converge")
    points = _configured_points(workload) if arguments.scope == "configured" else iter(_single_point(workload, arguments))
    output = arguments.output or Path(__file__).with_name("results") / f"displacements__{arguments.workload}__{arguments.family}.json"
    result = {
        "experiment": "displacement_scan",
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "workload_id": arguments.workload,
        "family": arguments.family,
        "scope": arguments.scope,
        "frozen_scf_controls": frozen_scf_controls,
        "effective_scf_controls": scf_controls,
        "central": {
            "converged": True,
            "energy": float(central_reference.e_tot),
            "cycle_history": central_history,
            "cycle_summary": _history_summary(central_history, scf_controls),
            "orbital_gradient_max_abs": central_orbital_gradient,
            "coordinates_hash": hash_array(central_coordinates),
        },
        "points": [],
    }
    for index, (context, coordinates) in enumerate(points):
        if arguments.limit is not None and index >= arguments.limit:
            break
        point = _point_result(workload, arguments.family, central_reference, context, coordinates, scf_controls)
        result["points"].append(point)
        write_json(output, result)
        print(json.dumps({"index": index, "context": context, "passed": point["passed"], "failure_stage": point["failure_stage"]}, sort_keys=True), flush=True)
        if arguments.stop_on_failure and not point["passed"]:
            break
    result["summary"] = {
        "completed_points": len(result["points"]),
        "passed_points": sum(point["passed"] for point in result["points"]),
        "failed_points": sum(not point["passed"] for point in result["points"]),
    }
    write_json(output, result)
    print(f"Wrote {output}")
    return 1 if result["summary"]["failed_points"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
