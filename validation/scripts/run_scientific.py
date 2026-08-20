"""Run the independent finite-difference and public-interface science gates."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from pyscf import gto, lib
from pyscf.lib import param

from deepks.deephf import evaluate_molecule, make_deephf
from deepks.deephf.workflow import main as workflow_main
from deepks.model.model import CorrNet

from common import (
    OUTPUT_DIR,
    PROJECTOR_BASIS,
    REPORT_DIR,
    REPOSITORY_DIR,
    VALIDATION_DIR,
    configure_single_thread,
    deterministic_model,
    environment_metadata,
    error_statistics,
    finite_difference,
    fresh_reference,
    json_safe,
    load_config,
    make_method,
    molecule,
    occupied_subspace_overlap,
    read_xyz,
    report_exception,
    sha256_file,
    state_summary,
    write_json,
)


FAMILY_INPUTS = {
    "rhf": ("formaldehyde.xyz", 0),
    "rks": ("formaldehyde.xyz", 0),
    "uhf": ("hydroxymethyl.xyz", 1),
    "uks": ("hydroxymethyl.xyz", 1),
}


def _coordinates_bohr(family: str) -> tuple[tuple[str, ...], np.ndarray]:
    filename, _spin = FAMILY_INPUTS[family]
    atoms, coordinates_angstrom = read_xyz(VALIDATION_DIR / "geometries" / filename)
    return atoms, coordinates_angstrom / float(param.BOHR)


def _driver_diagnostics(driver, backend: str) -> dict:
    result = driver.response_result if backend == "direct" else driver.adjoint_result
    diagnostics = result.diagnostics
    values = json_safe(diagnostics)
    for name in (
        "maximum_residual",
        "residual_tolerance",
        "response_dimension",
        "maximum_solver_residual",
        "maximum_transpose_residual",
        "maximum_physical_residual",
        "solve_count",
    ):
        if hasattr(diagnostics, name):
            values[name] = json_safe(getattr(diagnostics, name))
    return values


def _state_continuity(
    central,
    displaced,
    central_summary: dict,
) -> dict:
    displaced_summary = state_summary(displaced)
    occupations_match = np.array_equal(
        np.asarray(central.mo_occ), np.asarray(displaced.mo_occ)
    )
    electron_counts_match = (
        displaced_summary["electron_counts"] == central_summary["electron_counts"]
    )
    dimensions_match = (
        displaced_summary["response_dimensions"]
        == central_summary["response_dimensions"]
    )
    overlaps = occupied_subspace_overlap(central, displaced)
    gaps = np.asarray(displaced_summary["minimum_orbital_gap"], dtype=np.float64)
    passed = bool(
        occupations_match
        and electron_counts_match
        and dimensions_match
        and displaced_summary["spin"] == central_summary["spin"]
        and np.min(gaps) > 1.0e-7
        and np.min(overlaps) > 0.9
    )
    return {
        "passed": passed,
        "occupations_match": occupations_match,
        "electron_counts_match": electron_counts_match,
        "response_dimensions_match": dimensions_match,
        "spin_match": displaced_summary["spin"] == central_summary["spin"],
        "minimum_orbital_gap": displaced_summary["minimum_orbital_gap"],
        "occupied_subspace_minimum_singular_values": overlaps,
    }


def _finite_difference_matrix(
    family: str,
    atoms: tuple[str, ...],
    coordinates_bohr: np.ndarray,
    central_reference,
    model,
    output: Path,
) -> dict:
    config = load_config()
    steps = tuple(float(step) for step in config["finite_difference_steps_bohr"])
    central_summary = state_summary(central_reference)
    plus_energy: dict[tuple[float, int, int], np.ndarray] = {}
    minus_energy: dict[tuple[float, int, int], np.ndarray] = {}
    plus_descriptor: dict[tuple[float, int, int], np.ndarray] = {}
    minus_descriptor: dict[tuple[float, int, int], np.ndarray] = {}
    continuity = []
    timings = []
    for step in steps:
        for atom in range(len(atoms)):
            for coordinate in range(3):
                for sign, energy_store, descriptor_store in (
                    (1, plus_energy, plus_descriptor),
                    (-1, minus_energy, minus_descriptor),
                ):
                    displaced_coordinates = coordinates_bohr.copy()
                    displaced_coordinates[atom, coordinate] += sign * step
                    started = time.perf_counter()
                    displaced_reference = fresh_reference(
                        family, atoms, displaced_coordinates
                    )
                    continuity_result = _state_continuity(
                        central_reference, displaced_reference, central_summary
                    )
                    if not continuity_result["passed"]:
                        raise RuntimeError(
                            f"{family.upper()} state continuity failed at step={step}, atom={atom}, coordinate={coordinate}, sign={sign}"
                        )
                    displaced_method = make_method(displaced_reference, model)
                    total_energy = np.asarray(
                        displaced_method.kernel(), dtype=np.float64
                    )
                    descriptor = np.asarray(
                        displaced_method.descriptor(), dtype=np.float64
                    )
                    key = (step, atom, coordinate)
                    energy_store[key] = total_energy
                    descriptor_store[key] = descriptor
                    timings.append(time.perf_counter() - started)
                    continuity.append(
                        {
                            "step_bohr": step,
                            "atom": atom,
                            "coordinate": coordinate,
                            "sign": sign,
                            **continuity_result,
                        }
                    )
    finite_differences = {}
    for step in steps:
        energy_fd = finite_difference(
            plus_energy,
            minus_energy,
            step=step,
            atom_count=len(atoms),
        ).reshape(len(atoms), 3)
        descriptor_fd = finite_difference(
            plus_descriptor,
            minus_descriptor,
            step=step,
            atom_count=len(atoms),
        )
        np.save(output / f"energy_fd_{step:.0e}.npy", energy_fd)
        np.save(output / f"descriptor_fd_{step:.0e}.npy", descriptor_fd)
        finite_differences[step] = {
            "energy": energy_fd,
            "descriptor": descriptor_fd,
        }
    return {
        "steps_bohr": steps,
        "values": finite_differences,
        "state_continuity": continuity,
        "wall_time_seconds": {
            "total": float(np.sum(timings)),
            "minimum_per_displacement": float(np.min(timings)),
            "maximum_per_displacement": float(np.max(timings)),
            "median_per_displacement": float(np.median(timings)),
        },
    }


def _central_science(family: str, output: Path) -> tuple[dict, dict]:
    config = load_config()
    acceptance = config["acceptance"]
    atoms, coordinates_bohr = _coordinates_bohr(family)
    reference_started = time.perf_counter()
    reference = fresh_reference(family, atoms, coordinates_bohr)
    reference_wall = time.perf_counter() - reference_started
    model = deterministic_model()
    method = make_method(reference, model)
    energy = float(method.kernel())
    descriptor_diagnostics = method.validate_force_compatibility()
    direct_started = time.perf_counter()
    direct = method.nuc_grad_method(backend="direct").run()
    direct_wall = time.perf_counter() - direct_started
    zvector_started = time.perf_counter()
    zvector = method.nuc_grad_method(backend="zvector").run()
    zvector_wall = time.perf_counter() - zvector_started

    native_driver = reference.nuc_grad_method()
    if family in {"rks", "uks"}:
        native_driver.grid_response = True
    native_gradient = np.asarray(native_driver.kernel())
    zero_method = make_method(reference, None)
    zero_method.kernel()
    zero_direct = zero_method.nuc_grad_method(backend="direct").run()
    zero_zvector = zero_method.nuc_grad_method(backend="zvector").run()
    zero_direct_error = error_statistics(zero_direct.de_full, native_gradient)
    zero_zvector_error = error_statistics(zero_zvector.de_full, native_gradient)
    backend_total_error = error_statistics(direct.de_full, zvector.de_full)
    backend_correction_error = error_statistics(
        direct.correction_gradient, zvector.correction_gradient
    )
    backend_response_error = error_statistics(
        direct.correction_gradient_response,
        zvector.correction_gradient_response,
    )
    force_sign_exact = np.array_equal(-direct.de_full, direct.forces())
    direct_diagnostics = _driver_diagnostics(direct, "direct")
    zvector_diagnostics = _driver_diagnostics(zvector, "zvector")
    direct_residual = float(direct_diagnostics["maximum_residual"])
    direct_residual_tolerance = float(direct_diagnostics["residual_tolerance"])
    adjoint_residual = max(
        float(zvector_diagnostics["maximum_solver_residual"]),
        float(zvector_diagnostics["maximum_transpose_residual"]),
        float(zvector_diagnostics["maximum_physical_residual"]),
    )
    adjoint_residual_tolerance = float(
        zvector_diagnostics["residual_tolerance"]
    )
    central = {
        "passed": bool(
            zero_direct_error["max_abs"] <= acceptance["zero_native_max_abs"]
            and zero_zvector_error["max_abs"] <= acceptance["zero_native_max_abs"]
            and backend_total_error["max_abs"] <= acceptance["backend_max_abs"]
            and backend_correction_error["max_abs"] <= acceptance["backend_max_abs"]
            and backend_response_error["max_abs"] <= acceptance["backend_max_abs"]
            and force_sign_exact
            and direct_residual <= direct_residual_tolerance
            and adjoint_residual <= adjoint_residual_tolerance
            and int(zvector_diagnostics["solve_count"]) == 1
        ),
        "energy": {
            "reference_eh": float(reference.e_tot),
            "correction_eh": float(method.e_corr),
            "total_eh": energy,
            "identity_error_eh": float(
                energy - (float(reference.e_tot) + float(method.e_corr))
            ),
        },
        "state": state_summary(reference),
        "descriptor_differentiability": json_safe(descriptor_diagnostics),
        "zero_direct_vs_native": zero_direct_error,
        "zero_zvector_vs_native": zero_zvector_error,
        "direct_vs_zvector_total": backend_total_error,
        "direct_vs_zvector_correction": backend_correction_error,
        "direct_vs_zvector_response": backend_response_error,
        "force_is_exact_negative_gradient": force_sign_exact,
        "direct_diagnostics": direct_diagnostics,
        "zvector_diagnostics": zvector_diagnostics,
        "wall_time_seconds": {
            "native_reference": reference_wall,
            "direct_gradient": direct_wall,
            "zvector_gradient": zvector_wall,
        },
    }
    arrays = {
        "atoms": atoms,
        "coordinates_bohr": coordinates_bohr,
        "reference": reference,
        "model": model,
        "method": method,
        "direct": direct,
        "zvector": zvector,
        "native_gradient": native_gradient,
    }
    for name, value in (
        ("coordinates_bohr", coordinates_bohr),
        ("descriptor", method.descriptor()),
        ("native_gradient", native_gradient),
        ("direct_gradient", direct.de_full),
        ("zvector_gradient", zvector.de_full),
        ("direct_force", -direct.de_full),
        ("direct_correction_gradient", direct.correction_gradient),
        ("direct_response_gradient", direct.correction_gradient_response),
        ("direct_dq_dR_explicit", direct.dq_dR_explicit),
        ("direct_dq_dR_response", direct.dq_dR_response),
        ("direct_dq_dR_relaxed", direct.dq_dR_relaxed),
    ):
        np.save(output / f"{name}.npy", np.asarray(value, dtype=np.float64))
    return central, arrays


def _finite_difference_assessment(matrix: dict, arrays: dict) -> dict:
    config = load_config()
    acceptance = config["acceptance"]
    direct = arrays["direct"]
    results = {}
    for step, values in matrix["values"].items():
        results[f"{step:.1e}"] = {
            "step_bohr": step,
            "total_gradient": error_statistics(direct.de_full, values["energy"]),
            "relaxed_descriptor_jacobian": error_statistics(
                direct.dq_dR_relaxed, values["descriptor"]
            ),
            "explicit_descriptor_jacobian": error_statistics(
                direct.dq_dR_explicit, values["descriptor"]
            ),
            "explicit_only_total_gradient": error_statistics(
                direct.reference_gradient + direct.correction_gradient_explicit,
                values["energy"],
            ),
        }
    finest_key = f"{min(matrix['steps_bohr']):.1e}"
    finest = results[finest_key]
    relaxed_error = finest["total_gradient"]["max_abs"]
    explicit_error = finest["explicit_only_total_gradient"]["max_abs"]
    ratio = explicit_error / max(relaxed_error, np.finfo(np.float64).tiny)
    response_signal = float(np.max(np.abs(direct.dq_dR_response)))
    response_gradient_signal = float(
        np.max(np.abs(direct.correction_gradient_response))
    )
    passed = bool(
        all(
            item["total_gradient"]["max_abs"]
            <= acceptance["gradient_fd_max_abs"]
            and item["relaxed_descriptor_jacobian"]["max_abs"]
            <= acceptance["descriptor_fd_max_abs"]
            for item in results.values()
        )
        and response_signal > acceptance["response_signal_min"]
        and ratio > acceptance["explicit_error_ratio_min"]
    )
    return {
        "passed": passed,
        "steps": results,
        "anti_vacuity": {
            "response_descriptor_max_abs_bohr_inverse": response_signal,
            "response_correction_gradient_max_abs_eh_per_bohr": response_gradient_signal,
            "explicit_to_relaxed_gradient_error_ratio": ratio,
            "required_response_signal": acceptance["response_signal_min"],
            "required_error_ratio": acceptance["explicit_error_ratio_min"],
        },
        "state_continuity": matrix["state_continuity"],
        "wall_time_seconds": matrix["wall_time_seconds"],
    }


def _rotation_matrix() -> np.ndarray:
    axis = np.asarray([1.0, 2.0, -1.0], dtype=np.float64)
    axis /= np.linalg.norm(axis)
    angle = 0.37
    cross = np.asarray(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return (
        np.eye(3) * np.cos(angle)
        + (1.0 - np.cos(angle)) * np.outer(axis, axis)
        + np.sin(angle) * cross
    )


def _fresh_result(atoms, coordinates_bohr, model, backend="direct"):
    reference = fresh_reference("rhf", atoms, coordinates_bohr)
    method = make_method(reference, model)
    energy = float(method.kernel())
    driver = method.nuc_grad_method(backend=backend).run()
    return reference, method, energy, driver


def _invariance_checks(arrays: dict) -> dict:
    config = load_config()
    acceptance = config["acceptance"]
    atoms = arrays["atoms"]
    coordinates = arrays["coordinates_bohr"]
    model = arrays["model"]
    method = arrays["method"]
    direct = arrays["direct"]
    energy = float(method.e_tot)
    force_sum = np.sum(-direct.de_full, axis=0)

    rotation = _rotation_matrix()
    rotated_coordinates = coordinates @ rotation.T
    _, rotated_method, rotated_energy, rotated_driver = _fresh_result(
        atoms, rotated_coordinates, model
    )
    rotation_gradient = error_statistics(
        rotated_driver.de_full, direct.de_full @ rotation.T
    )
    rotation_descriptor = error_statistics(
        rotated_method.descriptor(), method.descriptor()
    )
    rotation_energy_error = abs(rotated_energy - energy)

    coordinates_angstrom = coordinates * float(param.BOHR)
    angstrom_molecule = molecule(
        atoms,
        coordinates_angstrom,
        unit="Angstrom",
        spin=0,
    )
    from deepks.deephf import build_reference

    angstrom_reference = build_reference(
        angstrom_molecule,
        "rhf",
        scf_args=config["scf_controls"],
        verbose=0,
    )
    angstrom_method = make_method(angstrom_reference, model)
    angstrom_energy = float(angstrom_method.kernel())
    angstrom_driver = angstrom_method.nuc_grad_method(backend="direct").run()
    unit_checks = {
        "energy_abs_error_eh": abs(angstrom_energy - energy),
        "descriptor": error_statistics(
            angstrom_method.descriptor(), method.descriptor()
        ),
        "gradient": error_statistics(angstrom_driver.de_full, direct.de_full),
        "force": error_statistics(-angstrom_driver.de_full, -direct.de_full),
    }

    permutation = np.asarray([0, 1, 3, 2], dtype=np.int64)
    permuted_atoms = tuple(atoms[index] for index in permutation)
    permuted_coordinates = coordinates[permutation]
    _, permuted_method, permuted_energy, permuted_driver = _fresh_result(
        permuted_atoms, permuted_coordinates, model
    )
    permutation_checks = {
        "energy_abs_error_eh": abs(permuted_energy - energy),
        "descriptor": error_statistics(
            permuted_method.descriptor(), method.descriptor()[permutation]
        ),
        "gradient": error_statistics(
            permuted_driver.de_full, direct.de_full[permutation]
        ),
    }

    passed = bool(
        np.max(np.abs(force_sum)) <= acceptance["rhf_force_sum_max_abs"]
        and rotation_gradient["max_abs"] <= acceptance["rhf_rotation_max_abs"]
        and rotation_energy_error <= 1.0e-10
        and rotation_descriptor["max_abs"] <= 1.0e-10
        and unit_checks["energy_abs_error_eh"] <= 1.0e-10
        and unit_checks["descriptor"]["max_abs"] <= 1.0e-10
        and unit_checks["gradient"]["max_abs"] <= 1.0e-8
        and permutation_checks["energy_abs_error_eh"] <= 1.0e-10
        and permutation_checks["descriptor"]["max_abs"] <= 1.0e-10
        and permutation_checks["gradient"]["max_abs"] <= 1.0e-8
    )
    return {
        "passed": passed,
        "force_sum_eh_per_bohr": force_sum,
        "force_sum_max_abs": float(np.max(np.abs(force_sum))),
        "rotation": {
            "matrix": rotation,
            "energy_abs_error_eh": rotation_energy_error,
            "descriptor": rotation_descriptor,
            "gradient": rotation_gradient,
        },
        "unit_equivalence": unit_checks,
        "hydrogen_permutation": {
            "permutation": permutation,
            **permutation_checks,
        },
    }


def _scanner_checks(arrays: dict) -> dict:
    atoms = arrays["atoms"]
    coordinates = arrays["coordinates_bohr"]
    model = arrays["model"]
    method = arrays["method"]
    forward = coordinates.copy()
    forward[2] += np.asarray([0.012, -0.009, 0.007])
    backward = coordinates.copy()
    backward[1] += np.asarray([-0.008, 0.006, -0.010])
    sequence = (coordinates, forward, backward, coordinates)
    backend_results = {}
    for backend in ("direct", "zvector"):
        scanner = method.nuc_grad_method(backend=backend).as_scanner()
        energy_errors = []
        gradient_errors = []
        object_graphs = []
        for frame_coordinates in sequence:
            scanned_energy, scanned_gradient = scanner(frame_coordinates)
            _, _, fresh_energy, fresh_driver = _fresh_result(
                atoms, frame_coordinates, model, backend=backend
            )
            energy_errors.append(abs(float(scanned_energy) - fresh_energy))
            gradient_errors.append(
                error_statistics(scanned_gradient, fresh_driver.de_full)
            )
            object_graphs.append(
                (
                    scanner.mol,
                    scanner.reference,
                    scanner.method,
                    scanner.gradient_driver,
                )
            )
        fresh_objects = all(
            len({id(frame[column]) for frame in object_graphs}) == len(sequence)
            for column in range(4)
        )
        backend_results[backend] = {
            "passed": bool(
                max(energy_errors) <= 2.0e-12
                and max(item["max_abs"] for item in gradient_errors) <= 2.0e-10
                and fresh_objects
            ),
            "sequence": "A-B-C-A",
            "energy_abs_errors_eh": energy_errors,
            "gradient_errors": gradient_errors,
            "fresh_object_graph_per_frame": fresh_objects,
        }
    return {
        "passed": all(item["passed"] for item in backend_results.values()),
        "backends": backend_results,
    }


def _persisted_values(directory: Path) -> dict[str, np.ndarray]:
    return {
        name: np.load(directory / f"{name}.npy")
        for name in (
            "converged",
            "e_base",
            "e_corr",
            "e_tot",
            "descriptor",
            "gradient",
            "force",
        )
    }


def _find_persisted_directory(root: Path) -> Path:
    matches = sorted(path.parent for path in root.rglob("converged.npy"))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one persisted system under {root}; found {len(matches)}"
        )
    return matches[0]


def _compare_public_result(actual: dict, expected: dict) -> dict:
    fields = {}
    for name in expected:
        actual_value = np.asarray(actual[name])
        expected_value = np.asarray(expected[name])
        if actual_value.shape == (1, *expected_value.shape):
            actual_value = actual_value[0]
        if actual_value.dtype == np.bool_ or expected_value.dtype == np.bool_:
            match = np.array_equal(actual_value, expected_value)
            fields[name] = {"exact_match": match}
        else:
            fields[name] = error_statistics(
                actual_value,
                expected_value,
                coordinate_axis=1 if expected_value.ndim >= 2 else 0,
            )
    passed = all(
        item.get("exact_match", item.get("max_abs", np.inf) <= 2.0e-10)
        for item in fields.values()
    )
    return {"passed": passed, "fields": fields}


def _public_interface_checks(arrays: dict, output: Path) -> dict:
    atoms = arrays["atoms"]
    coordinates = arrays["coordinates_bohr"]
    model = arrays["model"]
    reference = arrays["reference"]
    method = arrays["method"]
    checkpoint = output / "model.pth"
    model.save(checkpoint)
    loaded = CorrNet.load(checkpoint, strict=True).double().eval()
    loaded_method = make_method(reference, loaded)
    loaded_energy = float(loaded_method.kernel())
    loaded_gradient = loaded_method.gradient(backend="direct")
    checkpoint_checks = {
        "sha256": sha256_file(checkpoint),
        "energy_identical": loaded_energy == float(method.e_tot),
        "descriptor_identical": np.array_equal(
            loaded_method.descriptor(), method.descriptor()
        ),
        "gradient_identical": np.array_equal(
            loaded_gradient, arrays["direct"].de_full
        ),
    }
    checkpoint_checks["passed"] = all(
        value
        for name, value in checkpoint_checks.items()
        if name != "sha256"
    )

    expected_direct = {
        "converged": np.asarray(True),
        "e_base": np.asarray(reference.e_tot, dtype=np.float64),
        "e_corr": np.asarray(method.e_corr, dtype=np.float64),
        "e_tot": np.asarray(method.e_tot, dtype=np.float64),
        "descriptor": np.asarray(method.descriptor(), dtype=np.float64),
        "gradient": np.asarray(arrays["direct"].de_full, dtype=np.float64),
        "force": np.asarray(-arrays["direct"].de_full, dtype=np.float64),
    }
    expected_zvector = {
        **expected_direct,
        "gradient": np.asarray(arrays["zvector"].de_full, dtype=np.float64),
        "force": np.asarray(-arrays["zvector"].de_full, dtype=np.float64),
    }
    factory_method = make_deephf(
        reference, loaded, projector_basis=PROJECTOR_BASIS
    )
    factory_method.kernel()
    factory_result = {
        **expected_direct,
        "e_corr": np.asarray(factory_method.e_corr),
        "e_tot": np.asarray(factory_method.e_tot),
        "descriptor": factory_method.descriptor(),
        "gradient": factory_method.gradient(backend="direct"),
        "force": factory_method.forces(backend="direct"),
    }
    factory_checks = _compare_public_result(factory_result, expected_direct)

    helper_molecule = molecule(atoms, coordinates, unit="Bohr", spin=0)
    helper_direct = evaluate_molecule(
        helper_molecule,
        loaded,
        family="rhf",
        backend="direct",
        projector_basis=PROJECTOR_BASIS,
        scf_args=load_config()["scf_controls"],
    )
    helper_checks = _compare_public_result(helper_direct, expected_direct)

    workflow_directory = output / "workflow_direct"
    if workflow_directory.exists():
        shutil.rmtree(workflow_directory)
    workflow_outputs = workflow_main(
        [str(VALIDATION_DIR / "geometries" / "formaldehyde.xyz")],
        reference="rhf",
        model_file=str(checkpoint),
        basis=load_config()["basis"],
        projector_basis=PROJECTOR_BASIS,
        backend="direct",
        dump_dir=str(workflow_directory),
        mol_args={"unit": "Angstrom", "symmetry": False, "cart": False},
        scf_args=load_config()["scf_controls"],
        verbose=0,
    )
    workflow_path, workflow_collected = workflow_outputs[0]
    workflow_checks = _compare_public_result(workflow_collected, expected_direct)
    workflow_persisted = _persisted_values(Path(workflow_path))
    workflow_persistence_checks = _compare_public_result(
        workflow_persisted, expected_direct
    )

    cli_results = {}
    executable = shutil.which("deepks")
    if executable is None:
        raise RuntimeError("the deepks command is unavailable in the validation environment")
    for backend, expected in (
        ("direct", expected_direct),
        ("zvector", expected_zvector),
    ):
        dump_root = output / f"cli_{backend}"
        if dump_root.exists():
            shutil.rmtree(dump_root)
        config_path = VALIDATION_DIR / "configs" / f"deephf_{backend}.yaml"
        process = subprocess.run(
            [executable, "deephf", str(config_path)],
            cwd=REPOSITORY_DIR,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        log_path = REPORT_DIR / f"cli_{backend}.log"
        log_path.write_text(process.stdout, encoding="utf-8")
        if process.returncode != 0:
            raise RuntimeError(
                f"deepks deephf {backend} failed with code {process.returncode}; see {log_path}"
            )
        persisted = _persisted_values(_find_persisted_directory(dump_root))
        cli_results[backend] = {
            "command": [executable, "deephf", str(config_path)],
            "return_code": process.returncode,
            "persisted": _compare_public_result(persisted, expected),
        }
        cli_results[backend]["passed"] = cli_results[backend]["persisted"][
            "passed"
        ]

    passed = bool(
        checkpoint_checks["passed"]
        and factory_checks["passed"]
        and helper_checks["passed"]
        and workflow_checks["passed"]
        and workflow_persistence_checks["passed"]
        and all(item["passed"] for item in cli_results.values())
    )
    return {
        "passed": passed,
        "checkpoint_reload": checkpoint_checks,
        "concrete_method": _compare_public_result(expected_direct, expected_direct),
        "public_factory": factory_checks,
        "public_helper": helper_checks,
        "public_workflow": workflow_checks,
        "workflow_persistence": workflow_persistence_checks,
        "cli": cli_results,
    }


def run_family(family: str) -> dict:
    output = OUTPUT_DIR / family
    output.mkdir(parents=True, exist_ok=True)
    central, arrays = _central_science(family, output)
    matrix = _finite_difference_matrix(
        family,
        arrays["atoms"],
        arrays["coordinates_bohr"],
        arrays["reference"],
        arrays["model"],
        output,
    )
    finite_difference_result = _finite_difference_assessment(matrix, arrays)
    result = {
        "stage": f"scientific_{family}",
        "family": family.upper(),
        "system": FAMILY_INPUTS[family][0],
        "configuration": load_config(),
        "environment": environment_metadata(),
        "central": central,
        "finite_difference": finite_difference_result,
    }
    if family == "rhf":
        result["invariance"] = _invariance_checks(arrays)
        result["scanner"] = _scanner_checks(arrays)
        formaldehyde_output = OUTPUT_DIR / "formaldehyde"
        formaldehyde_output.mkdir(parents=True, exist_ok=True)
        arrays["model"].save(formaldehyde_output / "model.pth")
        result["public_interfaces"] = _public_interface_checks(
            arrays, formaldehyde_output
        )
    section_passes = [
        item["passed"]
        for item in result.values()
        if isinstance(item, dict) and "passed" in item
    ]
    result["passed"] = bool(section_passes and all(section_passes))
    write_json(REPORT_DIR / f"scientific_{family}.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--family", choices=tuple(FAMILY_INPUTS), required=True
    )
    arguments = parser.parse_args()
    configure_single_thread()
    lib.num_threads(1)
    try:
        result = run_family(arguments.family)
    except BaseException as error:
        failure = {
            **report_exception(f"scientific_{arguments.family}", error),
            "environment": environment_metadata(),
        }
        write_json(REPORT_DIR / f"scientific_{arguments.family}.json", failure)
        raise
    print(json.dumps({"stage": result["stage"], "passed": result["passed"]}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
