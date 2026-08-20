"""Run deterministic teacher-student and physical RMP2 force-training workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
from pyscf import lib, mp
from pyscf.lib import param

from deepks.data.force_schema import load_force_dataset
from deepks.deephf import write_rhf_force_dataset
from deepks.model.evaluate import predict_correction
from deepks.model.model import CorrNet
from deepks.model.reader import FORCE_MODE_DEEPHF_RELAXED, GroupReader
from deepks.model.test import main as saved_data_main
from deepks.model.train import train

from common import (
    OUTPUT_DIR,
    PROJECTOR_BASIS,
    REPORT_DIR,
    VALIDATION_DIR,
    configure_single_thread,
    deterministic_model,
    environment_metadata,
    error_statistics,
    fresh_reference,
    load_config,
    make_method,
    read_xyz,
    report_exception,
    sha256_file,
    write_json,
    zero_trainable_model,
)


SPLIT_NAMES = ("train", "validation", "heldout")


def _water_dimer_geometries() -> tuple[tuple[str, ...], list[np.ndarray]]:
    atoms, base = read_xyz(VALIDATION_DIR / "geometries" / "water_dimer.xyz")
    separations = (-0.18, -0.12, -0.06, 0.00, 0.06, 0.12, -0.15, 0.09, -0.03, 0.16)
    angles = (-0.10, 0.07, -0.04, 0.11, -0.08, 0.03, 0.13, -0.12, 0.06, -0.02)
    donor_stretches = (0.012, -0.010, 0.018, -0.016, 0.007, -0.005, 0.021, -0.019, 0.014, -0.013)
    acceptor_stretches = (-0.009, 0.013, -0.006, 0.016, -0.014, 0.008, -0.011, 0.019, -0.017, 0.010)
    out_of_plane = (0.025, -0.020, 0.015, -0.030, 0.018, -0.012, 0.028, -0.024, 0.021, -0.016)
    geometries = []
    for index in range(10):
        coordinates = base.copy()
        coordinates[3:, 0] += separations[index]
        origin = coordinates[3].copy()
        cosine = np.cos(angles[index])
        sine = np.sin(angles[index])
        rotation = np.asarray(
            [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        coordinates[4:] = origin + (coordinates[4:] - origin) @ rotation.T
        donor_vector = coordinates[2] - coordinates[0]
        coordinates[2] = coordinates[0] + donor_vector * (1.0 + donor_stretches[index])
        acceptor_vector = coordinates[5] - coordinates[3]
        coordinates[5] = coordinates[3] + acceptor_vector * (1.0 + acceptor_stretches[index])
        coordinates[1, 2] += out_of_plane[index]
        coordinates[4, 2] -= 0.7 * out_of_plane[index]
        geometries.append(coordinates / float(param.BOHR))
    return atoms, geometries


def _split_slices() -> dict[str, slice]:
    splits = load_config()["training"]["splits"]
    train_stop = int(splits["train"])
    validation_stop = train_stop + int(splits["validation"])
    heldout_stop = validation_stop + int(splits["heldout"])
    return {
        "train": slice(0, train_stop),
        "validation": slice(train_stop, validation_stop),
        "heldout": slice(validation_stop, heldout_stop),
    }


def _teacher_targets(references, teacher) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    energies = []
    forces = []
    diagnostics = []
    for reference in references:
        method = make_method(reference, teacher)
        energy = float(method.kernel())
        driver = method.nuc_grad_method(backend="direct").run()
        energies.append(energy)
        forces.append(-driver.de_full)
        diagnostics.append(
            {
                "reference_energy_eh": float(reference.e_tot),
                "correction_energy_eh": float(method.e_corr),
                "response_max_abs_eh_per_bohr": float(
                    np.max(np.abs(driver.correction_gradient_response))
                ),
                "response_diagnostics": driver.response_result.diagnostics,
            }
        )
    return (
        np.asarray(energies, dtype=np.float64),
        np.asarray(forces, dtype=np.float64),
        diagnostics,
    )


def _rmp2_targets(references) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    energies = []
    forces = []
    diagnostics = []
    for reference in references:
        started = time.perf_counter()
        calculation = mp.MP2(reference).run(verbose=0)
        gradient = np.asarray(
            calculation.nuc_grad_method().kernel(verbose=0), dtype=np.float64
        )
        if not np.isfinite(calculation.e_tot) or not np.isfinite(gradient).all():
            raise RuntimeError("RMP2 returned a nonfinite target")
        energies.append(float(calculation.e_tot))
        forces.append(-gradient)
        diagnostics.append(
            {
                "reference_energy_eh": float(reference.e_tot),
                "mp2_correlation_energy_eh": float(calculation.e_corr),
                "mp2_total_energy_eh": float(calculation.e_tot),
                "wall_time_seconds": time.perf_counter() - started,
            }
        )
    return (
        np.asarray(energies, dtype=np.float64),
        np.asarray(forces, dtype=np.float64),
        diagnostics,
    )


def _write_splits(
    workflow: str,
    references: list,
    energies: np.ndarray,
    forces: np.ndarray,
) -> tuple[dict, dict]:
    root = OUTPUT_DIR / "training" / workflow
    root.mkdir(parents=True, exist_ok=True)
    contracts = {}
    generation_times = {}
    for split, frame_slice in _split_slices().items():
        directory = root / f"force_{split}"
        if directory.exists():
            shutil.rmtree(directory)
        started = time.perf_counter()
        contract = write_rhf_force_dataset(
            directory,
            references[frame_slice],
            projector_basis=PROJECTOR_BASIS,
            e_target=np.asarray(energies[frame_slice], dtype=np.float64),
            f_target=np.asarray(forces[frame_slice], dtype=np.float64),
        )
        generation_times[split] = time.perf_counter() - started
        loaded, arrays = load_force_dataset(directory)
        if loaded.manifest_fingerprint != contract.manifest_fingerprint:
            raise RuntimeError(f"{workflow} {split} manifest hash changed on reload")
        energy_directory = root / f"energy_{split}"
        if energy_directory.exists():
            shutil.rmtree(energy_directory)
        energy_directory.mkdir(parents=True)
        for name in ("atom", "descriptor", "e_corr_target"):
            np.save(energy_directory / f"{name}.npy", arrays[name])
        np.save(
            energy_directory / "converged.npy",
            np.ones(arrays["descriptor"].shape[0], dtype=np.bool_),
        )
        contracts[split] = {
            "object": contract,
            "directory": directory,
            "energy_directory": energy_directory,
            "manifest_fingerprint": contract.manifest_fingerprint,
            "compatibility_fingerprint": contract.compatibility_fingerprint,
            "dimensions": contract.dimensions,
        }
    fingerprints = {
        item["compatibility_fingerprint"] for item in contracts.values()
    }
    if len(fingerprints) != 1:
        raise RuntimeError(f"{workflow} split compatibility fingerprints differ")
    return contracts, generation_times


def _metric_values(target: np.ndarray, prediction: np.ndarray) -> dict:
    difference = np.asarray(prediction) - np.asarray(target)
    return {
        "mae": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "max_abs": float(np.max(np.abs(difference), initial=0.0)),
    }


def _predict_dataset(model, directory: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    contract, arrays = load_force_dataset(directory)
    prediction = predict_correction(
        model,
        torch.from_numpy(arrays["descriptor"].copy()),
        dq_dR_relaxed=torch.from_numpy(arrays["dq_dR_relaxed"].copy()),
        require_force=True,
    )
    return (
        prediction.energy.detach().cpu().numpy(),
        prediction.force.detach().cpu().numpy(),
        arrays,
    )


def _train_models(workflow: str, contracts: dict) -> dict:
    config = load_config()
    training_config = config["training"]
    root = OUTPUT_DIR / "training" / workflow
    seed = int(config["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    energy_train = GroupReader(
        [contracts["train"]["energy_directory"]],
        batch_size=contracts["train"]["dimensions"]["n_frame"],
    )
    energy_validation = GroupReader(
        [contracts["validation"]["energy_directory"]],
        batch_size=contracts["validation"]["dimensions"]["n_frame"],
    )
    energy_model = zero_trainable_model()
    energy_checkpoint = root / "energy_only.pth"
    energy_started = time.perf_counter()
    energy_result = train(
        energy_model,
        energy_train,
        n_epoch=int(training_config[f"{workflow}_epochs"]),
        test_reader=energy_validation,
        energy_factor=1.0,
        force_factor=0.0,
        start_lr=float(training_config["start_lr"]),
        decay_steps=max(1, int(training_config[f"{workflow}_epochs"]) // 2),
        decay_rate=0.3,
        display_epoch=int(training_config[f"{workflow}_epochs"]),
        ckpt_file=energy_checkpoint,
        device="cpu",
    )
    energy_wall = time.perf_counter() - energy_started

    torch.manual_seed(seed)
    np.random.seed(seed)
    force_train = GroupReader(
        [contracts["train"]["directory"]],
        batch_size=contracts["train"]["dimensions"]["n_frame"],
        force_mode=FORCE_MODE_DEEPHF_RELAXED,
    )
    force_validation = GroupReader(
        [contracts["validation"]["directory"]],
        batch_size=contracts["validation"]["dimensions"]["n_frame"],
        force_mode=FORCE_MODE_DEEPHF_RELAXED,
    )
    force_model = zero_trainable_model()
    force_checkpoint = root / "energy_force.pth"
    force_started = time.perf_counter()
    force_result = train(
        force_model,
        force_train,
        n_epoch=int(training_config[f"{workflow}_epochs"]),
        test_reader=force_validation,
        energy_factor=float(training_config["energy_factor"]),
        force_factor=float(training_config["force_factor"]),
        start_lr=float(training_config["start_lr"]),
        decay_steps=max(1, int(training_config[f"{workflow}_epochs"]) // 2),
        decay_rate=0.3,
        display_epoch=int(training_config[f"{workflow}_epochs"]),
        ckpt_file=force_checkpoint,
        device="cpu",
        force_contract=contracts["train"]["object"],
    )
    force_wall = time.perf_counter() - force_started

    loaded_energy = CorrNet.load(energy_checkpoint, strict=True).double().eval()
    loaded_force = CorrNet.load(
        force_checkpoint,
        require_force_metadata=True,
        expected_force_contract=contracts["heldout"]["object"],
    ).double().eval()
    heldout_directory = contracts["heldout"]["directory"]
    zero_energy, zero_force, heldout_arrays = _predict_dataset(
        deterministic_model(scale=0.0), heldout_directory
    )
    energy_energy, energy_force, _ = _predict_dataset(
        loaded_energy, heldout_directory
    )
    force_energy, force_force, _ = _predict_dataset(
        loaded_force, heldout_directory
    )
    target_energy = heldout_arrays["e_corr_target"]
    target_force = heldout_arrays["f_corr_target"]
    comparisons = {
        "zero_correction_rhf": {
            "energy": _metric_values(target_energy, zero_energy),
            "force": _metric_values(target_force, zero_force),
        },
        "energy_only": {
            "energy": _metric_values(target_energy, energy_energy),
            "force": _metric_values(target_force, energy_force),
        },
        "energy_plus_force": {
            "energy": _metric_values(target_energy, force_energy),
            "force": _metric_values(target_force, force_force),
        },
    }

    before_energy, before_force, _ = _predict_dataset(
        force_result.model.eval(), heldout_directory
    )
    checkpoint_exact = bool(
        np.array_equal(before_energy, force_energy)
        and np.array_equal(before_force, force_force)
    )
    saved_result = saved_data_main(
        [str(heldout_directory)],
        model_file=str(force_checkpoint),
        output_prefix=None,
        force_mode=FORCE_MODE_DEEPHF_RELAXED,
    )[0]
    restart_checkpoint = root / "energy_force_restart.pth"
    restart_result = train(
        loaded_force,
        force_train,
        n_epoch=1,
        test_reader=force_validation,
        energy_factor=float(training_config["energy_factor"]),
        force_factor=float(training_config["force_factor"]),
        start_lr=1.0e-5,
        display_epoch=1,
        ckpt_file=restart_checkpoint,
        device="cpu",
        force_contract=contracts["train"]["object"],
    )
    CorrNet.load(
        restart_checkpoint,
        require_force_metadata=True,
        expected_force_contract=contracts["train"]["object"],
    )
    return {
        "models": {
            "energy_only": loaded_energy,
            "energy_plus_force": loaded_force,
        },
        "heldout_arrays": heldout_arrays,
        "comparisons": comparisons,
        "checkpoint_reload_exact": checkpoint_exact,
        "saved_data_test": {
            "energy_mae": saved_result.energy_mae,
            "energy_rmse": saved_result.energy_rmse,
            "force_mae": saved_result.force_mae,
            "force_rmse": saved_result.force_rmse,
        },
        "restart": {
            "checkpoint": str(restart_checkpoint),
            "sha256": sha256_file(restart_checkpoint),
            "training_energy_rmse": restart_result.training_metrics.energy_rmse,
            "training_force_rmse": restart_result.training_metrics.force_rmse,
        },
        "training": {
            "energy_only_wall_seconds": energy_wall,
            "energy_plus_force_wall_seconds": force_wall,
            "energy_only_seconds_per_epoch": energy_wall / int(training_config[f"{workflow}_epochs"]),
            "energy_plus_force_seconds_per_epoch": force_wall / int(training_config[f"{workflow}_epochs"]),
            "energy_checkpoint": str(energy_checkpoint),
            "force_checkpoint": str(force_checkpoint),
            "energy_checkpoint_sha256": sha256_file(energy_checkpoint),
            "force_checkpoint_sha256": sha256_file(force_checkpoint),
            "final_training_energy_rmse": force_result.training_metrics.energy_rmse,
            "final_training_force_rmse": force_result.training_metrics.force_rmse,
            "final_validation_energy_rmse": force_result.validation_metrics.energy_rmse,
            "final_validation_force_rmse": force_result.validation_metrics.force_rmse,
        },
    }


def _fresh_inference(
    references: list,
    models: dict,
    heldout_slice: slice,
    heldout_arrays: dict,
) -> dict:
    results = {}
    heldout_references = references[heldout_slice]
    for model_name, model in models.items():
        backend_results = {}
        for backend in ("direct", "zvector"):
            correction_energies = []
            correction_forces = []
            total_energies = []
            total_forces = []
            solve_counts = []
            for reference in heldout_references:
                method = make_method(reference, model)
                total_energies.append(float(method.kernel()))
                driver = method.nuc_grad_method(backend=backend).run()
                total_forces.append(-driver.de_full)
                correction_energies.append(float(method.e_corr))
                correction_forces.append(-driver.correction_gradient)
                if backend == "zvector":
                    solve_counts.append(int(driver.adjoint_result.diagnostics.solve_count))
            correction_energies = np.asarray(correction_energies).reshape(-1, 1)
            correction_forces = np.asarray(correction_forces)
            backend_results[backend] = {
                "correction_energy": _metric_values(
                    heldout_arrays["e_corr_target"], correction_energies
                ),
                "correction_force": _metric_values(
                    heldout_arrays["f_corr_target"], correction_forces
                ),
                "total_energy": _metric_values(
                    heldout_arrays["e_target"],
                    np.asarray(total_energies).reshape(-1, 1),
                ),
                "total_force": _metric_values(
                    heldout_arrays["f_target"], np.asarray(total_forces)
                ),
                "solve_counts": solve_counts,
                "total_energies_eh": total_energies,
            }
        direct_energy = np.asarray(backend_results["direct"]["total_energies_eh"])
        zvector_energy = np.asarray(backend_results["zvector"]["total_energies_eh"])
        backend_results["energy_backend_max_abs"] = float(
            np.max(np.abs(direct_energy - zvector_energy), initial=0.0)
        )
        results[model_name] = backend_results
    return results


def run_workflow(workflow: str) -> dict:
    config = load_config()
    atoms, geometries = _water_dimer_geometries()
    references = []
    reference_times = []
    for coordinates in geometries:
        started = time.perf_counter()
        references.append(fresh_reference("rhf", atoms, coordinates))
        reference_times.append(time.perf_counter() - started)
    if workflow == "teacher":
        teacher = deterministic_model(scale=0.5)
        energies, forces, target_diagnostics = _teacher_targets(
            references, teacher
        )
    else:
        energies, forces, target_diagnostics = _rmp2_targets(references)
    contracts, generation_times = _write_splits(
        workflow, references, energies, forces
    )
    training = _train_models(workflow, contracts)
    heldout_slice = _split_slices()["heldout"]
    inference = _fresh_inference(
        references,
        training["models"],
        heldout_slice,
        training["heldout_arrays"],
    )
    comparison = training["comparisons"]
    passed = bool(
        training["checkpoint_reload_exact"]
        and all(
            count == 1
            for model_results in inference.values()
            for count in model_results["zvector"]["solve_counts"]
        )
        and all(
            np.isfinite(value)
            for model_results in comparison.values()
            for metric in model_results.values()
            for value in metric.values()
        )
    )
    if workflow == "teacher":
        passed = bool(
            passed
            and comparison["energy_plus_force"]["energy"]["rmse"]
            < comparison["zero_correction_rhf"]["energy"]["rmse"]
            and comparison["energy_plus_force"]["force"]["rmse"]
            < comparison["zero_correction_rhf"]["force"]["rmse"]
        )
    report = {
        "stage": f"training_{workflow}",
        "passed": passed,
        "workflow": workflow,
        "system": "asymmetric water dimer",
        "configuration": config,
        "environment": environment_metadata(),
        "frame_count": len(references),
        "split_slices": {
            name: [value.start, value.stop]
            for name, value in _split_slices().items()
        },
        "native_reference_wall_seconds": {
            "total": float(np.sum(reference_times)),
            "median_per_frame": float(np.median(reference_times)),
        },
        "target_diagnostics": target_diagnostics,
        "dataset_generation_wall_seconds": generation_times,
        "contracts": {
            name: {
                key: value
                for key, value in item.items()
                if key not in {"object"}
            }
            for name, item in contracts.items()
        },
        "heldout_comparison": comparison,
        "training": training["training"],
        "checkpoint_reload_exact": training["checkpoint_reload_exact"],
        "saved_data_test": training["saved_data_test"],
        "restart": training["restart"],
        "fresh_geometry_inference": inference,
    }
    write_json(REPORT_DIR / f"training_{workflow}.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow", choices=("teacher", "mp2"), required=True)
    arguments = parser.parse_args()
    configure_single_thread()
    lib.num_threads(1)
    try:
        result = run_workflow(arguments.workflow)
    except BaseException as error:
        failure = {
            **report_exception(f"training_{arguments.workflow}", error),
            "environment": environment_metadata(),
        }
        write_json(REPORT_DIR / f"training_{arguments.workflow}.json", failure)
        raise
    print(json.dumps({"stage": result["stage"], "passed": result["passed"]}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())

