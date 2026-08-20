"""Collect controlled single-threaded performance samples without timing gates."""

from __future__ import annotations

import json
import resource
import sys
import time

import numpy as np
from pyscf import lib
from pyscf.lib import param

from deepks.deephf import generate_rhf_force_frame
from deepks.model.model import CorrNet

from common import (
    OUTPUT_DIR,
    PROJECTOR_BASIS,
    REPORT_DIR,
    VALIDATION_DIR,
    configure_single_thread,
    environment_metadata,
    fresh_reference,
    load_config,
    make_method,
    read_xyz,
    report_exception,
    write_json,
)


def _statistics(samples: list[float]) -> dict:
    values = np.asarray(samples, dtype=np.float64)
    median = float(np.median(values))
    return {
        "raw_seconds": values,
        "median_seconds": median,
        "minimum_seconds": float(np.min(values)),
        "maximum_seconds": float(np.max(values)),
        "median_absolute_deviation_seconds": float(
            np.median(np.abs(values - median))
        ),
    }


def _measure(function, *, warmup: int, repetitions: int):
    for _ in range(warmup):
        function()
    samples = []
    last_result = None
    for _ in range(repetitions):
        started = time.perf_counter()
        last_result = function()
        samples.append(time.perf_counter() - started)
    return _statistics(samples), last_result


def run_performance() -> dict:
    config = load_config()
    performance = config["performance"]
    warmup = int(performance["warmup"])
    repetitions = int(performance["repetitions"])
    atoms, coordinates_angstrom = read_xyz(
        VALIDATION_DIR / "geometries" / "formaldehyde.xyz"
    )
    coordinates = coordinates_angstrom / float(param.BOHR)
    checkpoint = OUTPUT_DIR / "formaldehyde" / "model.pth"
    if not checkpoint.is_file():
        raise RuntimeError("the formaldehyde validation checkpoint is unavailable")
    model = CorrNet.load(checkpoint, strict=True).double().eval()
    central_reference = fresh_reference("rhf", atoms, coordinates)

    native_samples, _ = _measure(
        lambda: fresh_reference("rhf", atoms, coordinates),
        warmup=warmup,
        repetitions=repetitions,
    )

    def descriptor_energy():
        method = make_method(central_reference, model)
        energy = method.kernel()
        descriptor = method.descriptor()
        return energy, descriptor

    descriptor_samples, _ = _measure(
        descriptor_energy, warmup=warmup, repetitions=repetitions
    )

    def direct_response():
        method = make_method(central_reference, model)
        method.kernel()
        return method.response()

    direct_response_samples, response = _measure(
        direct_response, warmup=warmup, repetitions=repetitions
    )

    def adjoint_solve():
        method = make_method(central_reference, model)
        method.kernel()
        return method.adjoint()

    adjoint_samples, adjoint = _measure(
        adjoint_solve, warmup=warmup, repetitions=repetitions
    )

    def complete_gradient(backend: str):
        method = make_method(central_reference, model)
        method.kernel()
        return method.nuc_grad_method(backend=backend).run()

    direct_gradient_samples, direct_driver = _measure(
        lambda: complete_gradient("direct"),
        warmup=warmup,
        repetitions=repetitions,
    )
    zvector_gradient_samples, zvector_driver = _measure(
        lambda: complete_gradient("zvector"),
        warmup=warmup,
        repetitions=repetitions,
    )

    target_energy = np.float64(central_reference.e_tot)
    target_force = np.asarray(
        -central_reference.nuc_grad_method().kernel(), dtype=np.float64
    )
    force_data_samples, force_frame = _measure(
        lambda: generate_rhf_force_frame(
            central_reference,
            projector_basis=PROJECTOR_BASIS,
            e_target=target_energy,
            f_target=target_force,
        ),
        warmup=warmup,
        repetitions=repetitions,
    )

    scanner_seed = make_method(central_reference, model)
    scanner_seed.kernel()

    def scanner_first_frame():
        scanner = scanner_seed.nuc_grad_method(backend="zvector").as_scanner()
        return scanner(coordinates)

    scanner_first_samples, _ = _measure(
        scanner_first_frame, warmup=warmup, repetitions=repetitions
    )
    scanner = scanner_seed.nuc_grad_method(backend="zvector").as_scanner()
    displaced = coordinates.copy()
    displaced[2] += np.asarray([0.01, -0.008, 0.006])
    scanner(coordinates)
    scanner(displaced)
    sequence = [coordinates, displaced] * (repetitions + warmup)
    sequence_index = 0

    def scanner_subsequent_frame():
        nonlocal sequence_index
        value = scanner(sequence[sequence_index])
        sequence_index += 1
        return value

    scanner_subsequent_samples, _ = _measure(
        scanner_subsequent_frame,
        warmup=warmup,
        repetitions=repetitions,
    )

    direct_diagnostics = response.diagnostics
    adjoint_diagnostics = adjoint.diagnostics
    peak_rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    peak_rss_mib = peak_rss / 1024.0
    training_timings = {}
    for workflow in ("teacher", "mp2"):
        path = REPORT_DIR / f"training_{workflow}.json"
        if path.is_file():
            with path.open("r", encoding="utf-8") as stream:
                report = json.load(stream)
            if "training" in report:
                training_timings[workflow] = {
                    key: value
                    for key, value in report["training"].items()
                    if "seconds" in key
                }
    report = {
        "stage": "performance",
        "passed": bool(
            int(adjoint_diagnostics.solve_count) == 1
            and direct_diagnostics.maximum_residual
            <= direct_diagnostics.residual_tolerance
            and adjoint_diagnostics.maximum_solver_residual
            <= adjoint_diagnostics.residual_tolerance
            and np.isfinite(force_frame.arrays["dq_dR_relaxed"]).all()
        ),
        "environment": environment_metadata(),
        "configuration": config,
        "warmup_count": warmup,
        "measured_repetition_count": repetitions,
        "samples": {
            "native_rhf_reference": native_samples,
            "descriptor_and_correction_energy": descriptor_samples,
            "direct_response_construction_and_solve": direct_response_samples,
            "zvector_operator_construction_and_solve": adjoint_samples,
            "complete_direct_gradient": direct_gradient_samples,
            "complete_zvector_gradient": zvector_gradient_samples,
            "scanner_first_frame": scanner_first_samples,
            "scanner_subsequent_frame": scanner_subsequent_samples,
            "force_data_generation_per_frame": force_data_samples,
        },
        "algorithmic_diagnostics": {
            "direct_response_dimension": direct_diagnostics.response_dimension,
            "direct_nuclear_right_hand_sides": int(central_reference.mol.natm * 3),
            "direct_maximum_residual": direct_diagnostics.maximum_residual,
            "direct_residual_tolerance": direct_diagnostics.residual_tolerance,
            "direct_refinement_cycles": direct_diagnostics.refinement_cycles,
            "zvector_response_dimension": adjoint_diagnostics.response_dimension,
            "zvector_solve_count": adjoint_diagnostics.solve_count,
            "zvector_solver": adjoint_diagnostics.solver,
            "zvector_maximum_solver_residual": adjoint_diagnostics.maximum_solver_residual,
            "zvector_residual_tolerance": adjoint_diagnostics.residual_tolerance,
            "direct_gradient_backend": direct_driver.backend,
            "zvector_gradient_backend": zvector_driver.backend,
        },
        "peak_resident_memory_mib": peak_rss_mib,
        "training_timings": training_timings,
    }
    write_json(REPORT_DIR / "performance.json", report)
    return report


def main() -> int:
    configure_single_thread()
    lib.num_threads(1)
    try:
        result = run_performance()
    except BaseException as error:
        failure = {
            **report_exception("performance", error),
            "environment": environment_metadata(),
        }
        write_json(REPORT_DIR / "performance.json", failure)
        raise
    print(json.dumps({"stage": result["stage"], "passed": result["passed"]}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())

