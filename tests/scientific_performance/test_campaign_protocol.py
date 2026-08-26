from pathlib import Path
import sys

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = (
    REPOSITORY_ROOT / "validation" / "scientific_performance" / "scripts"
)
sys.path.insert(0, str(SCRIPT_DIR))

import common  # noqa: E402
import worker  # noqa: E402


def test_family_specific_steps_and_scf_controls_are_resolved():
    ordinary = common.workload_by_id("L1-def2-SVP")
    l1_tzvp = common.workload_by_id("L1-def2-TZVP")
    l3 = common.workload_by_id("L3-def2-SVP")

    assert common.finite_difference_steps(ordinary, "rhf") == (1.0e-3, 3.0e-4)
    assert common.finite_difference_steps(ordinary, "rks") == (1.0e-3, 3.0e-4)
    assert common.finite_difference_steps(l1_tzvp, "rhf") == (8.0e-4, 3.0e-4)
    assert common.finite_difference_steps(l1_tzvp, "rks") == (1.0e-3, 3.0e-4)
    assert common.finite_difference_steps(l3, "uhf") == (2.5e-3, 2.0e-3)
    assert common.finite_difference_steps(l3, "uks") == (3.0e-3, 2.0e-3)

    uhf_controls = common.effective_scf_controls(l3, "uhf")
    uks_controls = common.effective_scf_controls(l3, "uks")
    assert uhf_controls["conv_tol"] == 1.0e-12
    assert uhf_controls["conv_tol_grad"] == 1.0e-7
    assert uks_controls["conv_tol"] == 1.0e-12
    assert uks_controls["conv_tol_grad"] == 1.0e-8


def test_effective_family_controls_are_written_to_result(monkeypatch):
    workload = common.workload_by_id("L3-def2-SVP")
    monkeypatch.setattr(common, "environment_metadata", lambda profile, threads: {})

    result = common.base_result(
        "scientific", "deterministic-1t", 1, workload, "uks"
    )

    assert result["numerical_controls"]["scf"]["conv_tol_grad"] == 1.0e-8
    finite_difference = result["numerical_controls"]["finite_difference"]
    assert finite_difference["mode"] == "selected"
    assert finite_difference["steps_bohr"] == (3.0e-3, 2.0e-3)
    assert finite_difference["direction_count"] == 5
    assert len(finite_difference["components"]) == 12


def _step_record(
    step: float,
    axis: int,
    force_error: float,
    descriptor_error: float,
    direction_errors: tuple[float, ...],
) -> dict:
    analytic_gradient = (1.0, 2.0, 3.0)
    descriptor_derivative = np.zeros((1, 2), dtype=np.float64)
    descriptor_derivative[0, 1] = -descriptor_error
    directions = []
    for index, error in enumerate(direction_errors):
        direction = np.zeros((1, 3), dtype=np.float64)
        direction[0, axis] = 1.0
        directions.append(
            {
                "index": index,
                "direction": direction,
                "energy_derivative": analytic_gradient[axis] - error,
            }
        )
    return {
        "step_bohr": step,
        "components": [
            {
                "atom": 0,
                "axis": axis,
                "axis_name": ("x", "y", "z")[axis],
                "energy_derivative": analytic_gradient[axis] - force_error,
                "descriptor_derivative": descriptor_derivative,
            }
        ],
        "directions": directions,
    }


def test_finite_difference_summary_preserves_each_steps_worst_errors():
    finite_difference = {
        "steps": {
            "1.0e-03": _step_record(
                1.0e-3, 0, 4.0e-6, 8.0e-6, (1.0e-6, 2.0e-6, 3.0e-6, 4.0e-6, 5.0e-6)
            ),
            "3.0e-04": _step_record(
                3.0e-4, 1, 9.0e-6, 6.0e-6, (2.0e-6, 3.0e-6, 4.0e-6, 5.0e-6, 8.0e-6)
            ),
        }
    }
    analytic_gradient = np.asarray([[1.0, 2.0, 3.0]])
    explicit_only_gradient = analytic_gradient + 0.1
    relaxed_descriptor = np.zeros((1, 3, 1, 2), dtype=np.float64)

    summary = worker._assess_finite_differences(
        finite_difference,
        analytic_gradient,
        explicit_only_gradient,
        relaxed_descriptor,
    )

    coarse = summary["per_step"]["1.0e-03"]
    fine = summary["per_step"]["3.0e-04"]
    np.testing.assert_allclose(
        coarse["maximum_component_error"], 4.0e-6, rtol=0.0, atol=1.0e-15
    )
    np.testing.assert_allclose(
        coarse["maximum_relaxed_descriptor_error"],
        8.0e-6,
        rtol=0.0,
        atol=1.0e-15,
    )
    assert coarse["worst_descriptor"]["descriptor_index"] == (0, 1)
    assert coarse["direction_count"] == 5
    np.testing.assert_allclose(
        fine["maximum_component_error"], 9.0e-6, rtol=0.0, atol=1.0e-15
    )
    np.testing.assert_allclose(
        fine["maximum_directional_error"], 8.0e-6, rtol=0.0, atol=1.0e-15
    )
    assert fine["worst_component"]["axis_name"] == "y"
    assert fine["worst_direction"]["direction_index"] == 4
    assert summary["maximum_component_error"] == fine["maximum_component_error"]
    assert summary["worst_component"]["step_bohr"] == 3.0e-4
