import numpy as np

from scripts.compare_force_methods import (
    compare_campaigns,
    plot_energy_comparison,
    plot_timing_comparison,
    validate_campaign_pair,
    write_report,
)


def _result(force_mode, *, wall_seconds, initial_temperature=66.7):
    return {
        "system": "gram_01",
        "configuration": "gram_01_rxn000026_p000026_0",
        "formula": "C2H4",
        "atoms": 6,
        "charge": 0,
        "multiplicity": 1,
        "reference_family": "RKS",
        "xc": "B3LYP5",
        "grid_mode": "default",
        "grid_level": 3,
        "small_rho_cutoff": 0.0,
        "basis": "def2-tzvp",
        "model_name": "b3lyp_gram_t1x.pth",
        "force_mode": force_mode,
        "finite_difference_step_bohr": (
            1.0e-4 if force_mode == "central_finite_difference" else None
        ),
        "target_temperature_K": 100.0,
        "initial_temperature_K": initial_temperature,
        "timestep_fs": 0.25,
        "steps": 2,
        "slurm_job_id": "1234",
        "simulated_duration_fs": 0.5,
        "stable_duration_at_1meV_per_atom_fs": 0.5,
        "initial_total_energy_eV": -100.0,
        "final_total_energy_eV": -99.9999,
        "maximum_absolute_drift_meV_per_atom": 0.4,
        "rms_drift_meV_per_atom": 0.2,
        "final_drift_meV_per_atom": 0.1,
        "linear_drift_meV_per_atom_per_fs": 0.01,
        "md_wall_time_seconds": wall_seconds * 0.5,
        "md_wall_seconds_per_simulated_fs": wall_seconds,
    }


def _series(drift):
    return {
        "time_fs": np.asarray([0.0, 0.25, 0.5]),
        "drift_meV_per_atom": np.asarray(drift),
    }


def _campaigns():
    analytic = {
        "gram_01": (
            _result("analytic", wall_seconds=100.0),
            _series([0.0, 0.1, -0.2]),
        )
    }
    numerical = {
        "gram_01": (
            _result("central_finite_difference", wall_seconds=900.0),
            _series([0.0, 0.2, -0.4]),
        )
    }
    return analytic, numerical


def test_force_method_comparison_uses_common_time_grid_and_wall_ratio():
    analytic, numerical = _campaigns()

    rows, series = compare_campaigns(analytic, numerical)

    assert len(rows) == len(series) == 1
    assert rows[0]["common_duration_fs"] == 0.5
    assert rows[0]["analytic_common_max_drift_meV_per_atom"] == 0.2
    assert rows[0]["numerical_common_max_drift_meV_per_atom"] == 0.4
    assert rows[0]["numerical_over_analytic_wall_time_ratio"] == 9.0


def test_force_method_comparison_requires_identical_initial_momenta():
    analytic, numerical = _campaigns()
    numerical["gram_01"][0]["initial_temperature_K"] = 70.0

    with np.testing.assert_raises_regex(ValueError, "same initial momenta"):
        validate_campaign_pair(analytic, numerical)


def test_force_method_comparison_plots_are_written(tmp_path):
    analytic, numerical = _campaigns()
    rows, series = compare_campaigns(analytic, numerical)
    energy = tmp_path / "energy.png"
    timing = tmp_path / "timing.png"

    plot_energy_comparison(rows, series, energy)
    plot_timing_comparison(rows, timing)

    assert energy.stat().st_size > 0
    assert timing.stat().st_size > 0


def test_force_method_report_is_chinese_and_uses_requested_section_order(tmp_path):
    analytic, numerical = _campaigns()
    rows, _series_values = compare_campaigns(analytic, numerical)
    output = tmp_path / "comparison.md"

    write_report(rows, output)

    report = output.read_text(encoding="utf-8")
    task_section = report.index("## 1. 九个模拟任务的属性与计算方法")
    analytic_section = report.index("## 2. 解析梯度")
    comparison_section = report.index("## 3. 解析梯度与数值梯度对照")
    assert task_section < analytic_section < comparison_section
    assert "神经网络确实参与每一帧总势能及其力的计算" in report
